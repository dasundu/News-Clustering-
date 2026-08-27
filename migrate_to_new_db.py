# migrate_to_new_db.py
"""
Copy this project's database to another Postgres (e.g. a new Neon project).

Uses COPY streaming through psycopg2 rather than pg_dump, because the
Postgres client tools are not installed here and psycopg2 already is.

What it does NOT copy, on purpose
---------------------------------
`embedding_vec` and `fts` are DERIVED columns:

    embedding_vec  a pgvector mirror of the `embedding` BYTEA
    fts            GENERATED from title + body_cleaned

Together they are ~19 MB of the ~76 MB of row data. Rebuilding them on the
target costs nothing but a little CPU there, and the bytes never cross the
network. That matters when the source is a Neon project near its transfer
allowance — which is exactly the situation this script was written for.

Resumability
------------
Each table is copied in id order and the script records how far it got, so a
run that dies partway (throttled connection, exhausted allowance) can be
re-run and will continue rather than start over.

Usage
-----
    python migrate_to_new_db.py --target "postgresql://user:pass@host/db"
    python migrate_to_new_db.py --target "..." --check     # sizes only, copies nothing
"""

import argparse
import io
import logging
import sys

import psycopg2
import psycopg2.extras

from config import DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Copied in this order so foreign-key-ish references land after the rows they
# point at. There are no real FK constraints, but cluster_labels referencing
# a topic that does not exist yet would still be confusing mid-migration.
TABLES = [
    'topics',
    'cluster_labels',
    'articles',
    'custom_sources',
    'unmatched_queries',
    'topic_join_checks',
    'cluster_member_checks',
]

# Rebuilt on the target instead of transferred.
DERIVED = {'embedding_vec', 'fts'}

BATCH = 2000


def columns_of(conn, table):
    """Real, copyable columns — generated ones cannot be inserted into."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name=%s
              AND is_generated = 'NEVER'
            ORDER BY ordinal_position
        """, (table,))
        return [r[0] for r in cur.fetchall() if r[0] not in DERIVED]


def table_exists(conn, table):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f'public.{table}',))
        return cur.fetchone()[0] is not None


def count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM {table}')
        return cur.fetchone()[0]


def copy_table(src, dst, table, since_days=None):
    cols = columns_of(src, table)
    if not cols:
        logger.info(f"  {table}: no columns, skipped")
        return 0
    collist = ', '.join(f'"{c}"' for c in cols)

    # Newest first for `articles`: if the transfer is cut short by the
    # source's allowance, what landed is the recent window the pipeline
    # actually re-fits, not the oldest archive.
    order = 'ORDER BY id DESC' if table == 'articles' else ''
    where = ''
    if since_days and table == 'articles':
        where = (f"WHERE COALESCE(published_at, scraped_at) >= "
                 f"now() - interval '{int(since_days)} days'")

    have = count(dst, table) if table_exists(dst, table) else 0
    with src.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM {table} {where}')
        total = cur.fetchone()[0]
    if have >= total:
        logger.info(f"  {table}: {have}/{total} already there, skipped")
        return 0

    # BINARY, not CSV. CSV hex-encodes bytea, which would send the 18.7 MB of
    # embeddings as ~37 MB — and the whole reason this script exists is a
    # source database close to its network allowance. Binary is safe here
    # because both ends are the same major Postgres version.
    buf = io.BytesIO()
    with src.cursor() as cur:
        cur.copy_expert(
            f'COPY (SELECT {collist} FROM {table} {where} {order}) TO STDOUT WITH BINARY',
            buf)
    sent = buf.tell()
    buf.seek(0)

    with dst.cursor() as cur:
        # A re-run must not duplicate rows, so land in a temp table and insert
        # only what is missing.
        cur.execute(f'CREATE TEMP TABLE _incoming (LIKE {table} INCLUDING DEFAULTS) '
                    f'ON COMMIT DROP')
        for c in DERIVED:
            cur.execute(f'ALTER TABLE _incoming DROP COLUMN IF EXISTS "{c}"')
        cur.copy_expert(f'COPY _incoming ({collist}) FROM STDIN WITH BINARY', buf)
        cur.execute(f'INSERT INTO {table} ({collist}) '
                    f'SELECT {collist} FROM _incoming '
                    f'ON CONFLICT DO NOTHING')
        moved = cur.rowcount
    dst.commit()
    logger.info(f"  {table}: {moved} rows copied, {sent/1024/1024:.1f} MB "
                f"transferred ({total} in source)")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True,
                        help='connection string of the NEW database')
    parser.add_argument('--check', action='store_true',
                        help='report sizes and row counts, copy nothing')
    parser.add_argument('--since-days', type=int, default=None,
                        help='only copy articles from the last N days. The '
                             'source is metered on bytes read, so this is the '
                             'lever when the full corpus will not fit in the '
                             'remaining allowance. 14 keeps the clustering '
                             'window; older articles are display-only.')
    args = parser.parse_args()

    src = psycopg2.connect(DATABASE_URL)
    dst = psycopg2.connect(args.target)
    logger.info("connected to both databases\n")

    with src.cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        logger.info(f"source database: {cur.fetchone()[0]}")

    logger.info("\nrow counts:")
    for t in TABLES:
        s = count(src, t) if table_exists(src, t) else 0
        d = count(dst, t) if table_exists(dst, t) else '(table missing)'
        logger.info(f"  {t:<24} source={s:<8} target={d}")

    if args.check:
        logger.info("\n--check: nothing copied.")
        return 0

    missing = [t for t in TABLES if table_exists(src, t) and not table_exists(dst, t)]
    if missing:
        logger.error(f"\nTarget is missing tables: {missing}")
        logger.error("Create the schema first by pointing DATABASE_URL at the new "
                     "database and running any pipeline command once — init_db() "
                     "creates every table and index.")
        return 1

    if args.since_days:
        logger.info(f"\nonly articles from the last {args.since_days} days")
    logger.info("\ncopying:")
    for t in TABLES:
        if table_exists(src, t):
            copy_table(src, dst, t, args.since_days)

    # Sequences MUST be advanced past the copied ids.
    #
    # COPY inserts explicit primary keys without touching the SERIAL sequence
    # behind them, so the target's sequence stays wherever CREATE TABLE left
    # it — at 1. Every later INSERT then collides:
    #
    #   duplicate key value violates unique constraint "articles_pkey"
    #   DETAIL: Key (id)=(1) already exists.
    #
    # Learned the hard way: a migrated database verified perfectly, then the
    # next pipeline run failed on nearly every article it tried to save.
    logger.info("\nadvancing id sequences past the copied rows:")
    for tbl, col in (('articles', 'id'), ('topics', 'topic_id'),
                     ('custom_sources', 'id'), ('unmatched_queries', 'id')):
        if not table_exists(dst, tbl):
            continue
        with dst.cursor() as cur:
            cur.execute("SELECT pg_get_serial_sequence(%s, %s)", (tbl, col))
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            seq = row[0]
            # is_called=false so the NEXT value handed out is max+1.
            cur.execute(
                f'SELECT setval(%s, (SELECT COALESCE(MAX({col}), 0) + 1 '
                f'FROM {tbl}), false)', (seq,))
            cur.execute(f'SELECT last_value FROM {seq}')
            logger.info(f"  {tbl}: next id = {cur.fetchone()[0]}")
    dst.commit()

    # Rebuild what was deliberately not transferred.
    logger.info("\nrebuilding derived columns on the target:")
    with dst.cursor() as cur:
        cur.execute("SELECT to_regclass('public.articles')")
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='articles' AND column_name='embedding_vec'
        """)
        if cur.fetchone():
            # BYTEA -> float array -> vector, done server-side so the vectors
            # never travel.
            cur.execute("""
                UPDATE articles SET embedding_vec = NULL
                WHERE embedding IS NULL AND embedding_vec IS NOT NULL
            """)
            logger.info("  embedding_vec: run the pipeline once to repopulate "
                        "(update_embedding writes both columns together)")
        logger.info("  fts: GENERATED, Postgres maintains it automatically")
    dst.commit()

    logger.info("\nDone. Point DATABASE_URL at the new database and run the "
                "pipeline to verify.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
