# rebuild_embedding_vec.py
"""
Repopulate `embedding_vec` from the `embedding` BYTEA column.

Why it is ever empty
--------------------
`embedding` (BYTEA) is the source of truth that clusterer.py reads;
`embedding_vec` (pgvector) mirrors it so /search can run similarity inside
Postgres. Normally update_embedding() writes both in one statement so they
cannot drift.

They drift in exactly one situation: a database migration that deliberately
skips the mirror. migrate_to_new_db.py does that — the mirror is ~19 MB of
the ~40 MB of row data, and re-deriving it on the target is free while
transferring it is not.

Idempotent: only rows where embedding_vec IS NULL are touched, so a run that
is interrupted can simply be repeated.

Usage
-----
    python rebuild_embedding_vec.py
    python rebuild_embedding_vec.py --batch 1000
"""

import argparse
import logging
import sys

import numpy as np
import psycopg2.extras

from config import EMBEDDING_DIMENSION
from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=500)
    args = parser.parse_args()

    db = ArticleDatabase()
    conn = db._connect()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM articles "
                    "WHERE embedding IS NOT NULL AND embedding_vec IS NULL")
        todo = cur.fetchone()[0]
    logger.info(f"{todo} rows need embedding_vec")
    if not todo:
        db._release(conn)
        return 0

    done = 0
    while True:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, embedding FROM articles
                WHERE embedding IS NOT NULL AND embedding_vec IS NULL
                LIMIT %s
            """, (args.batch,))
            rows = cur.fetchall()
        if not rows:
            break

        updates = []
        for r in rows:
            vec = np.frombuffer(bytes(r['embedding']), dtype='float32')
            if vec.size != EMBEDDING_DIMENSION:
                # A vector of the wrong width predates a provider change and
                # cannot go into a fixed-width vector column. Skipped rather
                # than failing the whole run; it will be re-embedded by the
                # pipeline under the current model.
                logger.warning(f"  id={r['id']}: {vec.size} dims, expected "
                               f"{EMBEDDING_DIMENSION} — skipped")
                continue
            updates.append(('[' + ','.join(f'{x:.7g}' for x in vec) + ']', r['id']))

        if updates:
            with conn.cursor() as cur:
                psycopg2.extras.execute_batch(
                    cur,
                    'UPDATE articles SET embedding_vec = %s::vector WHERE id = %s',
                    updates, page_size=200)
            conn.commit()

        done += len(rows)
        logger.info(f"  {done}/{todo}")
        if len(rows) < args.batch:
            break

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM articles WHERE embedding_vec IS NOT NULL")
        have = cur.fetchone()[0]
    db._release(conn)
    logger.info(f"\nDone. {have} rows now have embedding_vec.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
