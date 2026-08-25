# prune_old_articles.py
"""
Retention: drop old articles that no longer earn their storage.

Why this is needed
------------------
Neon's free tier allows 500 MB. Measured at ~18 KB per article once every
column and index is counted, and an intake of ~500 articles/day once general
news stopped being filtered out, the database fills in roughly a month.

How it prunes
-------------
By DEFAULT it does not delete the row — it blanks the heavy columns (body,
body_cleaned, embedding, embedding_vec) and keeps a ~200-byte stub carrying
the url, title, source and date. That is ~98% of the weight gone.

The stub is not sentimentality, it is the duplicate defence. `url UNIQUE` +
`url_exists()` is the ONLY thing stopping a source re-serving an article the
scraper already has. Measured the hard way: a hard delete of 105 old
business_today and lankabusinessonline articles had the very next run report
"business_today: 50 found, 50 new" — the same June articles, straight back,
consuming the per-source cap that should have gone to today's news. Deleting
the row deletes the memory that the article was ever seen.

Blanked rows drop out of everything on their own: clustering needs a body,
re-extraction is gated on body_fetched=0, and re-embedding on a non-empty
body. Use --hard-delete only for articles a source will never list again.

What is pruned
--------------
An article is dropped only if BOTH are true:

  1. It is older than RETENTION_DAYS. Clustering only ever re-fits the last
     CLUSTER_WINDOW_DAYS, so an older article's cluster assignment is already
     final — deleting it changes no future result.

  2. No other outlet covered the same event. Single-outlet clusters cannot
     support cross-outlet comparison, which is the point of the corpus. An
     event two or more outlets covered is kept regardless of age.

The second condition is what makes this safe to run automatically: the
articles that carry the project's actual value are never touched by it.

Usage
-----
    python prune_old_articles.py                       # show what would go
    python prune_old_articles.py --apply               # blank (keeps url stub)
    python prune_old_articles.py --apply --vacuum-full # and shrink the file
    python prune_old_articles.py --apply --hard-delete # remove rows entirely
"""

import argparse
import logging
import sys

import psycopg2.extras

from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

RETENTION_DAYS = 60

# Articles kept because several outlets covered the same event.
_PROTECTED = """
    SELECT cluster_id FROM articles
    WHERE cluster_id >= 0
    GROUP BY cluster_id
    HAVING COUNT(DISTINCT source) >= 2
"""

_TARGET = f"""
    FROM articles a
    WHERE COALESCE(a.published_at, a.scraped_at) < now() - make_interval(days => %s)
      AND a.body IS NOT NULL AND a.body <> ''
      AND (a.cluster_id IS NULL
           OR a.cluster_id < 0
           OR a.cluster_id NOT IN ({_PROTECTED}))
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=RETENTION_DAYS)
    parser.add_argument('--apply', action='store_true',
                        help='delete (default is a dry run)')
    parser.add_argument('--hard-delete', action='store_true',
                        help='remove rows entirely instead of blanking them. '
                             'Drops the url record, so the scraper can and '
                             'will re-fetch anything the source still lists.')
    parser.add_argument('--vacuum-full', action='store_true',
                        help='also shrink the table file on disk. Takes an '
                             'EXCLUSIVE lock, so run it between pipeline '
                             'cycles, not during one.')
    args = parser.parse_args()

    db = ArticleDatabase()
    conn = db._connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            SELECT count(*) n,
                   pg_size_pretty(COALESCE(sum(pg_column_size(a.*)), 0)) sz,
                   min(COALESCE(a.published_at, a.scraped_at))::date oldest,
                   max(COALESCE(a.published_at, a.scraped_at))::date newest
            {_TARGET}
        """, (args.days,))
        target = cur.fetchone()

        cur.execute("SELECT count(*) n, "
                    "pg_size_pretty(pg_total_relation_size('articles')) sz "
                    "FROM articles")
        before = cur.fetchone()

        logger.info(f"articles now      : {before['n']}  ({before['sz']} on disk)")
        logger.info(f"older than {args.days}d and single-outlet: {target['n']}")
        if not target['n']:
            db._release(conn)
            logger.info("Nothing to prune.")
            return 0
        logger.info(f"  row data        : {target['sz']}")
        logger.info(f"  date range      : {target['oldest']} .. {target['newest']}")

        cur.execute(f"SELECT a.source, count(*) n {_TARGET} "
                    f"GROUP BY 1 ORDER BY 2 DESC LIMIT 8", (args.days,))
        logger.info("  by source:")
        for r in cur.fetchall():
            logger.info(f"     {r['source']:<22} {r['n']}")

        cur.execute(f"SELECT a.title {_TARGET} LIMIT 5", (args.days,))
        logger.info("  examples:")
        for r in cur.fetchall():
            logger.info(f"     {r['title'][:66]}")

        if not args.apply:
            db._release(conn)
            logger.info("\nDRY RUN — nothing deleted. Re-run with --apply.")
            return 0

        if args.hard_delete:
            cur.execute(f"DELETE FROM articles WHERE id IN (SELECT a.id {_TARGET})",
                        (args.days,))
        else:
            # body_fetched stays 1 and the url stays put, so neither the
            # extractor nor the scraper will pick this row up again.
            cur.execute(f"""
                UPDATE articles SET body = '', body_cleaned = '',
                       embedding = NULL, embedding_vec = NULL,
                       cluster_id = NULL, body_fetched = 1
                WHERE id IN (SELECT a.id {_TARGET})
            """, (args.days,))
        deleted = cur.rowcount
        # A label with no members left is not a cluster any more. Topics with
        # no clusters are cleaned up by the clusterer's own delete_empty_topics.
        cur.execute("""
            DELETE FROM cluster_labels cl
            WHERE NOT EXISTS (SELECT 1 FROM articles a
                              WHERE a.cluster_id = cl.cluster_id)
        """)
        labels = cur.rowcount
    conn.commit()

    # A DELETE only marks rows dead; the pages stay in the file. Plain VACUUM
    # marks them reusable — new articles fill them instead of growing the
    # file — but the file itself does NOT shrink, so the size reported by the
    # host does not move. Measured: after deleting 1,926 articles the table
    # still read 133 MB.
    #
    # VACUUM FULL rewrites the table into a fresh file and drops the old one,
    # which is what actually returns space (133 MB -> 89 MB, 3 seconds). It
    # is opt-in because it takes an ACCESS EXCLUSIVE lock: every read and
    # write blocks until it finishes, so firing it automatically could freeze
    # a pipeline run mid-cycle.
    conn.set_isolation_level(0)
    with conn.cursor() as cur:
        if args.vacuum_full:
            logger.info("VACUUM FULL - the table is locked until this finishes...")
            cur.execute('VACUUM (FULL, ANALYZE) articles')
        else:
            cur.execute('VACUUM (ANALYZE) articles')
    conn.set_isolation_level(1)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT count(*) n, "
                    "pg_size_pretty(pg_total_relation_size('articles')) sz "
                    "FROM articles")
        after = cur.fetchone()
    db._release(conn)

    if args.hard_delete:
        logger.info(f"\nDeleted {deleted} articles and {labels} empty cluster "
                    f"labels. Their urls are gone too, so a source still "
                    f"listing them will serve them back as new.")
    else:
        logger.info(f"\nBlanked {deleted} articles (url stubs kept so they are "
                    f"never re-scraped) and removed {labels} empty cluster labels.")
    logger.info(f"articles table: {before['sz']} -> {after['sz']}  "
                f"({before['n']} -> {after['n']} rows)")
    if not args.vacuum_full:
        logger.info(
            "\nThe freed space is reusable by new articles, but the file did not "
            "shrink.\nTo hand it back to the host, re-run with --vacuum-full "
            "(locks the table briefly)."
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
