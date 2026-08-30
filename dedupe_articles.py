# dedupe_articles.py
"""
Collapse articles that were stored more than once, and recover the bodies
their duplication destroyed.

Why they exist
--------------
Every scraper method called db.url_exists() with whatever url the page gave
it — some stripped the trailing slash, most did not. Outlets also serve one
story under several paths:

    asianmirror     /news/slug          and  /news/slug/
    economynext     ...-281250          and  ...-281250/
    daily_ft        /columns/X/4-795843 and  /opinion/X/14-795843
    daily_mirror    /breaking-news/X    and  /recomended-news/X
    adaderana       /news/<cuid>, /news/122298/<slug>, news.php?nid=122298

Measured: 485 duplicate rows, ~6% of the corpus.

Why they cost more than storage
-------------------------------
extractor.py discards a body that appears twice from one source in a batch,
because that is the signature of a sidebar or featured-article block being
extracted instead of the real one. A genuine duplicate row is indistinguishable
from that: two copies, one body, both thrown away.

That is where the "extraction failures" came from. Of the empty bodies in the
last 60 days, the share that have a duplicate twin:

    daily_mirror  93.3%      daily_ft      88.8%      economynext  44.2%

Nothing was wrong with the extractor. It was working exactly as designed on
input that should never have reached it.

They also feed HDBSCAN 485 pairs of near-identical vectors. Density is the one
thing that algorithm keys on, so a duplicate is a guaranteed dense pair — an
artificial reason to form a cluster.

What this does
--------------
Per (source, title) group it keeps ONE row and blanks the rest, exactly the
way prune_old_articles.py does: heavy columns cleared, url stub kept.

The stub matters. Deleting the row deletes the record that the url was ever
seen, and url_exists() is the only thing stopping the source serving it
straight back — measured the hard way when a hard delete had the next run
report "business_today: 50 found, 50 new".

If the surviving row has no body — the usual case, since duplication is what
destroyed it — body_fetched is reset to 0 so the extractor picks it up again.
By then it is the only copy in the batch, so the duplicate check leaves it
alone and the real body finally lands.

Usage
-----
    python dedupe_articles.py                 # show what would change
    python dedupe_articles.py --apply
    python dedupe_articles.py --apply --vacuum-full
"""

import argparse
import logging
import sys

import psycopg2.extras

from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# A headline appearing on this many separate days is a template, not a story:
# helakuru runs "Today's weather forecast" daily (9 dates in the corpus),
# adaderana "Fairly heavy rainfall expected in some parts of the island".
# Those are genuinely different articles that share a title, and collapsing
# them would delete real news — the exact mistake this guard exists to stop.
# They are excluded wholesale; no duplicate worth recovering has a title that
# recurs across a week.
_RECURRING_TITLE_DATES = 3

# Ranks the row to keep: a real body first, then the one with an embedding,
# then a cluster assignment, then the oldest id. Keeping the row that already
# has a body means no re-extraction is needed for that group at all.
#
# Grouping is per PUBLISH DATE, not a rolling window. A rolling window is what
# save_article() enforces going forward, where it only ever sees one candidate
# at a time; applied to a bulk update it merges a Monday story with a Wednesday
# one that happens to share a headline. Same-day is the conservative reading,
# and every real duplicate measured here carries the same publication date.
_SURVIVOR = """
    SELECT id,
           row_number() OVER (
               PARTITION BY source, lower(trim(title)),
                            COALESCE(published_at, scraped_at)::date
               ORDER BY (body IS NOT NULL AND body <> '') DESC,
                        (embedding IS NOT NULL) DESC,
                        (cluster_id >= 0) DESC,
                        id ASC
           ) AS rn,
           count(*) OVER (
               PARTITION BY source, lower(trim(title)),
                            COALESCE(published_at, scraped_at)::date
           ) AS grp,
           source, title, body
    FROM articles a
    WHERE title IS NOT NULL AND trim(title) <> ''
      AND NOT EXISTS (
          SELECT 1 FROM articles b
          WHERE b.source = a.source
            AND lower(trim(b.title)) = lower(trim(a.title))
          HAVING count(DISTINCT COALESCE(b.published_at, b.scraped_at)::date)
                 >= %s
      )
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help='write the changes (default is a dry run)')
    parser.add_argument('--vacuum-full', action='store_true',
                        help='also shrink the table file. Takes an EXCLUSIVE '
                             'lock — run it between pipeline cycles.')
    args = parser.parse_args()

    db = ArticleDatabase()
    conn = db._connect()
    w = (_RECURRING_TITLE_DATES,)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"""
            WITH r AS ({_SURVIVOR})
            SELECT count(*) FILTER (WHERE rn > 1)                    AS losers,
                   count(DISTINCT source || '|' || lower(trim(title)))
                       FILTER (WHERE grp > 1)                        AS groups,
                   count(*) FILTER (WHERE rn = 1 AND grp > 1
                                    AND (body IS NULL OR body = '')) AS requeue
            FROM r
        """, w)
        plan = cur.fetchone()

        if not plan['losers']:
            db._release(conn)
            logger.info("No duplicates found.")
            return 0

        logger.info(f"duplicate groups        : {plan['groups']}")
        logger.info(f"rows to blank           : {plan['losers']}")
        logger.info(f"bodies to re-extract    : {plan['requeue']}")

        cur.execute(f"""
            WITH r AS ({_SURVIVOR})
            SELECT source,
                   count(*) FILTER (WHERE rn > 1) AS n,
                   count(*) FILTER (WHERE rn = 1 AND grp > 1
                                    AND (body IS NULL OR body = '')) AS requeue
            FROM r WHERE grp > 1 GROUP BY 1 ORDER BY 2 DESC
        """, w)
        logger.info("")
        logger.info("  source                  blank  re-extract")
        for r in cur.fetchall():
            logger.info(f"     {r['source']:<22} {r['n']:>4}  {r['requeue']:>9}")

        if not args.apply:
            db._release(conn)
            logger.info("")
            logger.info("DRY RUN — nothing changed. Re-run with --apply.")
            return 0

        # Losers become url stubs: everything heavy cleared, body_fetched left
        # at 1 so the extractor never revisits them, url kept so the scraper
        # never re-adds them.
        cur.execute(f"""
            WITH r AS ({_SURVIVOR})
            UPDATE articles a
               SET body = '', body_cleaned = '', embedding = NULL,
                   embedding_vec = NULL, cluster_id = NULL, body_fetched = 1
              FROM r
             WHERE a.id = r.id AND r.rn > 1
        """, w)
        blanked = cur.rowcount

        # Survivors whose body was destroyed by the duplication go back into
        # the extraction queue. They are the only copy now, so the duplicate
        # check that discarded them last time will not fire again.
        cur.execute(f"""
            WITH r AS ({_SURVIVOR})
            UPDATE articles a
               SET body_fetched = 0
              FROM r
             WHERE a.id = r.id AND r.rn = 1 AND r.grp > 1
               AND (a.body IS NULL OR a.body = '')
        """, w)
        requeued = cur.rowcount

        cur.execute("""
            DELETE FROM cluster_labels cl
            WHERE NOT EXISTS (SELECT 1 FROM articles a
                              WHERE a.cluster_id = cl.cluster_id)
        """)
        labels = cur.rowcount
    conn.commit()

    conn.set_isolation_level(0)
    with conn.cursor() as cur:
        if args.vacuum_full:
            logger.info("")
            logger.info("VACUUM FULL — the table is locked until this finishes...")
            cur.execute('VACUUM (FULL, ANALYZE) articles')
        else:
            cur.execute('VACUUM (ANALYZE) articles')
    conn.set_isolation_level(1)
    db._release(conn)

    logger.info("")
    logger.info(f"Blanked {blanked} duplicate rows (url stubs kept).")
    logger.info(f"Requeued {requeued} articles for body extraction.")
    logger.info(f"Removed {labels} cluster labels left with no members.")
    logger.info("The requeued bodies land on the next pipeline run.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
