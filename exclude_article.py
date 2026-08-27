# exclude_article.py
"""
Remove one article from whatever cluster it gets grouped into.

Why this exists
---------------
Automated removal of wrongly-grouped articles was built, measured against
live data, and rejected. The reason is specific and repeatable: a wrong
article often scores HIGHER than a correct one.

    cluster 2788, "Rumesh Tharanga wins Lausanne Diamond League javelin"
      0.963  MAIN Championship ... Muay Thai at BMICH      <- does not belong
      0.925  Rumesh Tharanga Climbs to World No. 1         <- belongs

No threshold separates those. An LLM judge over the undecidable band was
also tried and made the multi-outlet cluster count worse than doing nothing
(221 -> 206), so it is off.

What is left is targeted human correction: rare, deliberate, reversible.

What it does NOT do
-------------------
It does not delete the article. The row and its url record stay, so the
scraper will not re-fetch it. Clustering simply refuses to group it, and
CLUSTER_PROMOTE_SOLO gives it a cluster of its own — the article is removed
from a group it does not belong to, not from the corpus.

The flag is read on every clustering run, so the correction survives
re-clustering. That is the whole point: editing cluster_id by hand would be
undone by the next pipeline cycle.

Usage
-----
    python exclude_article.py 8985            # exclude
    python exclude_article.py 8985 --undo     # put it back in play
    python exclude_article.py --list
"""

import argparse
import logging
import sys

import psycopg2.extras

from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('article_id', type=int, nargs='?')
    parser.add_argument('--undo', action='store_true',
                        help='clear the exclusion instead of setting it')
    parser.add_argument('--list', action='store_true',
                        help='show every currently excluded article')
    args = parser.parse_args()

    db = ArticleDatabase()

    if args.list:
        conn = db._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, source, title, cluster_id FROM articles "
                        "WHERE cluster_excluded IS TRUE ORDER BY id")
            rows = cur.fetchall()
        db._release(conn)
        if not rows:
            logger.info("No articles are excluded.")
            return 0
        logger.info(f"{len(rows)} excluded article(s):")
        for r in rows:
            logger.info(f"  id={r['id']:<6} [{r['source']}] {r['title'][:60]}")
        return 0

    if args.article_id is None:
        parser.error("give an article id, or --list")

    conn = db._connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, source, title, cluster_id FROM articles WHERE id=%s",
                    (args.article_id,))
        row = cur.fetchone()
    db._release(conn)

    if not row:
        logger.error(f"No article with id {args.article_id}.")
        return 1

    logger.info(f"[{row['source']}] {row['title'][:70]}")
    logger.info(f"currently in cluster {row['cluster_id']}")

    db.set_cluster_excluded(args.article_id, not args.undo)
    if args.undo:
        logger.info("\nException cleared — it can be clustered normally again.")
    else:
        logger.info("\nExcluded. It keeps its own single-article cluster from "
                    "the next clustering run onward.")
    logger.info("Run the pipeline (or clusterer alone) to apply it.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
