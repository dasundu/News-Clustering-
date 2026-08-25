# backfill_solo_clusters.py
"""
One-time backfill: give every historical noise article its own cluster.

Why this is needed
------------------
Event clustering only ever re-fits the last CLUSTER_WINDOW_DAYS. That is
deliberate — a past event does not change, and the window keeps run time
bounded as the corpus grows. The side effect is that any decision made about
an older article is permanent.

CLUSTER_PROMOTE_SOLO makes single-outlet stories their own clusters so they
can reach a topic. But it only applies to articles inside the window, so
every article that was already outside it stayed at cluster_id = -1 forever.
Measured before this script ran:

    inside  the 14d window   1,464 articles   100% clustered   44% in a topic
    outside the window       2,017 articles    36% clustered   23% in a topic

This script applies the same promotion to the backlog, once.

What it does NOT do
-------------------
It does not re-cluster anything. Existing clusters and their members are left
exactly as they are; this only assigns ids to articles that currently have
none. Topic membership is left to the next pipeline run, which now sees every
cluster (see ClusterTopicBuilder._all_centroids).

Usage
-----
    python backfill_solo_clusters.py            # show what would change
    python backfill_solo_clusters.py --apply    # write it
"""

import argparse
import logging
import sys

import psycopg2.extras

import embedding_provider
from clusterer import _sanitize_label
from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Written in chunks so a failure part-way leaves a consistent prefix rather
# than an all-or-nothing transaction over thousands of rows.
CHUNK = 500


def find_orphans(db: ArticleDatabase) -> list:
    """
    Articles that can carry a topic but have no cluster to carry it.

    The three filters mirror what the topic pass requires: a body (so the
    article is real), an embedding, and the CURRENT embedding model — a
    cluster whose vectors predate a provider switch produces no centroid, so
    promoting it would only recreate the stale assignments this project just
    finished clearing.
    """
    conn = db._connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, title FROM articles
            WHERE (cluster_id IS NULL OR cluster_id = -1)
              AND embedding IS NOT NULL
              AND body IS NOT NULL AND body <> ''
              AND embedding_model = %s
            ORDER BY id
        """, (embedding_provider.model_id(),))
        rows = cur.fetchall()
    db._release(conn)
    return [{'id': r['id'], 'title': r['title']} for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true',
                        help='write the changes (default is a dry run)')
    args = parser.parse_args()

    db = ArticleDatabase()
    orphans = find_orphans(db)
    if not orphans:
        logger.info("Nothing to backfill — every eligible article has a cluster.")
        return 0

    next_id = db.get_max_cluster_id() + 1
    logger.info(f"{len(orphans)} articles have no cluster")
    logger.info(f"They would become clusters {next_id}..{next_id + len(orphans) - 1}")

    # A solo cluster's label is simply its own headline — the same result the
    # labelling tier reaches for a one-member cluster, minus a round trip.
    plan = []
    for offset, art in enumerate(orphans):
        label = _sanitize_label(art['title'] or '')
        if not label:
            continue
        plan.append((next_id + offset, art['id'], label))

    if not args.apply:
        logger.info("\nDRY RUN — nothing written. Re-run with --apply.\nExamples:")
        for cid, _, label in plan[:5]:
            logger.info(f"  cluster {cid}: {label[:66]}")
        return 0

    written = 0
    for start in range(0, len(plan), CHUNK):
        chunk = plan[start:start + CHUNK]
        # Labels first: a cluster_id on an article whose cluster has no label
        # renders as an unnamed event, whereas a label with no members yet is
        # invisible. Ordering it this way keeps every intermediate state sane.
        db.save_cluster_labels_bulk([(cid, label) for cid, _, label in chunk])
        db.update_cluster_ids([(cid, aid) for cid, aid, _ in chunk])
        written += len(chunk)
        logger.info(f"  {written}/{len(plan)}")

    logger.info(f"\nDone. {written} articles promoted to their own clusters.")
    logger.info("Run the pipeline (or clusterer alone) to file them under topics.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
