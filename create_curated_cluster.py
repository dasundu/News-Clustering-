# create_curated_cluster.py
"""
Create a NEW event cluster from hand-cleaned copies of existing articles,
leaving the original cluster and its articles completely untouched.

Why a new cluster rather than editing the old one
-------------------------------------------------
The first attempt rewrote articles 2691, 2609 and the rest in place. It
worked, but it replaced real scraped text with edited text, and the scraped
corpus is the evidence this project rests on. Cluster 375 is now back exactly
as the scraper found it, and this script adds a separate, clearly-marked
cluster beside it.

How the copies stay separate
----------------------------
Each copy is a genuine article row, distinguished three ways:

  url           the original url with `?curated=1` appended. It still opens
                the real article, so the link is not a lie — but it is a
                different string, so `url UNIQUE` is satisfied AND
                url_exists() still finds the ORIGINAL url, which is what
                stops the scraper re-fetching it as new.
  body_fetched  1, so the extractor never tries to fetch these.
  cluster_id    set explicitly here, not by the pipeline.

Why the pipeline will not overwrite this cluster
------------------------------------------------
Clustering only re-fits articles inside CLUSTER_WINDOW_DAYS, measured on
COALESCE(published_at, scraped_at). These copies keep the originals'
publication dates of 7-11 August 2026, which is outside that window. The
clusterer never sees them, so it never reassigns them.

That is what makes this stable with no new "pinned cluster" machinery: the
cluster is old enough that the pipeline has already stopped caring about it.
A curated cluster built from THIS WEEK'S articles would need a real pin,
because the next run would re-cluster it — worth remembering before reusing
this script on recent news.

The embedder WILL pick these rows up and embed them, which is wanted: it
makes the cluster searchable and lets the topic pass see it. Embedding an
article never moves it between clusters.

Usage
-----
    python create_curated_cluster.py            # show what would be created
    python create_curated_cluster.py --apply
    python create_curated_cluster.py --undo     # delete the copies again
"""

import argparse
import logging
import sys

import psycopg2.extras

from curated_cluster_data import ARTICLES
from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

EVENT_TITLE = "Unrest at the New Magazine Prison Brought Under Control"
CATEGORY = "politics"

# Appended to each copied url. A query parameter rather than a fragment
# because canonical_url() strips fragments — `#curated` would collapse back
# onto the original url and collide with it. It is also not in the tracking
# list canonical_url() removes, so it survives normalisation.
URL_MARKER = "curated=1"


def _marked(url: str) -> str:
    return url + ("&" if "?" in url else "?") + URL_MARKER


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='write the new cluster')
    p.add_argument('--undo', action='store_true', help='delete the copies')
    p.add_argument('--cluster-id', type=int, default=None,
                   help='force a cluster id (default: next free one)')
    args = p.parse_args()

    db = ArticleDatabase()
    conn = db._connect()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, cluster_id, url FROM articles WHERE url LIKE %s",
                    ('%' + URL_MARKER,))
        existing = cur.fetchall()

        if args.undo:
            if not existing:
                logger.info("Nothing to undo — no curated copies found.")
                db._release(conn)
                return 0
            cids = sorted({r['cluster_id'] for r in existing
                           if r['cluster_id'] is not None})
            cur.execute("DELETE FROM articles WHERE url LIKE %s", ('%' + URL_MARKER,))
            gone = cur.rowcount
            cur.execute("""DELETE FROM cluster_labels cl WHERE NOT EXISTS
                           (SELECT 1 FROM articles a
                            WHERE a.cluster_id = cl.cluster_id)""")
            conn.commit()
            db._release(conn)
            logger.info(f"Deleted {gone} curated copies and the now-empty "
                        f"label(s) for cluster(s) {cids}.")
            return 0

        if existing:
            cids = sorted({r['cluster_id'] for r in existing})
            logger.warning(f"{len(existing)} curated copies already exist, in "
                           f"cluster(s) {cids}.")
            logger.warning("Run with --undo first if you want to rebuild them.")
            db._release(conn)
            return 1

        ids = [a['id'] for a in ARTICLES]
        cur.execute("""SELECT id, source, url, published_at, category
                       FROM articles WHERE id = ANY(%s)""", (ids,))
        originals = {r['id']: r for r in cur.fetchall()}
        missing = [i for i in ids if i not in originals]
        if missing:
            logger.error(f"Source articles not found: {missing}")
            db._release(conn)
            return 1

        if args.cluster_id is not None:
            new_cid = args.cluster_id
        else:
            # Next free id. Taking max+1 also means the pipeline's own
            # minting (which uses the same expression) will start above this
            # one, so the two can never collide.
            cur.execute("SELECT COALESCE(MAX(cluster_id), -1) + 1 AS n FROM articles")
            new_cid = cur.fetchone()['n']

        logger.info(f"new cluster id    : {new_cid}")
        logger.info(f"event title       : {EVENT_TITLE}")
        logger.info(f"copies to create  : {len(ARTICLES)}")
        logger.info(f"originals         : cluster 375, left untouched")
        logger.info("")
        for a in ARTICLES[:3]:
            o = originals[a['id']]
            logger.info(f"  copy of id={a['id']}  ({o['source']})")
            logger.info(f"     {a['title'][:72]}")
            logger.info(f"     {_marked(o['url'])[:80]}")

        if not args.apply:
            logger.info("")
            logger.info("DRY RUN — nothing written. Re-run with --apply.")
            db._release(conn)
            return 0

        created = []
        for a in ARTICLES:
            o = originals[a['id']]
            cur.execute("""
                INSERT INTO articles
                    (title, body, body_cleaned, url, published_at, source,
                     category, image_url, body_fetched, cluster_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 1, %s)
                RETURNING id
            """, (a['title'][:500], a['body'][:50000], a['body'][:50000],
                  _marked(o['url']), o['published_at'], o['source'],
                  CATEGORY, a['image_url'], new_cid))
            created.append(cur.fetchone()['id'])

        cur.execute("""INSERT INTO cluster_labels (cluster_id, event_title)
                       VALUES (%s, %s)
                       ON CONFLICT (cluster_id)
                       DO UPDATE SET event_title = EXCLUDED.event_title""",
                    (new_cid, EVENT_TITLE))
    conn.commit()
    db._release(conn)

    logger.info("")
    logger.info(f"Created cluster {new_cid} with {len(created)} articles "
                f"(ids {created[0]}-{created[-1]}).")
    logger.info("")
    logger.info(f"  GET /clusters/{new_cid}")
    logger.info(f"  python create_curated_cluster.py --undo   # to remove")
    return 0


if __name__ == '__main__':
    sys.exit(main())
