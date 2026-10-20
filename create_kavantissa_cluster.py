# create_kavantissa_cluster.py
"""
Create a curated demonstration cluster for the Kavantissa story: how the
government and the opposition responded to the tusker's death.

Built for the viva demo with the bias-detection component, which compares how
outlets cover the government and the opposition on one story. The pipeline
split this story into many small events (statements, protest, arrests), none
with more than seven articles, so this cluster gathers the political reaction
in one place. It is curated, not pipeline output, and should be presented as
such.

What it contains
----------------
Copies of twelve real articles from six outlets, with their original titles,
bodies, publication dates and images. Nothing is rewritten. The originals stay
in their own clusters, untouched.

How the copies stay separate and stable
---------------------------------------
  url           the original url with `?curated=kavantissa` appended. It still
                opens the real article, but it is a different string, so the
                originals keep their url records and are never re-scraped.
  body_fetched  2. The extractor only reads rows at 0, and the cleaner and
                embedder only rows at 1, so the pipeline never touches these.
                With no embedding they never enter clustering, which is what
                keeps this cluster fixed even though the articles are recent.
                (create_curated_cluster.py relied on old dates for the same
                effect; these articles are inside the 14-day window.)
  cluster_id    set here, as the next free id.

They are not searchable through /search, which needs embeddings; the feed and
the name search find them by title.

Usage
-----
    python create_kavantissa_cluster.py            # show what would be created
    python create_kavantissa_cluster.py --apply
    python create_kavantissa_cluster.py --undo     # delete the copies again
"""

import argparse
import logging
import sys

import psycopg2.extras

from database import ArticleDatabase

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

EVENT_TITLE = "Government and opposition respond to the death of tusker Kavantissa"
CATEGORY = "politics"
URL_MARKER = "curated=kavantissa"
CURATED_FETCH_STATE = 2

# Cover photo for the demo, chosen by hand (not from any of the articles).
# The cluster cover is the newest article's image, so it is set on the copy of
# the newest article only; the original article keeps its own photo.
COVER_IMAGE = ("https://untgtsclbjlgemwljimq.supabase.co/storage/v1/object/"
               "tuskerblog/public/kavan.jpg")

# Original article ids. The newest one (13516, Ada Derana) supplies the
# cluster's cover image, because covers come from the newest article.
SOURCE_IDS = [
    12474,  # hirunews      Government deeply saddened by tusker's death: PM Harini
    12486,  # daily_mirror  Kavantissa's death highlights need to end wildlife killings: PM
    12541,  # lankanewsweb  Government vows tougher wildlife protection measures
    13384,  # lankanewsweb  Tusker succumbed to gunshot wounds: Minister
    13444,  # helakuru      Prepared to take primary responsibility - Environment Minister
    11869,  # daily_mirror  Get foreign experts to assist in the post-mortem: Sajith
    12418,  # hirunews      Former President Mahinda Rajapaksa pays tribute
    13344,  # hirunews      Protesters demand the Environment Minister's resignation
    13387,  # helakuru      Protest held in front of the Ministry of Environment
    12703,  # adaderana     Wildlife ranger vows justice for Kawanthissa
    12834,  # newsfirst     SLPP co-op director remanded over the death
    13516,  # adaderana     Multiple investigations underway (Cabinet spokesman)
]


def _marked(url: str) -> str:
    return url + ("&" if "?" in url else "?") + URL_MARKER


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='write the new cluster')
    p.add_argument('--undo', action='store_true', help='delete the copies')
    args = p.parse_args()

    db = ArticleDatabase()
    conn = db._connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id, cluster_id FROM articles WHERE url LIKE %s",
                    ('%' + URL_MARKER,))
        existing = cur.fetchall()

        if args.undo:
            if not existing:
                logger.info("Nothing to undo: no Kavantissa copies found.")
                db._release(conn)
                return 0
            cids = sorted({r['cluster_id'] for r in existing})
            cur.execute("DELETE FROM articles WHERE url LIKE %s", ('%' + URL_MARKER,))
            gone = cur.rowcount
            cur.execute("DELETE FROM cluster_labels WHERE cluster_id = ANY(%s)", (cids,))
            conn.commit()
            db._release(conn)
            logger.info(f"Deleted {gone} copies and the label for cluster(s) {cids}.")
            return 0

        if existing:
            cids = sorted({r['cluster_id'] for r in existing})
            logger.warning(f"{len(existing)} copies already exist in cluster(s) {cids}. "
                           "Run with --undo first to rebuild.")
            db._release(conn)
            return 1

        cur.execute("""SELECT id, title, COALESCE(body_cleaned, body) AS body, url,
                              published_at, source, image_url
                       FROM articles WHERE id = ANY(%s)""", (SOURCE_IDS,))
        originals = {r['id']: r for r in cur.fetchall()}
        missing = [i for i in SOURCE_IDS if i not in originals]
        if missing:
            logger.error(f"Source articles not found: {missing}")
            db._release(conn)
            return 1

        # Next free id; the pipeline mints new ids the same way, so it will
        # start above this one.
        cur.execute("SELECT COALESCE(MAX(cluster_id), -1) + 1 AS n FROM articles")
        new_cid = cur.fetchone()['n']

        outlets = sorted({o['source'] for o in originals.values()})
        logger.info(f"new cluster id : {new_cid}")
        logger.info(f"event title    : {EVENT_TITLE}")
        logger.info(f"articles       : {len(SOURCE_IDS)} from {len(outlets)} outlets ({', '.join(outlets)})")
        for i in SOURCE_IDS:
            o = originals[i]
            logger.info(f"  {o['published_at']:%Y-%m-%d %H:%M}  {o['source']:<13} {o['title'][:70]}")

        if not args.apply:
            logger.info("\nDRY RUN: nothing written. Re-run with --apply.")
            db._release(conn)
            return 0

        newest = max(SOURCE_IDS, key=lambda i: originals[i]['published_at'])
        for i in SOURCE_IDS:
            o = originals[i]
            image = COVER_IMAGE if i == newest else o['image_url']
            cur.execute("""
                INSERT INTO articles
                    (title, body, body_cleaned, url, published_at, source,
                     category, image_url, body_fetched, cluster_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (o['title'], o['body'], o['body'], _marked(o['url']), o['published_at'],
                  o['source'], CATEGORY, image, CURATED_FETCH_STATE, new_cid))
        cur.execute("""INSERT INTO cluster_labels (cluster_id, event_title)
                       VALUES (%s, %s)
                       ON CONFLICT (cluster_id) DO UPDATE SET event_title = EXCLUDED.event_title""",
                    (new_cid, EVENT_TITLE))
    conn.commit()
    db._release(conn)
    logger.info(f"\nCreated cluster {new_cid} with {len(SOURCE_IDS)} articles.")
    logger.info(f"  GET /clusters/{new_cid}")
    logger.info("  python create_kavantissa_cluster.py --undo   # to remove")
    return 0


if __name__ == '__main__':
    sys.exit(main())
