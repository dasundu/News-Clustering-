# fix_wrong_bodies.py
"""
Find and repair articles that are storing a DIFFERENT article's text, and
backfill body_fingerprint for the whole corpus.

The failure
-----------
Some sites hand back the wrong article — a featured or promoted story instead
of the one requested. extractor.py has always discarded a body it sees twice
from one source, but that check only looked INSIDE one batch. A site that does
this consistently defeated it: each run saw the bad block once and judged it
unique.

Measured across the corpus: 65 articles storing another article's text, 64 of
them inside live clusters.

Why it is worse than wrong text on a page
-----------------------------------------
Identical text embeds to an identical vector, so the affected articles do not
merely READ wrong — they cluster together. Real example:

    cluster 2696
      "SLANZBC team explores Sri Lanka trade ties in Australia"
      "Sri Lanka sells Rs50bn in 2030 and 2035 bonds"
      "China gives Sri Lanka RMB 79mn to build 14 bridges"

Three unrelated stories, grouped only because all three were handed the same
wrong body. That is a fabricated event, which is the one thing this project
must not produce.

How the right owner is chosen
-----------------------------
Within a group sharing one body, the article whose HEADLINE the body actually
matches keeps it, scored by extractor._title_overlap — the same measure the
rescue path already uses. Everything else in the group is blanked and requeued.

If no member clears _MIN_TITLE_OVERLAP, the body belongs to none of them and
all are blanked. Real case: a body about "New Anthoney's Farms ... US Soy"
filed under both "Gatz Re-Imagined redefines Colombo nightlife" and "Fitch
rates DFCC Bank's Basel III debt" — it is neither.

Blanked rows keep their url (so the scraper never re-fetches them as new) and
go back into the extraction queue with body_fetched=0. If the site serves the
same wrong text again, the cross-run fingerprint check added to extractor.py
now rejects it, and the article correctly ends up with NO body rather than the
wrong one.

Usage
-----
    python fix_wrong_bodies.py                # report only
    python fix_wrong_bodies.py --backfill     # only fill body_fingerprint
    python fix_wrong_bodies.py --apply        # backfill + repair
"""

import argparse
import logging
import sys

import psycopg2.extras

from database import ArticleDatabase
from extractor import _MIN_TITLE_OVERLAP, _fingerprint, _title_overlap

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def backfill(db, conn) -> int:
    """Compute body_fingerprint for every row that has a body but no hash.

    Done in Python, not SQL, so the stored value is byte-identical to what
    extractor._fingerprint() produces at write time. A SQL approximation of
    the same normalisation would drift and the comparison would silently stop
    matching.
    """
    total = 0
    while True:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""SELECT id, body FROM articles
                           WHERE body <> '' AND body_fingerprint IS NULL
                           LIMIT 500""")
            rows = cur.fetchall()
        if not rows:
            break
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                'UPDATE articles SET body_fingerprint = %s WHERE id = %s',
                [(_fingerprint(r['body']), r['id']) for r in rows],
                page_size=200)
        conn.commit()
        total += len(rows)
        logger.info(f"  fingerprinted {total}")
    return total


_GROUPS = """
    WITH bad AS (
        SELECT source, body_fingerprint
        FROM articles
        WHERE body <> '' AND body_fingerprint IS NOT NULL
        GROUP BY source, body_fingerprint
        HAVING COUNT(DISTINCT lower(trim(title))) > 1
    )
    SELECT a.id, a.source, a.title, a.cluster_id, a.body_fingerprint AS fp,
           a.body
    FROM articles a
    JOIN bad b ON b.source = a.source
              AND b.body_fingerprint = a.body_fingerprint
    ORDER BY a.source, a.body_fingerprint, a.id
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='backfill and repair')
    p.add_argument('--backfill', action='store_true',
                   help='only fill body_fingerprint, change nothing else')
    args = p.parse_args()

    db = ArticleDatabase()
    conn = db._connect()

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM articles "
                    "WHERE body <> '' AND body_fingerprint IS NULL")
        pending = cur.fetchone()[0]

    if pending:
        if args.apply or args.backfill:
            logger.info(f"Backfilling body_fingerprint for {pending} articles...")
            backfill(db, conn)
        else:
            logger.info(f"{pending} articles have no body_fingerprint yet.")
            logger.info("Run with --backfill (or --apply) to compute them; the "
                        "report below needs them.")
            db._release(conn)
            return 0
        if args.backfill and not args.apply:
            db._release(conn)
            logger.info("Backfill done. Re-run with --apply to repair.")
            return 0

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_GROUPS)
        rows = cur.fetchall()

    if not rows:
        db._release(conn)
        logger.info("No articles are sharing a body with a different headline.")
        return 0

    groups: dict = {}
    for r in rows:
        groups.setdefault((r['source'], r['fp']), []).append(r)

    keep, blank = [], []
    for (source, _fp), members in groups.items():
        scored = [(_title_overlap(m['title'], m['body'] or ''), m) for m in members]
        scored.sort(key=lambda t: -t[0])
        best_score, best = scored[0]
        if best_score >= _MIN_TITLE_OVERLAP:
            keep.append((best, best_score))
            blank.extend(m for _s, m in scored[1:])
        else:
            # The body matches none of these headlines.
            blank.extend(m for _s, m in scored)

    logger.info("")
    logger.info(f"groups sharing one body : {len(groups)}")
    logger.info(f"articles involved       : {len(rows)}")
    logger.info(f"  keep (headline matches): {len(keep)}")
    logger.info(f"  blank and re-extract   : {len(blank)}")
    logger.info(f"  of those, in clusters  : "
                f"{sum(1 for m in blank if (m['cluster_id'] or -1) >= 0)}")

    by_source: dict = {}
    for m in blank:
        by_source[m['source']] = by_source.get(m['source'], 0) + 1
    logger.info("")
    for src, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        logger.info(f"     {src:<22} {n:>4}")

    if not args.apply:
        logger.info("")
        logger.info("examples:")
        for m in blank[:5]:
            logger.info(f"   id={m['id']:<6} cid={str(m['cluster_id']):>6}  "
                        f"{m['title'][:56]}")
            logger.info(f"           body: {(m['body'] or '')[:64]}")
        logger.info("")
        logger.info("DRY RUN — nothing changed. Re-run with --apply.")
        db._release(conn)
        return 0

    ids = [m['id'] for m in blank]
    with conn.cursor() as cur:
        # url and body_fetched=0 together mean: never re-scraped as a new
        # article, but re-extracted on the next run. The cross-run fingerprint
        # check now rejects the same wrong text if the site serves it again.
        cur.execute("""
            UPDATE articles
               SET body = '', body_cleaned = '', body_fingerprint = NULL,
                   embedding = NULL, embedding_vec = NULL,
                   cluster_id = NULL, body_fetched = 0
             WHERE id = ANY(%s)
        """, (ids,))
        cleared = cur.rowcount
        cur.execute("""DELETE FROM cluster_labels cl WHERE NOT EXISTS
                       (SELECT 1 FROM articles a
                        WHERE a.cluster_id = cl.cluster_id)""")
        labels = cur.rowcount
    conn.commit()
    db._release(conn)

    logger.info("")
    logger.info(f"Blanked {cleared} articles carrying another article's text.")
    logger.info(f"Removed {labels} cluster labels left with no members.")
    logger.info("They are queued for re-extraction on the next pipeline run.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
