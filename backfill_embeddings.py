# backfill_embeddings.py
"""
Re-embed the whole corpus under the currently configured provider.

Needed after switching EMBEDDING_PROVIDER: vector spaces are not comparable
across models, so every article must be recomputed before clustering can be
trusted. The normal pipeline does this on its own, but only
EMBED_BATCH_SIZE articles per run — this drives it to completion in one go.

Safe to interrupt and safe to re-run: embedder.run() commits incrementally
and selects work by `embedding_model IS DISTINCT FROM <current>`, so each
pass resumes exactly where the last stopped. On a rate-limited free tier it
simply stops early; run it again the next day.
"""

import logging
import sys
import time

import embedding_provider
from database import ArticleDatabase
from embedder import ArticleEmbedder


def pending(db, model_id):
    conn = db._connect()
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM articles WHERE body_fetched=1 "
            "AND (embedding IS NULL OR embedding_model IS DISTINCT FROM %s)",
            (model_id,),
        )
        n = cur.fetchone()[0]
    db._release(conn)
    return n


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    db = ArticleDatabase()
    model_id = embedding_provider.model_id()
    embedder = ArticleEmbedder()

    total = pending(db, model_id)
    print(f"provider={model_id}  pending={total}", flush=True)

    done = 0
    start = time.time()
    while True:
        n = embedder.run()
        if n == 0:
            break                      # finished, or quota stopped us
        done += n
        left = pending(db, model_id)
        rate = done / max(time.time() - start, 1) * 60
        eta = left / rate if rate else 0
        print(f"  {done} embedded | {left} left | {rate:.0f}/min | eta {eta:.0f} min",
              flush=True)
        if left == 0:
            break

    left = pending(db, model_id)
    print(f"\ndone: {done} embedded this pass, {left} still pending", flush=True)
    if left:
        print("Stopped early — almost certainly the free-tier daily quota "
              "(1000/day). Re-run tomorrow, or enable billing to finish now.",
              flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
