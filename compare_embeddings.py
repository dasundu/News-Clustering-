# compare_embeddings.py
"""
Re-derive every similarity threshold when the embedding provider changes.

Why this exists
---------------
Cosine similarity is not comparable across embedding models. Measured on
this project's own corpus, two *unrelated* articles score around 0.2 under
the local MiniLM model but around 0.70 under Gemini — which is above
CLUSTER_MATCH_THRESHOLD's current value of 0.60. Carrying the old constants
over would make everything match everything and collapse clustering into a
single blob, silently, with no error.

What it does
------------
1. Samples articles that still carry local (BYTEA) embeddings.
2. Re-embeds the exact same prepared text through the configured provider,
   so both spaces describe identical inputs.
3. Reports how each space separates same-event pairs from unrelated pairs —
   the real quality signal, independent of any threshold.
4. For each tuned constant, finds the percentile it occupies in the old
   space and reports the value at that same percentile in the new one. That
   is a principled starting point, not a guess — though it is still a
   starting point, and the honest fix remains a labeled evaluation set.

Usage
-----
    python compare_embeddings.py [--sample 500]

Costs roughly $0.01 per 500 articles against the Gemini API.
"""

import argparse
import logging
import sys

import numpy as np
import psycopg2.extras

import embedding_provider
from config import (
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL,
    CLUSTER_MATCH_THRESHOLD, CLUSTER_MIN_COHERENCE,
    CLUSTER_MERGE_SIMILARITY_THRESHOLD,
    TOPIC_SIMILARITY_THRESHOLD, TOPIC_MATCH_THRESHOLD,
    SEARCH_MIN_SCORE_HIGH, SEARCH_MIN_SCORE_LOW,
)
from database import ArticleDatabase
from embedder import ArticleEmbedder

logger = logging.getLogger(__name__)

# Every constant that lives in similarity space and therefore moves when the
# provider changes. classifier._MARGIN is a *difference* between two
# similarities rather than an absolute one, so it is reported separately.
THRESHOLDS = [
    ('CLUSTER_MATCH_THRESHOLD', CLUSTER_MATCH_THRESHOLD, 'config.py'),
    ('CLUSTER_MIN_COHERENCE', CLUSTER_MIN_COHERENCE, 'config.py'),
    ('CLUSTER_MERGE_SIMILARITY_THRESHOLD', CLUSTER_MERGE_SIMILARITY_THRESHOLD, 'config.py'),
    ('TOPIC_SIMILARITY_THRESHOLD', TOPIC_SIMILARITY_THRESHOLD, 'config.py'),
    ('TOPIC_MATCH_THRESHOLD', TOPIC_MATCH_THRESHOLD, 'config.py'),
    ('SEARCH_MIN_SCORE_HIGH', SEARCH_MIN_SCORE_HIGH, 'config.py / .env'),
    ('SEARCH_MIN_SCORE_LOW', SEARCH_MIN_SCORE_LOW, 'config.py / .env'),
]


def _fetch_sample(db, limit):
    """Articles that already have a local vector, plus the fields needed to
    rebuild the exact text that was embedded."""
    conn = db._connect()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, title, COALESCE(body_cleaned, '') AS body_cleaned,
                   embedding, cluster_id
            FROM articles
            WHERE embedding IS NOT NULL
              AND cluster_id IS NOT NULL AND cluster_id >= 0
            ORDER BY id DESC
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
    db._release(conn)
    return [dict(r) for r in rows]


def _pair_similarities(matrix, cluster_ids, rng, n_pairs=20000):
    """Cosine similarity for same-cluster pairs and for random pairs.
    Vectors are unit-length, so a dot product is the cosine."""
    n = len(matrix)
    same, rand = [], []

    by_cluster = {}
    for i, cid in enumerate(cluster_ids):
        by_cluster.setdefault(cid, []).append(i)
    for members in by_cluster.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                same.append(float(matrix[members[a]] @ matrix[members[b]]))

    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        if i != j and cluster_ids[i] != cluster_ids[j]:
            rand.append(float(matrix[i] @ matrix[j]))

    return np.array(same), np.array(rand)


def _auc(same, rand):
    """P(a random same-event pair scores above a random unrelated pair).
    Scale-invariant, which a raw mean difference is not — Gemini compresses
    every similarity into a narrow high band, so comparing absolute gaps
    between models is meaningless."""
    labels = np.concatenate([np.ones(len(same)), np.zeros(len(rand))])
    scores = np.concatenate([same, rand])
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = labels.sum(), len(labels) - labels.sum()
    return (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _cohens_d(same, rand):
    pooled = np.sqrt(
        ((len(same) - 1) * same.var(ddof=1) + (len(rand) - 1) * rand.var(ddof=1))
        / (len(same) + len(rand) - 2)
    )
    return (same.mean() - rand.mean()) / max(pooled, 1e-9)


def _describe(name, same, rand):
    auc, d = _auc(same, rand), _cohens_d(same, rand)
    print(f"  {name}")
    print(f"    same-event pairs   n={len(same):<6} mean={same.mean():.4f}  "
          f"sd={same.std():.4f}  p10={np.percentile(same,10):.4f}")
    print(f"    unrelated pairs    n={len(rand):<6} mean={rand.mean():.4f}  "
          f"sd={rand.std():.4f}  p90={np.percentile(rand,90):.4f}")
    print(f"    AUC {auc:.4f}   Cohen's d {d:.3f}   "
          f"(scale-invariant; a raw mean gap is not)")
    return auc, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=500,
                    help='articles to compare (default 500, ~$0.01 via Gemini)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    if EMBEDDING_PROVIDER == 'local':
        print("EMBEDDING_PROVIDER=local — nothing to compare against. "
              "Set it to gemini to derive new thresholds.")
        return 1

    db = ArticleDatabase()
    rows = _fetch_sample(db, args.sample)
    if len(rows) < 50:
        print(f"Only {len(rows)} embedded+clustered articles found; need at least 50.")
        return 1

    old = np.stack([np.frombuffer(bytes(r['embedding']), dtype='float32') for r in rows])
    old = old / np.maximum(np.linalg.norm(old, axis=1, keepdims=True), 1e-9)
    cluster_ids = [r['cluster_id'] for r in rows]

    texts = [ArticleEmbedder._build_text(r) for r in rows]
    print(f"Embedding {len(texts)} articles through {embedding_provider.model_id()} ...")
    new = embedding_provider.encode(texts, task='document')

    rng = np.random.default_rng(42)
    print()
    print("=" * 72)
    print(f"SPACE COMPARISON  ({len(rows)} articles)")
    print("=" * 72)
    old_same, old_rand = _pair_similarities(old, cluster_ids, rng)
    new_same, new_rand = _pair_similarities(new, cluster_ids, rng)
    old_auc, old_d = _describe(
        f"OLD  local:{LOCAL_EMBEDDING_MODEL}  dim={old.shape[1]}", old_same, old_rand)
    print()
    new_auc, new_d = _describe(
        f"NEW  {embedding_provider.model_id()}  dim={new.shape[1]}", new_same, new_rand)

    print()
    print("  READ THIS BEFORE CONCLUDING ANYTHING FROM THE NUMBERS ABOVE:")
    print("  the 'same-event' pairs come from cluster assignments the OLD model")
    print("  produced, so the old model is being graded on its own output. That")
    print("  biases the comparison in its favour and makes it a self-consistency")
    print("  check, NOT evidence that either model clusters better. Deciding that")
    print("  needs a hand-labeled sample.")
    print()
    print(f"  What it does establish: both spaces separate strongly "
          f"(AUC {old_auc:.3f} vs {new_auc:.3f}, d {old_d:.2f} vs {new_d:.2f}),")
    print("  so the new provider is viable — and the thresholds below must change.")

    print()
    print("=" * 72)
    print("RECOMMENDED THRESHOLDS  (value at the same percentile of unrelated pairs)")
    print("=" * 72)
    print(f"  {'constant':<38} {'old':>6}  {'pct':>6}  {'new':>6}")
    print(f"  {'-'*38} {'-'*6}  {'-'*6}  {'-'*6}")
    for name, value, where in THRESHOLDS:
        pct = float((old_rand < value).mean() * 100)
        suggested = float(np.percentile(new_rand, pct)) if 0 < pct < 100 else float('nan')
        print(f"  {name:<38} {value:>6.2f}  {pct:>5.1f}%  {suggested:>6.2f}")

    # _adaptive_epsilon clamps a EUCLIDEAN distance over unit vectors, so it
    # has to be mapped through the distance distribution, not the similarity
    # one: d = sqrt(2 - 2*cos).
    old_d_all = np.sqrt(np.maximum(0.0, 2 - 2 * np.concatenate([old_same, old_rand])))
    new_d_all = np.sqrt(np.maximum(0.0, 2 - 2 * np.concatenate([new_same, new_rand])))
    print()
    print("  _adaptive_epsilon clamp (clusterer.py, euclidean over unit vectors)")
    for label, value in (('low', 0.15), ('high', 0.45)):
        pct = float((old_d_all < value).mean() * 100)
        suggested = float(np.percentile(new_d_all, pct)) if pct > 0 else value * 0.5
        print(f"    {label:<5} {value:.2f}  ->  {suggested:.3f}   (pct {pct:.1f}% of distances)")

    print()
    print("  These are starting points derived from distribution shape, not")
    print("  measured quality. Validate with a clustering run and, ideally, a")
    print("  labeled sample before trusting them.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
