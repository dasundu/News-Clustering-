# gold_set.py
"""
Extrinsic evaluation of event clustering against hand-labelled ground truth.

The thesis reports internal validity (coherence, silhouette, separation) and
identity stability, but never checks the clusters against a human judgement of
which articles report the same event. This closes that gap.

Two steps:

  python gold_set.py export
      Writes gold/to_label.csv: every embedded article published on the gold
      days, with the columns needed to judge it. The system's own cluster ids
      are NOT in this file, so they cannot steer the labelling. They are
      frozen separately in gold/system_snapshot.json at export time.

  (label: fill the `event` column; see gold/README.md for the rule)

  python gold_set.py score gold/labelled.csv [--second other_labeller.csv]
      Scores the deployed system and three baselines against the labels:
        system          cluster ids as the pipeline stored them
        hdbscan_raw     plain HDBSCAN on the same embeddings, no refinement
                        pass and no identity matching (an ablation)
        tfidf_kmeans    TF-IDF with K-Means, k chosen by silhouette
                        (the baseline named in the proposal)
        tfidf_kmeans_k* TF-IDF with K-Means given the true number of events,
                        a deliberately generous version of the baseline
        all_singletons  every article its own event, the trivial reference
      With --second, also reports agreement between the two labellers.

Metrics: ARI, NMI, homogeneity, completeness, V-measure, pairwise
precision/recall/F1 and B-cubed precision/recall/F1. Articles the system left
as noise (cluster_id -1 or NULL) count as their own single-article events,
since that is what the system asserts about them.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

GOLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gold')
DEFAULT_DAYS = ('2026-08-21', '2026-08-22')
SNAPSHOT = os.path.join(GOLD_DIR, 'system_snapshot.json')
TO_LABEL = os.path.join(GOLD_DIR, 'to_label.csv')
SNIPPET_CHARS = 300


# ── export ────────────────────────────────────────────────────────────────────

def export(days):
    import embedding_provider
    from database import ArticleDatabase

    db = ArticleDatabase()
    conn = db._connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source, title, COALESCE(body_cleaned, body, '') AS text,
                   COALESCE(published_at, scraped_at) AS at, url, cluster_id
            FROM articles
            WHERE embedding IS NOT NULL AND embedding_model = %s
              AND COALESCE(published_at, scraped_at)::date = ANY(%s::date[])
            ORDER BY COALESCE(published_at, scraped_at), id
            """,
            (embedding_provider.model_id(), list(days)),
        )
        rows = cur.fetchall()
    db._release(conn)
    if not rows:
        sys.exit(f'No embedded articles on {days}.')

    os.makedirs(GOLD_DIR, exist_ok=True)
    with open(TO_LABEL, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['article_id', 'published', 'source', 'title', 'snippet', 'url', 'event'])
        for aid, source, title, text, at, url, _ in rows:
            snippet = ' '.join((text or '').split())[:SNIPPET_CHARS]
            w.writerow([aid, f'{at:%Y-%m-%d %H:%M}', source, title, snippet, url, ''])

    snapshot = {
        'days': list(days),
        'model': embedding_provider.model_id(),
        'system': {str(aid): (cid if cid is not None and cid >= 0 else None)
                   for aid, _, _, _, _, _, cid in rows},
    }
    with open(SNAPSHOT, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=1)

    outlets = len({r[1] for r in rows})
    print(f'  {len(rows)} articles from {outlets} outlets on {", ".join(days)}')
    print(f'  to label:        {TO_LABEL}')
    print(f'  system snapshot: {SNAPSHOT}  (do not open before labelling)')


# ── metrics ───────────────────────────────────────────────────────────────────

def _as_partition(labels):
    """None means 'alone': give each such article a label nobody else has."""
    out, fresh = [], 0
    for lab in labels:
        if lab is None:
            out.append(('__alone__', fresh))
            fresh += 1
        else:
            out.append(lab)
    return out


def _pairwise(gold, pred):
    """Precision/recall/F1 over article pairs placed together."""
    same_g, same_p, both = 0, 0, 0
    for i, j in combinations(range(len(gold)), 2):
        g, p = gold[i] == gold[j], pred[i] == pred[j]
        same_g += g
        same_p += p
        both += g and p
    prec = both / same_p if same_p else 1.0
    rec = both / same_g if same_g else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def _bcubed(gold, pred):
    """B-cubed precision/recall/F1: per article, averaged over articles."""
    by_g, by_p = defaultdict(set), defaultdict(set)
    for i, (g, p) in enumerate(zip(gold, pred)):
        by_g[g].add(i)
        by_p[p].add(i)
    prec = rec = 0.0
    for i, (g, p) in enumerate(zip(gold, pred)):
        overlap = len(by_g[g] & by_p[p])
        prec += overlap / len(by_p[p])
        rec += overlap / len(by_g[g])
    n = len(gold)
    prec, rec = prec / n, rec / n
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def score_partition(gold, pred):
    from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                                 homogeneity_completeness_v_measure)
    g = [str(x) for x in gold]
    p = [str(x) for x in pred]
    h, c, v = homogeneity_completeness_v_measure(g, p)
    pp, pr, pf = _pairwise(g, p)
    bp, br, bf = _bcubed(g, p)
    return {'ARI': adjusted_rand_score(g, p), 'NMI': normalized_mutual_info_score(g, p),
            'homog': h, 'compl': c, 'V': v,
            'pair_P': pp, 'pair_R': pr, 'pair_F1': pf,
            'bcubed_P': bp, 'bcubed_R': br, 'bcubed_F1': bf,
            'clusters': len(set(p))}


# ── baselines ─────────────────────────────────────────────────────────────────

def _load_articles(ids):
    from database import ArticleDatabase
    db = ArticleDatabase()
    conn = db._connect()
    with conn.cursor() as cur:
        cur.execute("SELECT id, title, COALESCE(body_cleaned, body, ''), embedding "
                    "FROM articles WHERE id = ANY(%s)", (ids,))
        rows = {r[0]: r for r in cur.fetchall()}
    db._release(conn)
    texts = [f"{rows[i][1]}. {rows[i][2]}" for i in ids]
    vecs = np.stack([np.frombuffer(bytes(rows[i][3]), dtype='float32') for i in ids])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return texts, vecs


def hdbscan_raw(vecs):
    import hdbscan
    from config import HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES, CLUSTER_SELECTION_EPSILON
    labels = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
                             min_samples=HDBSCAN_MIN_SAMPLES, metric='euclidean',
                             cluster_selection_method='eom',
                             cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON).fit_predict(vecs)
    return [int(x) if x >= 0 else None for x in labels]


def _tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=1).fit_transform(texts)


def tfidf_kmeans(texts, k):
    from sklearn.cluster import KMeans
    return [int(x) for x in KMeans(n_clusters=k, n_init=10, random_state=42)
            .fit_predict(_tfidf(texts))]


def tfidf_kmeans_silhouette(texts):
    """k by best cosine silhouette over a coarse grid, as a deployed system
    without labels would have to choose it."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    x = _tfidf(texts)
    n = x.shape[0]
    best_k, best_s, best_labels = None, -1.0, None
    for k in range(10, n - 1, max(5, n // 30)):
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(x)
        s = silhouette_score(x, labels, metric='cosine')
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    return [int(v) for v in best_labels], best_k


# ── score ─────────────────────────────────────────────────────────────────────

def _read_labels(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    missing = [r['article_id'] for r in rows if not (r.get('event') or '').strip()]
    if missing:
        sys.exit(f'{len(missing)} articles in {path} have no event label, '
                 f'e.g. article_id {missing[0]}. Label every row first.')
    return {int(r['article_id']): ' '.join(r['event'].split()).lower() for r in rows}


def score(labelled, second=None):
    gold_map = _read_labels(labelled)
    with open(SNAPSHOT, encoding='utf-8') as f:
        snap = json.load(f)
    ids = sorted(gold_map)
    unknown = [i for i in ids if str(i) not in snap['system']]
    if unknown:
        sys.exit(f'{len(unknown)} labelled articles are not in the snapshot, e.g. {unknown[0]}.')

    gold = [gold_map[i] for i in ids]
    sizes = Counter(gold)
    print(f'  {len(ids)} articles, {len(sizes)} gold events: '
          f'{sum(1 for s in sizes.values() if s == 1)} single-article, '
          f'{sum(1 for s in sizes.values() if s > 1)} with 2+ articles '
          f'(largest {max(sizes.values())})')

    texts, vecs = _load_articles(ids)
    km_labels, km_k = tfidf_kmeans_silhouette(texts)
    systems = {
        'system': _as_partition([snap['system'][str(i)] for i in ids]),
        'hdbscan_raw': _as_partition(hdbscan_raw(vecs)),
        f'tfidf_kmeans (k={km_k})': km_labels,
        f'tfidf_kmeans_k* (k={len(sizes)})': tfidf_kmeans(texts, len(sizes)),
        'all_singletons': list(range(len(ids))),
    }

    cols = ['ARI', 'NMI', 'homog', 'compl', 'V', 'pair_P', 'pair_R', 'pair_F1',
            'bcubed_P', 'bcubed_R', 'bcubed_F1', 'clusters']
    results = {name: score_partition(gold, pred) for name, pred in systems.items()}
    width = max(len(n) for n in results) + 2
    print('\n  ' + ''.ljust(width) + ''.join(c.rjust(12) for c in cols))
    for name, r in results.items():
        cells = ''.join((f'{r[c]:12d}' if c == 'clusters' else f'{r[c]:12.3f}') for c in cols)
        print('  ' + name.ljust(width) + cells)

    singles = results['all_singletons']
    print('\n  Read ARI and pairwise F1 first. With this many single-article events, NMI,\n'
          '  homogeneity and V-measure reward splitting: putting every article alone\n'
          f'  scores NMI {singles["NMI"]:.3f} while finding no event at all.')

    out = os.path.join(GOLD_DIR, 'results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'labelled': labelled, 'articles': len(ids), 'gold_events': len(sizes),
                   'results': results}, f, indent=1)
    print(f'\n  wrote {out}')

    if second:
        other = _read_labels(second)
        shared = sorted(set(other) & set(gold_map))
        a = [gold_map[i] for i in shared]
        b = [other[i] for i in shared]
        r = score_partition(a, b)
        print(f'\n  Agreement between labellers on {len(shared)} shared articles: '
              f'ARI {r["ARI"]:.3f}, NMI {r["NMI"]:.3f}, pairwise F1 {r["pair_F1"]:.3f}')


def main():
    ap = argparse.ArgumentParser(description='Gold-set export and scoring for event clustering.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('export', help='write gold/to_label.csv and freeze the system labels')
    e.add_argument('--days', nargs='+', default=list(DEFAULT_DAYS), help='YYYY-MM-DD ...')
    s = sub.add_parser('score', help='score the system and baselines against labels')
    s.add_argument('labelled')
    s.add_argument('--second', help="a second labeller's file, for agreement")
    args = ap.parse_args()
    if args.cmd == 'export':
        export(args.days)
    else:
        score(args.labelled, args.second)


if __name__ == '__main__':
    main()
