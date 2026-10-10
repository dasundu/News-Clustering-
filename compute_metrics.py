# compute_metrics.py
"""
Compute every metric the clustering component can report WITHOUT a labelled
gold set, and write them to metrics_report.txt.

Why this exists
---------------
Intrinsic metrics that require ground truth — ARI, NMI, pairwise F1 — cannot
be computed until a gold set is annotated. But a large family of *internal*
validation metrics needs no labels at all: they measure whether the produced
clusters are internally tight and mutually separated, which is a real and
reportable property of a clustering.

Everything here is computed from live data and is safe to cite in a thesis,
with the single caveat stated at the end of the report: internal validation
measures geometric quality, not semantic correctness. It tells you the
clusters are well-formed. It does not tell you they are right. Only the gold
set can do that.

Usage
-----
    python compute_metrics.py
    python compute_metrics.py --window 14      # clustering-window subset only
"""

import argparse
import logging
import sys
from collections import Counter

import numpy as np
import psycopg2.extras

logging.disable(logging.CRITICAL)

from database import ArticleDatabase          # noqa: E402
from clusterer import _cached_embeddings      # noqa: E402
import embedding_provider                     # noqa: E402

OUT = 'metrics_report.txt'
_lines = []


def emit(s=''):
    print(s, flush=True)
    _lines.append(s)


def head(t):
    emit()
    emit('=' * 74)
    emit(f'  {t}')
    emit('=' * 74)


def row(label, value, note=''):
    emit(f'  {label:<44} {str(value):>14}  {note}')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=None,
                    help='restrict geometric metrics to the last N days')
    ap.add_argument('--sample', type=int, default=4000,
                    help='max articles for O(n^2) metrics')
    args = ap.parse_args()

    db = ArticleDatabase()
    mid = embedding_provider.model_id()
    conn = db._connect()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def q(sql, *a):
        cur.execute(sql, a)
        return cur.fetchall()

    # ── 1. corpus ────────────────────────────────────────────────────────
    head('1. CORPUS COMPOSITION')
    r = q("""SELECT count(*) total,
                    count(*) FILTER (WHERE body <> '')            with_body,
                    count(*) FILTER (WHERE embedding IS NOT NULL) embedded,
                    count(*) FILTER (WHERE cluster_id >= 0)       clustered,
                    count(DISTINCT source)                        sources,
                    min(COALESCE(published_at, scraped_at))::date lo,
                    max(COALESCE(published_at, scraped_at))::date hi
             FROM articles""")[0]
    total = r['total']
    row('Articles ingested', f"{total:,}")
    row('With extracted body text', f"{r['with_body']:,}",
        f"{100 * r['with_body'] / total:.1f}% of ingested")
    row('Embedded', f"{r['embedded']:,}",
        f"{100 * r['embedded'] / total:.1f}%")
    row('Assigned to a cluster', f"{r['clustered']:,}",
        f"{100 * r['clustered'] / total:.1f}%")
    row('Distinct producing outlets', r['sources'])
    row('Collection period', f"{r['lo']} to {r['hi']}")

    cl = q('SELECT count(*) n, count(topic_id) t FROM cluster_labels')[0]
    tp = q('SELECT count(*) n FROM topics')[0]
    row('Event clusters', f"{cl['n']:,}")
    row('Topics', f"{tp['n']:,}")
    row('Clusters assigned to a topic', f"{cl['t']:,}",
        f"{100 * cl['t'] / cl['n']:.1f}% of clusters")

    # ── 2. extraction yield ──────────────────────────────────────────────
    head('2. INGESTION AND EXTRACTION YIELD (per source)')
    emit(f"  {'source':<22}{'articles':>10}{'with body':>11}{'yield':>9}")
    emit('  ' + '-' * 52)
    rows = q("""SELECT source, count(*) n,
                       count(*) FILTER (WHERE body <> '') b
                FROM articles GROUP BY 1 ORDER BY 2 DESC""")
    for x in rows:
        pct = 100 * x['b'] / x['n'] if x['n'] else 0
        flag = '  <-- low' if pct < 50 else ''
        emit(f"  {x['source']:<22}{x['n']:>10,}{x['b']:>11,}{pct:>8.1f}%{flag}")
    ok = sum(1 for x in rows if x['n'] and 100 * x['b'] / x['n'] >= 50)
    emit('  ' + '-' * 52)
    row('Sources at >=50% extraction yield', f"{ok}/{len(rows)}",
        f"{100 * ok / len(rows):.0f}%")

    # ── 3. cluster size distribution ─────────────────────────────────────
    head('3. CLUSTER SIZE DISTRIBUTION')
    sizes = [x['sz'] for x in q("""SELECT count(*) sz FROM articles
                                   WHERE cluster_id >= 0 GROUP BY cluster_id""")]
    a = np.array(sizes)
    row('Clusters', f"{len(a):,}")
    row('Mean articles per cluster', f"{a.mean():.2f}")
    row('Median', int(np.median(a)))
    row('Std deviation', f"{a.std():.2f}")
    row('Max', int(a.max()))
    row('Singleton clusters', f"{int((a == 1).sum()):,}",
        f"{100 * (a == 1).mean():.1f}% of clusters")
    row('Multi-article clusters', f"{int((a > 1).sum()):,}",
        f"{100 * (a > 1).mean():.1f}%")

    # ── 4. cross-outlet coverage — the metric that matters ───────────────
    head('4. CROSS-OUTLET COVERAGE')
    emit('  The population usable for cross-outlet comparison.')
    emit()
    dist = q("""SELECT nsrc, count(*) n FROM (
                  SELECT cluster_id, count(DISTINCT source) nsrc
                  FROM articles WHERE cluster_id >= 0 GROUP BY 1) x
                GROUP BY 1 ORDER BY 1""")
    tot_c = sum(d['n'] for d in dist)
    emit(f"  {'outlets':>9}{'clusters':>11}{'share':>9}{'cumulative':>12}")
    emit('  ' + '-' * 42)
    cum = 0
    for d in dist:
        cum += d['n']
        emit(f"  {d['nsrc']:>9}{d['n']:>11,}{100 * d['n'] / tot_c:>8.1f}%"
             f"{100 * cum / tot_c:>11.1f}%")
    emit('  ' + '-' * 42)
    for k in (2, 3, 5):
        n = sum(d['n'] for d in dist if d['nsrc'] >= k)
        row(f'Clusters with >={k} outlets', f"{n:,}",
            f"{100 * n / tot_c:.1f}% of clusters")
    msrc = sum(d['nsrc'] * d['n'] for d in dist) / tot_c
    row('Mean outlets per cluster', f"{msrc:.2f}")

    # ── 5. topic layer ───────────────────────────────────────────────────
    head('5. TOPIC LAYER')
    tsz = [x['n'] for x in q("""SELECT count(*) n FROM cluster_labels
                                WHERE topic_id IS NOT NULL GROUP BY topic_id""")]
    if tsz:
        t = np.array(tsz)
        row('Topics', f"{len(t):,}")
        row('Mean clusters per topic', f"{t.mean():.2f}")
        row('Median', int(np.median(t)))
        row('Max', int(t.max()))
        row('Topic coverage (clusters in a topic)', f"{cl['t']:,}",
            f"{100 * cl['t'] / cl['n']:.1f}%")
        multi = q("""SELECT count(*) n FROM (
                       SELECT cluster_id FROM articles WHERE cluster_id >= 0
                       GROUP BY 1 HAVING count(DISTINCT source) >= 2) x""")[0]['n']
        cov = q("""SELECT count(*) n FROM cluster_labels cl
                   WHERE cl.topic_id IS NOT NULL AND EXISTS (
                     SELECT 1 FROM articles a WHERE a.cluster_id = cl.cluster_id
                     GROUP BY a.cluster_id HAVING count(DISTINCT a.source) >= 2)""")[0]['n']
        row('Topic coverage of multi-outlet clusters', f"{cov:,}/{multi:,}",
            f"{100 * cov / multi:.1f}%")

    # ── 6. geometric validation (no labels needed) ───────────────────────
    head('6. INTERNAL VALIDATION  (no ground truth required)')

    where = 'cluster_id >= 0 AND embedding IS NOT NULL'
    params = []
    if args.window:
        where += " AND COALESCE(published_at, scraped_at) >= NOW() - (%s||' days')::interval"
        params.append(args.window)
    meta = q(f'SELECT id, cluster_id, source FROM articles WHERE {where}', *params)

    if len(meta) > args.sample:
        rng = np.random.default_rng(42)
        keep = set(rng.choice([m['id'] for m in meta], size=args.sample, replace=False))
        meta = [m for m in meta if m['id'] in keep]

    ids = [m['id'] for m in meta]
    vec = _cached_embeddings(db, ids, mid)
    meta = [m for m in meta if m['id'] in vec]
    X = np.stack([vec[m['id']] for m in meta])
    labels = np.array([m['cluster_id'] for m in meta])
    emit(f"  Computed over {len(meta):,} articles"
         + (f" from the last {args.window} days" if args.window else " (sampled)"))
    emit()

    # intra-cluster coherence
    intra, sizes_m = [], []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        if len(idx) < 2:
            continue
        V = X[idx]
        cen = V.mean(axis=0)
        cen /= (np.linalg.norm(cen) + 1e-9)
        intra.append(float((V @ cen).mean()))
        sizes_m.append(len(idx))
    if intra:
        ia = np.array(intra)
        row('Mean intra-cluster coherence', f"{ia.mean():.4f}",
            'cosine to own centroid')
        row('Median', f"{np.median(ia):.4f}")
        row('Std deviation', f"{ia.std():.4f}")
        row('Min / Max', f"{ia.min():.3f} / {ia.max():.3f}")
        row('Clusters above 0.90 coherence',
            f"{int((ia >= 0.90).sum())}/{len(ia)}",
            f"{100 * (ia >= 0.90).mean():.1f}%")

    # same-cluster vs different-cluster similarity
    emit()
    rng = np.random.default_rng(42)
    same, diff = [], []
    by_c = {}
    for i, l in enumerate(labels):
        by_c.setdefault(l, []).append(i)
    multi_c = [v for v in by_c.values() if len(v) >= 2]
    for _ in range(8000):
        g = multi_c[rng.integers(len(multi_c))]
        i, j = rng.choice(len(g), size=2, replace=False)
        same.append(float(X[g[i]] @ X[g[j]]))
    for _ in range(8000):
        i, j = rng.integers(len(X)), rng.integers(len(X))
        if labels[i] != labels[j]:
            diff.append(float(X[i] @ X[j]))
    s, d = np.array(same), np.array(diff)
    row('Same-cluster mean similarity', f"{s.mean():.4f}", f"sd {s.std():.4f}")
    row('Different-cluster mean similarity', f"{d.mean():.4f}", f"sd {d.std():.4f}")
    row('Separation (difference of means)', f"{s.mean() - d.mean():.4f}")
    pooled = np.sqrt((s.var() + d.var()) / 2)
    row("Cohen's d (effect size)", f"{(s.mean() - d.mean()) / pooled:.3f}",
        'convention: >0.8 is large')
    for p in (1, 5):
        row(f'Same-cluster {p}th percentile', f"{np.percentile(s, p):.4f}")
    for p in (95, 99):
        row(f'Different-cluster {p}th percentile', f"{np.percentile(d, p):.4f}")
    ov = float((d >= np.percentile(s, 5)).mean())
    row('Distribution overlap', f"{100 * ov:.2f}%",
        'different-cluster pairs above same-cluster 5th pct')

    # sklearn internal indices
    emit()
    try:
        from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                     calinski_harabasz_score)
        mask = np.isin(labels, [c for c, v in by_c.items() if len(v) >= 2])
        Xm, Lm = X[mask], labels[mask]
        if len(np.unique(Lm)) > 1:
            row('Silhouette coefficient',
                f"{silhouette_score(Xm, Lm, metric='cosine'):.4f}",
                'range -1 to 1, higher better')
            row('Davies-Bouldin index',
                f"{davies_bouldin_score(Xm, Lm):.4f}", 'lower better')
            row('Calinski-Harabasz index',
                f"{calinski_harabasz_score(Xm, Lm):,.1f}", 'higher better')
            row('  computed over', f"{len(Xm):,} articles",
                f"{len(np.unique(Lm)):,} clusters")
    except Exception as e:
        emit(f'  sklearn indices unavailable: {e}')

    # ── 7. outlet concentration ──────────────────────────────────────────
    head('7. SOURCE CONCENTRATION')
    sc = Counter(m['source'] for m in meta)
    tot_s = sum(sc.values())
    shares = np.array([v / tot_s for v in sc.values()])
    row('Herfindahl-Hirschman Index', f"{(shares ** 2).sum():.4f}",
        'concentration; 1/n = perfectly even')
    row('Even-distribution reference (1/n)', f"{1 / len(sc):.4f}")
    row('Largest source share', f"{100 * shares.max():.1f}%")
    row('Top-3 share', f"{100 * np.sort(shares)[-3:].sum():.1f}%")

    db._release(conn)

    # ── caveat ───────────────────────────────────────────────────────────
    head('INTERPRETATION — STATE THIS IN THE THESIS')
    emit("""
  Every figure above is computed from live data and requires no labelled
  ground truth. Sections 1-5 and 7 are descriptive statistics. Section 6
  is INTERNAL validation: it measures whether clusters are geometrically
  tight and mutually separated.

  Internal validation does not establish semantic correctness. A clustering
  can be geometrically excellent and still group the wrong articles. The
  following remain REQUIRED and are not substitutable by anything above:

    - Adjusted Rand Index          needs a labelled gold set
    - Normalised Mutual Information  needs a labelled gold set
    - Pairwise precision/recall/F1   needs a labelled gold set
    - Homogeneity / completeness     needs a labelled gold set
    - Baseline comparison            needs the same gold set
    - Identity retention across runs needs instrumented consecutive runs

  Report Section 6 as evidence that the clusters are WELL-FORMED, never as
  evidence that they are CORRECT. Conflating the two is the single most
  common error in clustering evaluation and an examiner will look for it.
""")

    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_lines))
    emit(f'  Written to {OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
