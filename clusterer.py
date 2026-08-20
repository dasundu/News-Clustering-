# clusterer.py
"""
Event-based clustering using HDBSCAN on L2-normalized sentence embeddings.
Same news event -> same cluster_id, held stable across runs (see below).
Cluster titles selected by quality-weighted centroid scoring.

Incremental design
-------------------
HDBSCAN's own output labels (0, 1, 2, ..., -1 for noise) are *not* stable
identifiers — they're just discovery order within a single fit, and can land
on a completely different grouping the next time the model is fit, even on
near-identical data. A naive "re-fit everything, store the labels" approach
(the previous version of this file) therefore breaks the system's own
"same event -> same cluster_id" invariant: a story could change cluster_id
every run, or worse, coincidentally reuse an old id that used to mean a
different, unrelated event.

To get real incremental behaviour — "add to an existing cluster, or create a
new one" — each run:
  1. Re-fits HDBSCAN structurally, but only on articles from the last
     CLUSTER_WINDOW_DAYS (bounds run time as the corpus grows, and keeps
     clustering focused on live coverage rather than resurrecting old news),
     with an adaptive cluster_selection_epsilon sized to that run's own data
     rather than one fixed value for every run (see _adaptive_epsilon).
  2. Runs a quality-refinement pass over those raw structural groups: splits
     anything oversized, too semantically loose, spanning too many days, or
     dominated by a single outlet; merges complementary single-outlet
     remnants back together (see _refine_clusters).
  3. Computes each *previous* persistent cluster's centroid from whichever of
     its members fall inside this window (i.e. its currently-active core).
  4. Matches each refined structural cluster to the previous cluster whose
     centroid it is most cosine-similar to, above CLUSTER_MATCH_THRESHOLD,
     one-to-one (greedy, highest similarity first). A match reuses that
     cluster's persistent id; no match mints a new one.
  5. Articles outside the window are never touched — they keep whatever
     cluster_id they were last assigned.

This is the standard approach for tracking topics across time-windowed
re-clustering (cf. Topic Detection and Tracking / TDT literature), and is a
deliberate middle ground between two extremes: true streaming/online
clustering (assign-or-spawn per article as it arrives, no batch re-fit at
all) would need a different algorithm entirely; naive full re-clustering
(the old approach) is simpler but has no identity guarantees at all.

Two levels: events, then topics
--------------------------------
Everything above describes *event* clusters — "these articles cover the same
news event", the unit a cross-outlet comparison needs. On top of that,
ClusterTopicBuilder groups related events into topics ("these events are all
part of the same ongoing story"), by running a second, looser clustering pass
over event-cluster centroids. Event clustering is left completely untouched
by it; topics are strictly additive. Both levels persist their ids across
runs through the same _match_to_previous mechanism, differing only in
threshold.
"""

import logging
import os
import re
from datetime import date, datetime
import numpy as np
import requests
import hdbscan
from sklearn.cluster import AgglomerativeClustering
import embedding_provider
from database import ArticleDatabase
from config import (
    HDBSCAN_MIN_CLUSTER, HDBSCAN_MIN_SAMPLES,
    CLUSTER_WINDOW_DAYS, CLUSTER_MATCH_THRESHOLD,
    CLUSTER_MAX_SIZE, CLUSTER_MIN_COHERENCE, CLUSTER_MAX_SPLIT_DEPTH,
    CLUSTER_MAX_DOMINANT_OUTLET_SHARE, CLUSTER_MERGE_SIMILARITY_THRESHOLD,
    CLUSTER_MAX_SPAN_DAYS,
    CLUSTER_LABEL_USE_LLM, CLUSTER_LABEL_QUALITY_FLOOR, GEMINI_LABEL_MODELS,
    TOPIC_SIMILARITY_THRESHOLD, TOPIC_MATCH_THRESHOLD, TOPIC_MIN_CLUSTERS,
    EPSILON_CLAMP_LOW, EPSILON_CLAMP_HIGH,
)

logger = logging.getLogger(__name__)

_OPINION_STARTS = {
    'he ', 'she ', 'i ', 'we ', 'they ', 'his ', 'her ', 'our ', 'my ',
    'you ', 'it ', 'this ', 'that ', 'there ', 'here ',
}
_WEAK_WORDS = {
    'opinion', 'column', 'editorial', 'analysis', 'comment', 'view',
    'letter', 'blog', 'exclusive', 'interview', 'feature',
}
_HARD_NEWS_TERMS = {
    'sri lanka', 'parliament', 'president', 'minister', 'government',
    'rupee', 'imf', 'economy', 'tax', 'court', 'police', 'election',
    'budget', 'inflation', 'export', 'import', 'bank', 'fuel', 'power',
}


def _title_quality(title: str) -> float:
    """Score a title 0-1: higher = better hard-news headline."""
    if not title:
        return 0.0
    words = title.split()
    n = len(words)
    if n < 4:
        return 0.1
    tl = title.lower()
    for start in _OPINION_STARTS:
        if tl.startswith(start):
            return 0.2
    if any(w in tl for w in _WEAK_WORDS):
        return 0.25
    score = min(n / 12, 0.35)
    if any(ch.isdigit() for ch in title):
        score += 0.2
    cap_count = sum(1 for w in words[1:] if w and w[0].isupper())
    score += min(cap_count / 5, 0.25)
    if any(term in tl for term in _HARD_NEWS_TERMS):
        score += 0.2
    return min(score, 1.0)


# ── Cluster labeling: LLM -> most-central-title -> KeyBERT -> sanitized ────
# Layered fallback adapted from the sibling project's (media-bias-detection)
# labeling chain. The "most-central title" tier is this project's own
# pre-existing heuristic (see _generate_labels); LLM and KeyBERT are new.

def _sanitize_label(text: str) -> str:
    """Strip a trailing outlet-name attribution and excess whitespace so a
    label reads as a clean headline no matter which tier produced it."""
    if not text:
        return ''
    from config import outlet_name_fragments
    text = re.sub(r'\s+', ' ', text).strip().strip('"\'')
    for sep in (' - ', ' | ', ' — '):
        if sep in text:
            parts = text.split(sep)
            if parts[-1].strip().lower() in outlet_name_fragments():
                text = sep.join(parts[:-1]).strip()
    return text[:200].strip()


_keybert_model = None


def _get_keybert():
    """
    Lazily loaded and cached for the process lifetime — most clusters never
    need this tier, so it shouldn't cost a model load on every run.

    Returns None when embeddings come from a hosted API rather than a local
    model. KeyBERT scores hundreds of candidate n-grams per call, which is
    fine against a resident model and wholly impractical against a
    per-request API. Callers fall through to the next labelling tier.
    """
    global _keybert_model
    if not embedding_provider.is_local():
        return None
    if _keybert_model is None:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        from config import LOCAL_EMBEDDING_MODEL
        logger.info(f"Loading KeyBERT backend ({LOCAL_EMBEDDING_MODEL}) for weak-title labels...")
        _keybert_model = KeyBERT(model=SentenceTransformer(LOCAL_EMBEDDING_MODEL))
    return _keybert_model


def _recase_from_source(phrase: str, source_text: str) -> str:
    """
    Recover a keyphrase's original casing and punctuation from the text it
    was extracted from. KeyBERT tokenizes through a lowercasing vectorizer,
    so it hands back 'adb policy based loan' for what the headline actually
    wrote as 'ADB policy-based loan' — acronyms flattened, hyphens dropped.
    Matching the words back against the source restores both. The gap
    allowed between consecutive words absorbs punctuation ("Judges’
    retirement") and one short stopword, since stop_words='english' drops
    those from the phrase but they belong in a readable label ("Vice
    President of SAFF"). Periods are excluded from the gap so a match can
    never run across the '. ' joining two separate headlines. Returns '' if
    the phrase can't be found, leaving the caller to fall back.
    """
    words = [w for w in re.split(r'\W+', phrase) if w]
    if not words:
        return ''
    gap = r'[^\w.]{1,3}(?:\w{1,4}[^\w.]{1,3})?'
    pattern = gap.join(re.escape(w) for w in words)
    match = re.search(pattern, source_text, re.IGNORECASE)
    return match.group(0) if match else ''


def _keybert_label(titles: list) -> str:
    """Extractive keyphrase from the cluster's combined titles, used only
    when none of the real headlines look like a solid hard-news title."""
    try:
        kb = _get_keybert()
        if kb is None:
            return ''          # not available under a hosted-API provider
        text = '. '.join(t for t in titles if t)[:2000]
        if not text.strip():
            return ''
        phrases = kb.extract_keywords(
            text, keyphrase_ngram_range=(2, 5), stop_words='english', top_n=1,
        )
        if not phrases:
            return ''
        phrase = phrases[0][0]
        if not phrase:
            return ''
        # Capitalize the lead word either way: a keyphrase lifted from
        # mid-sentence otherwise yields a label starting lowercase.
        phrase = _recase_from_source(phrase, text) or phrase
        return phrase[:1].upper() + phrase[1:]
    except Exception as e:
        logger.warning(f"KeyBERT labeling failed: {e}")
        return ''


def _llm_label(titles: list) -> str:
    """Ask Gemini for one neutral headline summarizing what several
    same-event titles have in common. Returns '' on any failure (missing
    key, network error, no candidate model available) so callers always
    have a local fallback to drop to — this tier is never load-bearing."""
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not titles:
        return ''

    prompt = (
        "These headlines from different Sri Lankan news outlets all cover "
        "the same news event:\n"
        + '\n'.join(f'- {t}' for t in titles[:6])
        + "\n\nWrite one concise, neutral, factual headline (max 15 words) "
          "summarizing the shared event. Do not mention any outlet's name. "
          "Reply with only the headline — no quotes, no explanation."
    )
    for model in GEMINI_LABEL_MODELS:
        try:
            r = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                json={'contents': [{'parts': [{'text': prompt}]}]},
                timeout=20,
            )
            if r.status_code >= 400:
                logger.debug(f"Gemini label via {model} failed: HTTP {r.status_code}")
                continue
            text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            if text:
                return text
        except Exception as e:
            logger.debug(f"Gemini label attempt via {model} failed: {e}")
            continue
    return ''


def _llm_topic_label(event_titles: list) -> str:
    """Ask Gemini for a short *topic* name covering several distinct events
    in one ongoing story — deliberately a different request from _llm_label,
    which summarizes several outlets' takes on one single event. Returns ''
    on any failure so the local tiers below it stay load-bearing."""
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not event_titles:
        return ''

    prompt = (
        "These headlines are separate news events that all belong to the "
        "same ongoing story or theme in Sri Lankan news:\n"
        + '\n'.join(f'- {t}' for t in event_titles[:8])
        + "\n\nWrite a short topic name (2-5 words) for the theme they share "
          "— the kind of heading a news site would file all of them under. "
          "Reply with only the topic name, no quotes, no explanation."
    )
    for model in GEMINI_LABEL_MODELS:
        try:
            r = requests.post(
                f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                json={'contents': [{'parts': [{'text': prompt}]}]},
                timeout=20,
            )
            if r.status_code >= 400:
                logger.debug(f"Gemini topic label via {model} failed: HTTP {r.status_code}")
                continue
            text = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
            if text:
                return text
        except Exception as e:
            logger.debug(f"Gemini topic label attempt via {model} failed: {e}")
            continue
    return ''


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _match_to_previous(new_centroids: dict, prev_centroids: dict,
                       threshold: float) -> dict:
    """
    Greedy one-to-one matching of this run's fresh groupings onto the
    previous run's persistent ids, by centroid cosine similarity, highest
    similarity first. Returns {new_label: previous_id} containing only the
    labels that matched; anything absent from the result is a genuinely new
    grouping whose caller must mint a fresh id for it.

    Shared by both levels of the hierarchy — events matching previous event
    clusters (_assign_persistent_ids) and topics matching previous topics
    (ClusterTopicBuilder) — since "keep the same id when this is the same
    thing as last run" is the identical problem at both levels, differing
    only in the threshold applied.
    """
    candidates = [
        (_cosine(new_centroids[nl], prev_centroids[pid]), nl, pid)
        for nl in new_centroids for pid in prev_centroids
    ]
    candidates.sort(key=lambda t: t[0], reverse=True)

    matched: dict = {}
    used_prev: set = set()
    for sim, nl, pid in candidates:
        if sim < threshold:
            break
        if nl in matched or pid in used_prev:
            continue
        matched[nl] = pid
        used_prev.add(pid)
    return matched


def _adaptive_epsilon(matrix: np.ndarray, low: float = None, high: float = None,
                       percentile: float = 25.0, sample_size: int = 500) -> float:
    """
    HDBSCAN's cluster_selection_epsilon as a percentile of this run's own
    pairwise embedding distances, clamped to a sane range, instead of one
    fixed value for every run. A quiet news day (articles more spread out)
    and a single dominant story breaking (many near-duplicate articles) call
    for different merge radii; this adapts to which one a given run looks
    like. Same idea as the sibling project's adaptive agglomerative-clustering
    threshold, re-derived for euclidean distance over L2-normalized vectors
    (the metric this project's HDBSCAN actually uses) rather than cosine
    distance.
    """
    # Clamp bounds are provider-specific: they are euclidean distances over
    # unit vectors, and the whole distance scale compresses under Gemini
    # (same-event pairs sit near 0.40 apart vs 0.68 under the local model).
    low = EPSILON_CLAMP_LOW if low is None else low
    high = EPSILON_CLAMP_HIGH if high is None else high

    n = len(matrix)
    if n < 4:
        return (low + high) / 2  # too few points to estimate a percentile

    if n > sample_size:
        rng = np.random.default_rng(seed=42)
        sample = matrix[rng.choice(n, size=sample_size, replace=False)]
    else:
        sample = matrix

    diffs = sample[:, None, :] - sample[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    upper = np.triu_indices(len(sample), k=1)
    pairwise = dists[upper]
    if pairwise.size == 0:
        return (low + high) / 2

    eps = float(np.percentile(pairwise, percentile))
    return float(np.clip(eps, low, high))


def _cluster_centroid(matrix: np.ndarray, indices: list) -> np.ndarray:
    return matrix[indices].mean(axis=0)


def _mean_similarity_to_centroid(matrix: np.ndarray, indices: list) -> float:
    centroid = _cluster_centroid(matrix, indices)
    norm = np.linalg.norm(centroid) + 1e-9
    unit_centroid = centroid / norm
    sims = matrix[indices] @ unit_centroid
    return float(np.mean(sims))


def _as_date(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return None


def _date_span_days(indices: list, dates: list) -> int:
    parsed = [d for i in indices if (d := _as_date(dates[i])) is not None]
    return (max(parsed) - min(parsed)).days if len(parsed) >= 2 else 0


def _split_cluster_if_needed(indices: list, matrix: np.ndarray, depth: int) -> list:
    """Recursively bisect a cluster while it's oversized or too loose to be
    one event, up to CLUSTER_MAX_SPLIT_DEPTH — but past that depth, keep
    splitting anyway if it's still over CLUSTER_MAX_SIZE, since an unsplit
    oversized cluster is worse than a slightly-too-deep split."""
    if len(indices) <= 2:
        return [indices]

    too_large = len(indices) > CLUSTER_MAX_SIZE
    too_loose = _mean_similarity_to_centroid(matrix, indices) < CLUSTER_MIN_COHERENCE

    if not too_large and not too_loose:
        return [indices]
    if depth >= CLUSTER_MAX_SPLIT_DEPTH and not too_large:
        return [indices]
    if depth >= CLUSTER_MAX_SPLIT_DEPTH * 3:  # absolute recursion guard
        return [indices]

    # AgglomerativeClustering(n_clusters=2) rather than HDBSCAN here: this
    # step needs a guaranteed binary split of a small subset, not density
    # detection — HDBSCAN could put everything in noise or find >2 groups.
    splitter = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='average')
    sub_labels = splitter.fit_predict(matrix[indices])
    left = [indices[i] for i, l in enumerate(sub_labels) if l == 0]
    right = [indices[i] for i, l in enumerate(sub_labels) if l == 1]
    if not left or not right:
        return [indices]  # degenerate split; accept as-is

    return (_split_cluster_if_needed(left, matrix, depth + 1) +
            _split_cluster_if_needed(right, matrix, depth + 1))


def _split_oversized_or_incoherent(clusters: dict, matrix: np.ndarray) -> dict:
    next_label = max(clusters.keys(), default=-1) + 1
    refined = {}
    for indices in clusters.values():
        for group in _split_cluster_if_needed(indices, matrix, depth=0):
            refined[next_label] = group
            next_label += 1
    return refined


def _split_by_time_span(clusters: dict, dates: list) -> dict:
    """Bound each structural cluster to CLUSTER_MAX_SPAN_DAYS from its
    earliest member, so one run's fit can't lump together two textually
    similar but unrelated events from different weeks."""
    refined = {}
    next_label = 0
    for indices in clusters.values():
        dated = sorted(
            ((i, d) for i in indices if (d := _as_date(dates[i])) is not None),
            key=lambda t: t[1],
        )
        undated = [i for i in indices if _as_date(dates[i]) is None]
        if not dated:
            refined[next_label] = list(indices)
            next_label += 1
            continue

        groups, group_start = [], None
        for i, d in dated:
            if group_start is None or (d - group_start).days > CLUSTER_MAX_SPAN_DAYS:
                groups.append([i])
                group_start = d
            else:
                groups[-1].append(i)
        if undated:
            groups[max(range(len(groups)), key=lambda p: len(groups[p]))].extend(undated)
        for group in groups:
            refined[next_label] = group
            next_label += 1
    return refined


def _split_outlet_dominated(clusters: dict, sources: list) -> dict:
    next_label = max(clusters.keys(), default=-1) + 1
    refined = {}
    for indices in clusters.values():
        if not indices:
            continue
        counts = {}
        for i in indices:
            counts[sources[i]] = counts.get(sources[i], 0) + 1
        dominant_outlet, dominant_count = max(counts.items(), key=lambda kv: kv[1])
        share = dominant_count / len(indices)

        if share <= CLUSTER_MAX_DOMINANT_OUTLET_SHARE or len(indices) < 4:
            refined[next_label] = indices
            next_label += 1
            continue

        dominant_idxs = [i for i in indices if sources[i] == dominant_outlet]
        other_idxs = [i for i in indices if sources[i] != dominant_outlet]
        if dominant_idxs and other_idxs:
            refined[next_label] = dominant_idxs
            next_label += 1
            refined[next_label] = other_idxs
            next_label += 1
        else:
            refined[next_label] = indices
            next_label += 1
    return refined


def _merge_single_outlet_clusters(clusters: dict, matrix: np.ndarray,
                                   sources: list, dates: list) -> dict:
    """Two single-outlet clusters with no outlet overlap and a strongly
    similar centroid are probably the same event, split apart only because
    each outlet's own articles clustered with themselves first; merge them
    into one multi-source cluster, bounded by size and date span."""
    items = {}
    for label, indices in clusters.items():
        items[label] = {
            'indices': list(indices),
            'outlets': {sources[i] for i in indices},
            'centroid': _cluster_centroid(matrix, indices),
        }

    for label in list(items.keys()):
        item = items.get(label)
        if item is None or len(item['outlets']) >= 2:
            continue

        best_label, best_sim = None, -1.0
        for cand_label, cand in items.items():
            if cand_label == label or item['outlets'] & cand['outlets']:
                continue
            sim = _cosine(item['centroid'], cand['centroid'])
            if sim > best_sim:
                best_sim, best_label = sim, cand_label

        if best_label is None or best_sim < CLUSTER_MERGE_SIMILARITY_THRESHOLD:
            continue
        target = items[best_label]
        combined = target['indices'] + item['indices']
        if len(combined) > CLUSTER_MAX_SIZE or _date_span_days(combined, dates) > CLUSTER_MAX_SPAN_DAYS:
            continue

        target['indices'] = combined
        target['outlets'] |= item['outlets']
        target['centroid'] = _cluster_centroid(matrix, combined)
        items.pop(label, None)

    return {label: item['indices'] for label, item in items.items()}


def _refine_clusters(labels: np.ndarray, matrix: np.ndarray,
                      sources: list, dates: list) -> np.ndarray:
    """
    Quality-control pass over HDBSCAN's raw structural groups, run before
    persistent-id matching so that matching operates on already-cleaned
    clusters. Mirrors the sibling project's split/merge/re-check pipeline:
    split anything oversized, too loose, or spanning too many days; split
    apart anything one outlet dominates; merge complementary single-outlet
    remnants back together; then re-check split conditions once more, since
    a merge can produce a group that's newly oversized or newly dominated.
    Returns a fresh label array (-1 preserved for noise and for anything a
    split leaves below HDBSCAN_MIN_CLUSTER members).
    """
    clusters = {
        int(label): [i for i, l in enumerate(labels) if l == label]
        for label in sorted(set(labels.tolist()) - {-1})
    }
    if not clusters:
        return labels

    clusters = _split_oversized_or_incoherent(clusters, matrix)
    clusters = _split_by_time_span(clusters, dates)
    clusters = _split_outlet_dominated(clusters, sources)

    clusters = _merge_single_outlet_clusters(clusters, matrix, sources, dates)
    clusters = _split_oversized_or_incoherent(clusters, matrix)
    clusters = _split_by_time_span(clusters, dates)
    clusters = _split_outlet_dominated(clusters, sources)

    refined = np.full(len(labels), -1, dtype=int)
    next_label = 0
    for indices in clusters.values():
        if len(indices) < HDBSCAN_MIN_CLUSTER:
            continue  # too small to survive refinement; falls back to noise
        for i in indices:
            refined[i] = next_label
        next_label += 1
    return refined


class ArticleClusterer:
    def __init__(self):
        self.db = ArticleDatabase()

    def run(self) -> dict:
        model_id = embedding_provider.model_id()
        rows = self.db.get_recent_embeddings(CLUSTER_WINDOW_DAYS, model_id)
        if len(rows) < 4:
            logger.info(f"Too few embedded articles in the last "
                        f"{CLUSTER_WINDOW_DAYS}d ({len(rows)}) to cluster")
            return {'total': len(rows), 'clusters': 0, 'noise': len(rows),
                    'new_clusters': 0, 'continued_clusters': 0}

        ids = [r['id'] for r in rows]
        prev_ids = [r['cluster_id'] for r in rows]
        sources = [r['source'] for r in rows]
        dates = [r['effective_date'] for r in rows]
        matrix = np.stack([
            np.frombuffer(r['embedding'], dtype='float32') for r in rows
        ])

        epsilon = _adaptive_epsilon(matrix)
        logger.info(f"Clustering {len(ids)} articles from the last "
                    f"{CLUSTER_WINDOW_DAYS}d... (adaptive epsilon={epsilon:.3f})")
        model = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=epsilon,
        )
        labels = model.fit_predict(matrix)
        labels = _refine_clusters(labels, matrix, sources, dates)

        final_ids, new_count, continued_count = self._assign_persistent_ids(
            labels, prev_ids, matrix
        )

        assignments = list(zip(final_ids, ids))
        self.db.update_cluster_ids(assignments)

        touched = sorted({c for c in final_ids if c != -1})
        # Labeling already loads every touched cluster's full membership and
        # computes its centroid; both are exactly what topic grouping needs,
        # so they're returned rather than recomputed with a second pass over
        # the same rows.
        centroids, titles = self._generate_labels(touched)
        topic_result = ClusterTopicBuilder(self.db).run(centroids, titles)

        noise = int((labels == -1).sum())
        logger.info(
            f"Clustering done: {len(touched)} active clusters "
            f"({continued_count} continued, {new_count} new), {noise} noise"
        )
        return {
            'total': len(ids),
            'clusters': len(touched),
            'noise': noise,
            'new_clusters': new_count,
            'continued_clusters': continued_count,
            'topics': topic_result['topics'],
            'new_topics': topic_result['new_topics'],
            'continued_topics': topic_result['continued_topics'],
        }

    def _assign_persistent_ids(self, labels: np.ndarray, prev_ids: list,
                                matrix: np.ndarray):
        """
        Map this run's structural HDBSCAN labels onto persistent cluster ids:
        reuse an existing id where the new grouping matches an existing
        cluster's current centroid closely enough, otherwise mint a new one.
        Returns (final_ids_per_article, num_new_clusters, num_continued_clusters).
        """
        # Previous clusters' centroids, computed only from the members that
        # fall inside this run's window (their currently-active core).
        prev_groups: dict = {}
        for i, pid in enumerate(prev_ids):
            if pid is not None and pid != -1:
                prev_groups.setdefault(pid, []).append(i)
        prev_centroids = {
            pid: matrix[idxs].mean(axis=0) for pid, idxs in prev_groups.items()
        }

        unique_new = sorted(set(labels.tolist()) - {-1})
        new_centroids = {
            nl: matrix[labels == nl].mean(axis=0) for nl in unique_new
        }

        new_to_persistent = _match_to_previous(
            new_centroids, prev_centroids, CLUSTER_MATCH_THRESHOLD
        )
        used_new = set(new_to_persistent)

        next_id = self.db.get_max_cluster_id() + 1
        for nl in unique_new:
            if nl not in new_to_persistent:
                new_to_persistent[nl] = next_id
                next_id += 1

        final_ids = [
            -1 if l == -1 else new_to_persistent[l] for l in labels.tolist()
        ]
        return final_ids, len(unique_new) - len(used_new), len(used_new)

    def _generate_labels(self, cluster_ids: list):
        """
        For each touched persistent cluster, pick a label via a layered
        fallback (adapted from the sibling project's LLM -> most-central-
        title -> KeyBERT -> sanitized chain):
          1. LLM (Gemini) — only if CLUSTER_LABEL_USE_LLM is enabled.
          2. Most-central real headline — this project's pre-existing
             heuristic: 60% centroid similarity (semantic centrality) + 40%
             title quality, scored across the cluster's *full* membership
             (not just this run's window slice).
          3. KeyBERT keyphrase extraction — only if (2)'s best title scores
             below CLUSTER_LABEL_QUALITY_FLOOR (too weak to use as-is).
          4. Whichever of (2)/(3) produced something, with a trailing
             outlet-name attribution stripped — this tier is guaranteed to
             have *something*, so a cluster is never left unlabeled.

        Returns ({cluster_id: full-membership centroid}, {cluster_id: label})
        for the clusters it got that far on — the inputs topic grouping needs
        (see ClusterTopicBuilder), collected here so that pass doesn't have
        to re-read and re-reduce the same embeddings. Populated incrementally
        so a failure part-way still hands back whatever completed.
        """
        centroids: dict = {}
        labels_by_cluster: dict = {}
        try:
            for cid in cluster_ids:
                members = self.db.get_cluster_member_data(
                    cid, embedding_provider.model_id())
                if not members:
                    continue
                vecs = np.stack([
                    np.frombuffer(m['embedding'], dtype='float32') for m in members
                ])
                titles = [m['title'] for m in members]

                centroid = vecs.mean(axis=0)
                centroids[cid] = centroid
                sims = vecs @ centroid
                sim_min, sim_max = sims.min(), sims.max()
                sims_norm = (sims - sim_min) / (sim_max - sim_min) if sim_max > sim_min else np.ones(len(sims))

                quality = np.array([_title_quality(t) for t in titles])
                combined = 0.6 * sims_norm + 0.4 * quality
                order = np.argsort(combined)[::-1]
                best_title = titles[int(order[0])]
                best_quality = float(quality[int(order[0])])

                label = ''
                if CLUSTER_LABEL_USE_LLM:
                    label = _llm_label([titles[i] for i in order[:6]])

                if not label and best_quality < CLUSTER_LABEL_QUALITY_FLOOR:
                    label = _keybert_label(titles)

                label = _sanitize_label(label or best_title)
                if label:
                    self.db.save_cluster_label(cid, label)
                    labels_by_cluster[cid] = label

            logger.info(f"Cluster labels generated/refreshed for {len(cluster_ids)} clusters")
        except Exception as e:
            logger.warning(f"Cluster label generation failed: {e}")
        return centroids, labels_by_cluster


class ClusterTopicBuilder:
    """
    The grouping layer *above* event clusters: rolls several related events
    ("IMF tranche approved", "IMF conditions debated in parliament") up under
    one topic, so browsing doesn't mean scrolling a flat list of dozens of
    individually-correct but disconnected clusters.

    Implemented as a second clustering pass over event-cluster centroids
    rather than by loosening the event-level thresholds, deliberately: event
    clusters stay exactly as tightly scoped as they are, because the
    fact-checking/bias-detection consumer needs "one specific event, as
    covered by N outlets" as its unit. Topics are strictly additive on top.

    Cheap, because the centroids it clusters were already computed during
    labeling — there are tens-to-hundreds of them, not thousands of articles.

    Topic identity persists across runs by the same centroid-matching
    mechanism events use (_match_to_previous, one level up), so a topic keeps
    its topic_id — and therefore its URL — as its member events change.
    """

    def __init__(self, db: ArticleDatabase):
        self.db = db

    def run(self, centroids: dict, titles: dict) -> dict:
        """
        centroids: {cluster_id: full-membership centroid} for clusters active
                   this run;  titles: {cluster_id: event label}.
        Clusters that don't group with at least TOPIC_MIN_CLUSTERS - 1 others
        get topic_id=None — a standalone event is left standalone rather than
        forced into a topic it doesn't belong to.
        """
        empty = {'topics': 0, 'new_topics': 0, 'continued_topics': 0}
        if len(centroids) < TOPIC_MIN_CLUSTERS or len(centroids) < 2:
            logger.info(f"Too few active clusters ({len(centroids)}) to group into topics")
            return empty

        try:
            cluster_ids = sorted(centroids)
            # Unit-normalized: a cluster centroid's *norm* reflects how tight
            # that cluster is (mean of unit vectors), which must not weight
            # how much it pulls a topic centroid around.
            matrix = np.stack([_unit(centroids[cid]) for cid in cluster_ids])

            groups = self._group_clusters(cluster_ids, matrix)
            if not groups:
                logger.info("No multi-event topics found this run")
                self.db.set_cluster_topics([(None, cid) for cid in cluster_ids])
                self.db.delete_empty_topics()
                return empty

            new_centroids = {
                gid: _unit(np.stack([
                    matrix[cluster_ids.index(cid)] for cid in members
                ]).mean(axis=0))
                for gid, members in groups.items()
            }

            prev_centroids = self._previous_topic_centroids(cluster_ids, matrix)
            matched = _match_to_previous(
                new_centroids, prev_centroids, TOPIC_MATCH_THRESHOLD
            )

            assignments: list = []
            grouped_ids: set = set()
            for gid, members in groups.items():
                label = self._label_topic([titles.get(cid, '') for cid in members])
                topic_id = matched.get(gid)
                if topic_id is None:
                    topic_id = self.db.create_topic(label)
                else:
                    self.db.update_topic_title(topic_id, label)
                for cid in members:
                    assignments.append((topic_id, cid))
                    grouped_ids.add(cid)

            # Explicitly clear the rest: a cluster that was in a topic last
            # run and grouped with nothing this run must leave it, not linger.
            assignments += [(None, cid) for cid in cluster_ids if cid not in grouped_ids]
            self.db.set_cluster_topics(assignments)
            self.db.delete_empty_topics()

            continued = len(matched)
            new = len(groups) - continued
            logger.info(
                f"Topics: {len(groups)} covering {len(grouped_ids)} clusters "
                f"({continued} continued, {new} new)"
            )
            return {'topics': len(groups), 'new_topics': new, 'continued_topics': continued}
        except Exception as e:
            logger.warning(f"Topic grouping failed: {e}")
            return empty

    def _group_clusters(self, cluster_ids: list, matrix: np.ndarray) -> dict:
        """
        Agglomerative clustering over cluster centroids, cut at a fixed
        similarity rather than a fixed group count — the number of live
        topics isn't knowable in advance and varies run to run.

        Agglomerative rather than HDBSCAN here (the inverse of the choice at
        the event level) because this pass needs a directly interpretable
        "merge while average pairwise similarity stays above X" rule over a
        small point set, not density estimation: with only tens of centroids,
        density is poorly defined and HDBSCAN would push most of them to
        noise. Groups smaller than TOPIC_MIN_CLUSTERS are dropped here, which
        is what leaves standalone events unassigned.
        """
        splitter = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - TOPIC_SIMILARITY_THRESHOLD,
            metric='cosine',
            linkage='average',
        )
        labels = splitter.fit_predict(matrix)

        groups: dict = {}
        for label in sorted(set(labels.tolist())):
            members = [cluster_ids[i] for i, l in enumerate(labels) if l == label]
            if len(members) >= TOPIC_MIN_CLUSTERS:
                groups[int(label)] = members
        return groups

    def _previous_topic_centroids(self, cluster_ids: list, matrix: np.ndarray) -> dict:
        """Each existing topic's centroid, recomputed from whichever of its
        member clusters are active this run — its currently-active core, the
        same basis _assign_persistent_ids uses for previous event clusters."""
        prev_assignments = self.db.get_topic_assignments()
        index_of = {cid: i for i, cid in enumerate(cluster_ids)}

        prev_groups: dict = {}
        for cid, tid in prev_assignments.items():
            if cid in index_of:
                prev_groups.setdefault(tid, []).append(index_of[cid])
        return {
            tid: _unit(matrix[idxs].mean(axis=0))
            for tid, idxs in prev_groups.items()
        }

    @staticmethod
    def _label_topic(event_titles: list) -> str:
        """
        LLM -> KeyBERT keyphrase -> shortest member headline.

        The tiers are ordered differently from event labeling on purpose: a
        topic wants a *heading* ("IMF programme review"), not a headline, so
        keyphrase extraction is the natural default tier here where for an
        event it's only a weak-title fallback. The last resort is the
        shortest member headline, shortest being the least
        event-specific — a full headline naming one event reads badly as the
        name of a group of several.
        """
        titles = [t for t in event_titles if t]
        if not titles:
            return 'Untitled topic'

        label = ''
        if CLUSTER_LABEL_USE_LLM:
            label = _llm_topic_label(titles)
        if not label:
            label = _keybert_label(titles)
        if not label:
            label = min(titles, key=len)
        return _sanitize_label(label) or 'Untitled topic'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    result = ArticleClusterer().run()
    print(result)
