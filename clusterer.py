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

import hashlib
import logging
import math
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
    TOPIC_LABEL_USE_LLM, TOPIC_LABEL_BATCH,
    TOPIC_JOIN_SIMILARITY, TOPIC_JOIN_LLM_BATCH, TOPIC_JOIN_CANDIDATES,
    TOPIC_JOIN_MAX_CANDIDATES,
    TOPIC_MERGE_FLOOR, TOPIC_MERGE_MAX_PAIRS, TOPIC_MERGE_MAX_SIZE,
    CLUSTER_PROMOTE_SOLO, CLUSTER_OUTLIER_MARGIN, CLUSTER_OUTLIER_FLOOR,
    CLUSTER_EVICT_OUTLIERS, CLUSTER_EVICT_LLM_LOW, CLUSTER_EVICT_LLM_HIGH,
    CLUSTER_EVICT_LLM_BATCH,
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


def _llm_validate_joins(cases: list) -> list:
    """
    Decide, for each (candidate_title, [core_titles]) case, whether the
    candidate belongs to that topic. Returns a list of bools the same length
    as `cases`.

    Why an LLM rather than another threshold: measured on this corpus,
    neither cosine nor IDF separates "Iran says it attacked US Navy ships"
    (belongs with the Iran-US topic) from "NDB Bank Shapes Supply Chain
    Finance" (does not belong with DFCC Bank's award coverage). Both pairs
    score ~0.91 cosine, and their shared terms rank the wrong way round.
    Telling a company press release apart from a developing story needs
    world knowledge, which is exactly what this tier already provides for
    labelling.

    Fails CLOSED — any error, any unparseable reply, and the affected cases
    come back False. A cluster that stays out of a topic is a missing
    answer; one let in unverified is a wrong one.
    """
    verdicts = [False] * len(cases)
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not cases:
        return verdicts

    for start in range(0, len(cases), TOPIC_JOIN_LLM_BATCH):
        chunk = cases[start:start + TOPIC_JOIN_LLM_BATCH]
        blocks = []
        for n, (candidate, core) in enumerate(chunk, 1):
            members = '\n'.join(f'  - {t}' for t in core[:5])
            blocks.append(f"CASE {n}\nTOPIC HEADLINES:\n{members}\n"
                          f"CANDIDATE: {candidate}")
        prompt = (
            "Each case below shows the headlines already filed under one "
            "Sri Lankan news topic, plus a candidate headline.\n\n"
            "Answer yes only if the candidate is about the SAME ongoing "
            "story, organisation, or specific subject as the topic. Answer "
            "no if it merely comes from the same sector or genre — two "
            "different banks' press releases, or two unrelated corporate "
            "announcements, are NOT the same topic.\n\n"
            + '\n\n'.join(blocks)
            + f"\n\nReply with only a JSON array of {len(chunk)} strings, "
              'each "yes" or "no", in case order. No explanation.'
        )
        for model in GEMINI_LABEL_MODELS:
            try:
                r = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                    headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                    json={'contents': [{'parts': [{'text': prompt}]}]},
                    timeout=40,
                )
                if r.status_code >= 400:
                    logger.debug(f"Join check via {model}: HTTP {r.status_code}")
                    continue
                text = r.json()['candidates'][0]['content']['parts'][0]['text']
                answers = re.findall(r'"(yes|no)"', text, flags=re.I)
                if len(answers) != len(chunk):
                    logger.debug(f"Join check via {model}: got {len(answers)} "
                                 f"answers for {len(chunk)} cases")
                    continue
                for i, a in enumerate(answers):
                    verdicts[start + i] = (a.lower() == 'yes')
                break
            except Exception as e:
                logger.debug(f"Join check via {model} failed: {e}")
                continue
    return verdicts


def _llm_topic_labels_batch(groups_titles: list) -> list:
    """
    Name many topics in one request. Returns a list the same length as the
    input; entries the model did not name come back ''.

    Why this exists: labelling used to be one HTTP request per topic. That
    was fine when a run relabelled "a handful" of topics, but backfilling
    the corpus changed 250 topics at once and blew straight through the free
    tier's per-day request quota — which is counted per REQUEST, not per
    topic. The joins, which run afterwards, then got nothing but 429s and
    silently accepted zero of 352 candidates.

    Batching turns ~250 requests into ~17 and makes the daily budget a
    non-issue for normal runs.
    """
    labels = [''] * len(groups_titles)
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not groups_titles:
        return labels

    for start in range(0, len(groups_titles), TOPIC_LABEL_BATCH):
        chunk = groups_titles[start:start + TOPIC_LABEL_BATCH]
        blocks = []
        for n, titles in enumerate(chunk, 1):
            body = '\n'.join(f'  - {t}' for t in titles[:6])
            blocks.append(f"TOPIC {n}:\n{body}")
        prompt = (
            "Each numbered group below contains headlines for separate news "
            "events that belong to one ongoing story or theme in Sri Lankan "
            "news.\n\nFor each group, write a short topic name (2-5 words) "
            "— the kind of heading a news site would file those stories "
            "under. Do not mention any outlet's name.\n\n"
            + '\n\n'.join(blocks)
            + f"\n\nReply with only a JSON array of {len(chunk)} strings, "
              "one topic name per group, in order. No explanation."
        )
        for model in GEMINI_LABEL_MODELS:
            try:
                r = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                    headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                    json={'contents': [{'parts': [{'text': prompt}]}]},
                    timeout=60,
                )
                if r.status_code >= 400:
                    logger.debug(f"Batch topic label via {model}: HTTP {r.status_code}")
                    continue
                text = r.json()['candidates'][0]['content']['parts'][0]['text']
                names = re.findall(r'"([^"]{2,80})"', text)
                if len(names) != len(chunk):
                    logger.debug(f"Batch topic label via {model}: {len(names)} "
                                 f"names for {len(chunk)} topics")
                    continue
                for i, name in enumerate(names):
                    labels[start + i] = name.strip()
                break
            except Exception as e:
                logger.debug(f"Batch topic label via {model} failed: {e}")
                continue
    return labels


def _llm_validate_merges(cases: list) -> list:
    """
    Decide, for each (titles_a, titles_b) pair of topics, whether the two are
    really one topic. Returns a list of bools.

    Complete linkage at TOPIC_SIMILARITY_THRESHOLD is strict on purpose, and
    strictness splits: measured on this corpus it produced "Judicial
    Retirement Amendment Controversy" (11 events) alongside "Judicial
    Retirement Age Debate" (6 events), and "Sri Lanka Economic Recovery"
    (25) alongside "Sri Lanka Economic Outlook" (6). Both pairs are one
    story each.

    Lowering the formation threshold to fix that is exactly the move this
    project already measured and rejected — it merges unrelated events long
    before it merges these. So formation stays strict and a judge decides
    the borderline cases, the same shape as _llm_validate_joins.

    Fails CLOSED: on any error the pair stays split, which is the state the
    system is already in.
    """
    verdicts = [False] * len(cases)
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not cases:
        return verdicts

    for start in range(0, len(cases), TOPIC_JOIN_LLM_BATCH):
        chunk = cases[start:start + TOPIC_JOIN_LLM_BATCH]
        blocks = []
        for n, (a, b) in enumerate(chunk, 1):
            left = '\n'.join(f'  - {t}' for t in a[:4])
            right = '\n'.join(f'  - {t}' for t in b[:4])
            blocks.append(f"CASE {n}\nTOPIC A:\n{left}\nTOPIC B:\n{right}")
        prompt = (
            "Each case shows headlines from two separate Sri Lankan news "
            "topics.\n\nAnswer yes only if A and B are the SAME ongoing "
            "story and should be merged into one topic. Answer no if they "
            "are merely in the same sector or subject area — different "
            "companies, different court cases, or general coverage of an "
            "industry are NOT the same topic.\n\n"
            + '\n\n'.join(blocks)
            + f"\n\nReply with only a JSON array of {len(chunk)} strings, "
              'each "yes" or "no", in case order. No explanation.'
        )
        for model in GEMINI_LABEL_MODELS:
            try:
                r = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                    headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                    json={'contents': [{'parts': [{'text': prompt}]}]},
                    timeout=40,
                )
                if r.status_code >= 400:
                    logger.debug(f"Merge check via {model}: HTTP {r.status_code}")
                    continue
                text = r.json()['candidates'][0]['content']['parts'][0]['text']
                answers = re.findall(r'"(yes|no)"', text, flags=re.I)
                if len(answers) != len(chunk):
                    continue
                for i, a in enumerate(answers):
                    verdicts[start + i] = (a.lower() == 'yes')
                break
            except Exception as e:
                logger.debug(f"Merge check via {model} failed: {e}")
                continue
    return verdicts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ── embedding cache ───────────────────────────────────────────────────────
# The scheduler is one long-lived process, so vectors fetched on one run are
# still in memory on the next. Only genuinely new articles are downloaded.
#
# Why this exists: measured against the live database, a clustering run
# pulled 32 MB of embeddings across three queries — the 14-day window, every
# cluster's members for the topic pass, and every cluster's members again for
# labelling. At hourly runs that is 22.7 GB/month against a 5 GB network
# allowance, and almost none of it changes between runs.
#
# Bounded by construction: one entry per article that has ever been in a
# clustering window, at 3 KB each. The whole corpus is ~12 MB, and articles
# are only ever added, never mutated — an embedding is rewritten only when
# the provider changes, which also changes model_id and invalidates the
# cache wholesale.
_EMB_CACHE: dict = {}
_EMB_CACHE_MODEL: str = ''


def _cached_embeddings(db, article_ids: list, model_id: str) -> dict:
    """{article_id: np.ndarray} for the ids given, fetching only misses."""
    global _EMB_CACHE, _EMB_CACHE_MODEL
    if _EMB_CACHE_MODEL != model_id:
        # A provider change puts every stored vector in a different space,
        # so nothing cached under the old model is comparable.
        _EMB_CACHE = {}
        _EMB_CACHE_MODEL = model_id

    missing = [i for i in article_ids if i not in _EMB_CACHE]
    if missing:
        raw = db.get_embeddings_by_ids(missing)
        for aid, blob in raw.items():
            _EMB_CACHE[aid] = np.frombuffer(blob, dtype='float32')
        logger.info(f"Embedding cache: {len(article_ids) - len(missing)} hits, "
                    f"{len(raw)} fetched")
    return {i: _EMB_CACHE[i] for i in article_ids if i in _EMB_CACHE}


def _unit_rows(m: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize, so a matmul yields cosine similarity."""
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


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

    The similarity matrix is computed with one matmul rather than a Python
    loop over every (new, previous) pair. Promoting single-outlet stories to
    their own clusters took that cross product from ~100k pairs to ~750k,
    and only the pairs above the threshold — a few thousand — ever need
    sorting.
    """
    if not new_centroids or not prev_centroids:
        return {}

    new_labels = list(new_centroids)
    prev_labels = list(prev_centroids)
    A = _unit_rows(np.stack([new_centroids[nl] for nl in new_labels]))
    B = _unit_rows(np.stack([prev_centroids[pid] for pid in prev_labels]))
    sims = A @ B.T

    rows, cols = np.nonzero(sims >= threshold)
    if len(rows) == 0:
        return {}
    order = np.argsort(-sims[rows, cols])

    matched: dict = {}
    used_prev: set = set()
    for k in order:
        i, j = int(rows[k]), int(cols[k])
        nl, pid = new_labels[i], prev_labels[j]
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


def _llm_validate_members(cases: list) -> list:
    """
    For each (candidate_title, [sibling_titles]) decide whether the candidate
    covers the same event as its cluster. Returns a list of bools.

    Only the borderline band reaches this. Measured on live data, similarity
    alone cannot decide there: "China Jails Evergrande Boss for Life" scores
    0.939 against a cluster of the same story, and "PickMe launches Colombo
    Connect" scores 0.934 against a cluster it does not belong to. 0.005
    apart, opposite answers.

    Fails toward KEEPING. Every other LLM tier in this project fails closed by
    doing nothing; here "doing nothing" must mean leaving the article in
    place, because an evicted article becomes a solo cluster and its event
    loses an outlet — which is more expensive than one odd headline staying.
    """
    verdicts = [True] * len(cases)          # default: keep
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key or not cases:
        return verdicts

    for start in range(0, len(cases), CLUSTER_EVICT_LLM_BATCH):
        chunk = cases[start:start + CLUSTER_EVICT_LLM_BATCH]
        blocks = []
        for n, (candidate, peers) in enumerate(chunk, 1):
            others = '\n'.join(f'  - {t}' for t in peers[:4])
            blocks.append(f"CASE {n}\nGROUP:\n{others}\nARTICLE: {candidate}")
        prompt = (
            "Each case shows headlines already grouped as ONE news event, "
            "plus one more article.\n\nAnswer yes if the article reports the "
            "SAME event as the group — even if it is worded very differently, "
            "or covers a later development in the same story. Answer no only "
            "if it is a different event, a different organisation, or a "
            "different subject that merely sounds similar.\n\n"
            + '\n\n'.join(blocks)
            + f"\n\nReply with only a JSON array of {len(chunk)} strings, "
              'each "yes" or "no", in case order. No explanation.'
        )
        for model in GEMINI_LABEL_MODELS:
            try:
                r = requests.post(
                    f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
                    headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
                    json={'contents': [{'parts': [{'text': prompt}]}]},
                    timeout=40,
                )
                if r.status_code >= 400:
                    logger.debug(f"Member check via {model}: HTTP {r.status_code}")
                    continue
                text = r.json()['candidates'][0]['content']['parts'][0]['text']
                answers = re.findall(r'"(yes|no)"', text, flags=re.I)
                if len(answers) != len(chunk):
                    continue
                for i, a in enumerate(answers):
                    verdicts[start + i] = (a.lower() == 'yes')
                break
            except Exception as e:
                logger.debug(f"Member check via {model} failed: {e}")
                continue
    return verdicts


def _evict_outliers(clusters: dict, matrix: np.ndarray, titles: list,
                    ids: list, db) -> dict:
    """
    Remove members HDBSCAN absorbed into a cluster they do not belong to.

    Each member is scored by its STRONGEST link to any sibling — "is anyone
    here actually covering the same event as you?" — and then:

        >= CLUSTER_EVICT_LLM_HIGH   keep, no request
        <  CLUSTER_EVICT_LLM_LOW    evict, no request
        in between                  ask the LLM

    The middle band exists because a relative rule was tried first and
    measured as roughly 50% wrong: near-duplicate wire copy inside a cluster
    drags the median link to ~1.0, lifting the bar until a legitimate
    paraphrase of the same event falls below it. No margin fixed it — from
    0.05 to 0.15 the wrong evictions and the right ones moved together.

    Evicted members are not lost: they fall to noise and CLUSTER_PROMOTE_SOLO
    turns each into its own single-article cluster.
    """
    if not CLUSTER_EVICT_OUTLIERS:
        return clusters

    # Collect the undecided cases across ALL clusters first, so they can go
    # to the model in a few batched requests rather than one per cluster.
    pending = []          # (label, index_into_cluster, article_id, peers_hash)
    cases = []
    verdicts_by_label: dict = {}
    for label, indices in clusters.items():
        if len(indices) < 3:
            continue
        vecs = matrix[indices]
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1.0)
        strongest = sims.max(axis=1)
        centroid = _unit(vecs.mean(axis=0))
        centrality = vecs @ centroid
        decided = {}
        for pos, (idx, link) in enumerate(zip(indices, strongest)):
            if link >= CLUSTER_EVICT_LLM_HIGH:
                decided[pos] = True
            elif link < CLUSTER_EVICT_LLM_LOW:
                decided[pos] = False
            else:
                # Show the cluster's MOST CENTRAL members, not whatever came
                # first. Picking by index order let one outlier illustrate
                # the group for another — measured on a real cluster of NPP
                # rally reports, an unrelated parliament speech was offered
                # as context for an SLPP article, and the model quite
                # reasonably concluded the group was vague enough to accept
                # it. The question has to show what the cluster IS.
                order = np.argsort(-centrality)
                peers = [titles[indices[j]] for j in order
                         if indices[j] != idx][:4]
                if not peers:
                    decided[pos] = True
                    continue
                peers_hash = hashlib.md5('|'.join(sorted(peers)).encode()).hexdigest()
                pending.append((label, pos, ids[idx], peers_hash))
                cases.append((titles[idx], peers))
        verdicts_by_label[label] = decided

    if cases:
        cached = db.get_member_verdicts()
        ask_idx = [k for k, p in enumerate(pending)
                   if (p[2], p[3]) not in cached]
        for k, p in enumerate(pending):
            hit = cached.get((p[2], p[3]))
            if hit is not None:
                verdicts_by_label[p[0]][p[1]] = hit
        if ask_idx:
            answers = _llm_validate_members([cases[k] for k in ask_idx])
            fresh = []
            for k, ok in zip(ask_idx, answers):
                label, pos, aid, ph = pending[k]
                verdicts_by_label[label][pos] = ok
                fresh.append((aid, ph, bool(ok)))
            db.save_member_verdicts(fresh)
        logger.info(f"Member check: {len(cases)} borderline, "
                    f"{len(cases) - len(ask_idx)} from cache, {len(ask_idx)} asked")

    kept = {}
    evicted = 0
    for label, indices in clusters.items():
        if len(indices) < 3:
            kept[label] = indices
            continue
        decided = verdicts_by_label.get(label, {})
        survivors = [idx for pos, idx in enumerate(indices)
                     if decided.get(pos, True)]
        evicted += len(indices) - len(survivors)
        if len(survivors) >= HDBSCAN_MIN_CLUSTER:
            kept[label] = survivors
    if evicted:
        logger.info(f"Evicted {evicted} articles that do not cover their "
                    f"cluster's event")
    return kept


def _refine_clusters(labels: np.ndarray, matrix: np.ndarray,
                      sources: list, dates: list, titles: list,
                      ids: list, db) -> np.ndarray:
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

    # Last: shed members the splits above could not catch, because they are
    # individually wrong rather than making the group oversized or loose.
    clusters = _evict_outliers(clusters, matrix, titles, ids, db)

    refined = np.full(len(labels), -1, dtype=int)
    next_label = 0
    for indices in clusters.values():
        if len(indices) < HDBSCAN_MIN_CLUSTER:
            continue  # too small to survive refinement; falls back to noise
        for i in indices:
            refined[i] = next_label
        next_label += 1
    return refined


def _promote_solo_stories(labels: np.ndarray) -> np.ndarray:
    """
    Turn every remaining noise point into its own one-article cluster.

    Runs AFTER _refine_clusters and BEFORE _assign_persistent_ids, so solo
    clusters get persistent ids on the same footing as any other cluster and
    keep them across runs.

    Safe against identity theft from real clusters: _match_to_previous is
    greedy highest-similarity-first and one-to-one, so a multi-article
    cluster matches its own previous centroid long before a lone article
    could claim that id.
    """
    out = labels.copy()
    solo = np.where(out == -1)[0]
    next_label = (int(out.max()) + 1) if (out >= 0).any() else 0
    for i in solo:
        out[i] = next_label
        next_label += 1
    return out


class ArticleClusterer:
    def __init__(self):
        self.db = ArticleDatabase()

    def run(self) -> dict:
        model_id = embedding_provider.model_id()
        rows = self.db.get_recent_embedding_meta(CLUSTER_WINDOW_DAYS, model_id)
        if len(rows) < 4:
            logger.info(f"Too few embedded articles in the last "
                        f"{CLUSTER_WINDOW_DAYS}d ({len(rows)}) to cluster")
            return {'total': len(rows), 'clusters': 0, 'noise': len(rows),
                    'new_clusters': 0, 'continued_clusters': 0}

        # Vectors come from the in-process cache; only ids it has never seen
        # are fetched. Anything still missing (a row deleted between the two
        # queries) is dropped rather than crashing the run.
        vectors = _cached_embeddings(self.db, [r['id'] for r in rows], model_id)
        rows = [r for r in rows if r['id'] in vectors]
        if len(rows) < 4:
            logger.info("Too few embeddings available to cluster")
            return {'total': len(rows), 'clusters': 0, 'noise': len(rows),
                    'new_clusters': 0, 'continued_clusters': 0}
        ids = [r['id'] for r in rows]
        prev_ids = [r['cluster_id'] for r in rows]
        sources = [r['source'] for r in rows]
        titles = [r.get('title') or '' for r in rows]
        dates = [r['effective_date'] for r in rows]
        matrix = np.stack([vectors[i] for i in ids])

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
        labels = _refine_clusters(labels, matrix, sources, dates,
                                  titles, ids, self.db)

        # Human overrides, applied after refinement so they cannot be undone
        # by a later merge. An excluded article falls to noise and, with
        # CLUSTER_PROMOTE_SOLO, becomes its own single-article cluster — it
        # is removed from a group it does not belong to, not from the corpus.
        excluded = self.db.get_excluded_article_ids()
        if excluded:
            hits = 0
            for i, aid in enumerate(ids):
                if aid in excluded and labels[i] != -1:
                    labels[i] = -1
                    hits += 1
            if hits:
                logger.info(f"Applied {hits} manual cluster exclusions")

        solo_count = 0
        if CLUSTER_PROMOTE_SOLO:
            solo_count = int((labels == -1).sum())
            labels = _promote_solo_stories(labels)

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
        # Topics are computed over EVERY cluster, not just this run's window.
        # Event clusters are windowed on purpose (bounds run time, and a past
        # event does not change), but a topic is a relationship between
        # events that gets re-derived under current thresholds — so it has to
        # see all of them or old assignments freeze in place.
        topic_result = ClusterTopicBuilder(self.db).run(centroids, titles)

        noise = int((labels == -1).sum())
        if solo_count:
            logger.info(f"Promoted {solo_count} single-outlet stories to "
                        f"their own clusters")
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
            model_id = embedding_provider.model_id()
            bulk = self.db.get_cluster_member_titles(cluster_ids, model_id)
            all_ids = [m['id'] for ms in bulk.values() for m in ms]
            vectors = _cached_embeddings(self.db, all_ids, model_id)
            for cid in cluster_ids:
                members = [m for m in bulk.get(cid, []) if m['id'] in vectors]
                if not members:
                    continue
                vecs = np.stack([vectors[m['id']] for m in members])
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
                    labels_by_cluster[cid] = label

            # Written once rather than per cluster: at ~850 clusters a run,
            # 850 separate round trips to a hosted database cost far more
            # than the labelling itself.
            self.db.save_cluster_labels_bulk(list(labels_by_cluster.items()))
            logger.info(f"Cluster labels generated/refreshed for {len(cluster_ids)} clusters")
        except Exception as e:
            logger.warning(f"Cluster label generation failed: {e}")
            # Partial progress is still worth keeping — the loop populates
            # incrementally and the next run resumes from whatever landed.
            try:
                self.db.save_cluster_labels_bulk(list(labels_by_cluster.items()))
            except Exception:
                pass
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
        centroids/titles: what this run computed for the clusters it touched.
        Both are EXTENDED here with every other cluster in the corpus, so the
        grouping covers all of them.

        Clusters that don't group with at least TOPIC_MIN_CLUSTERS - 1 others
        get topic_id=None — a standalone event is left standalone rather than
        forced into a topic it doesn't belong to.
        """
        empty = {'topics': 0, 'new_topics': 0, 'continued_topics': 0}
        titles = dict(titles)
        titles.update(self.db.get_cluster_labels())

        # ONE centroid source, for every cluster.
        #
        # The caller's centroids are computed over the CLUSTERING WINDOW —
        # only the members from the last CLUSTER_WINDOW_DAYS. Centroids built
        # here cover full membership. Mixing the two silently breaks complete
        # linkage's guarantee: it holds against the matrix it was given, so
        # grouping windowed centroids and then reading back full ones showed
        # a worst intra-topic pair of 0.854 against a 0.90 threshold. Full
        # membership wins; the caller's are kept only as a fallback for
        # clusters with no currently-qualifying articles.
        merged = dict(centroids)
        merged.update(self._all_centroids())
        centroids = merged
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
            groups = self._merge_similar_topics(groups, cluster_ids, matrix, titles)
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

            # Relabelling is the only expensive step here (it may call an
            # LLM), so it runs only for topics whose membership actually
            # changed. Typically a handful per run rather than all of them.
            prev_hashes = self.db.get_topic_member_hashes()

            # Decide what needs a label BEFORE creating anything, so every
            # label for this run can be fetched in one batched pass. Doing it
            # inside the loop meant one HTTP request per topic, which is what
            # exhausted the free-tier daily request quota and left the join
            # stage with nothing but 429s.
            plan = []
            for gid, members in groups.items():
                member_hash = hashlib.md5(
                    ','.join(str(c) for c in sorted(members)).encode()
                ).hexdigest()
                topic_id = matched.get(gid)
                needs_label = (topic_id is None
                               or prev_hashes.get(topic_id) != member_hash)
                plan.append((gid, members, member_hash, topic_id, needs_label))

            pending = [p for p in plan if p[4]]
            fresh: dict = {}
            if pending and (TOPIC_LABEL_USE_LLM or CLUSTER_LABEL_USE_LLM):
                batch_titles = [
                    [t for t in (titles.get(cid, '') for cid in p[1]) if t]
                    for p in pending
                ]
                for p, name in zip(pending, _llm_topic_labels_batch(batch_titles)):
                    if name:
                        fresh[p[0]] = _sanitize_label(name) or ''

            assignments: list = []
            grouped_ids: set = set()
            gid_to_topic: dict = {}
            relabelled = 0
            kept = 0
            for gid, members, member_hash, topic_id, needs_label in plan:
                label = fresh.get(gid)

                if topic_id is None:
                    # A brand new topic has no name to keep, so the local
                    # tiers are better than nothing here.
                    if not label:
                        label = self._label_topic(
                            [titles.get(cid, '') for cid in members])
                    topic_id = self.db.create_topic(label, member_hash)
                    relabelled += 1
                elif needs_label and label:
                    self.db.update_topic_title(topic_id, label, member_hash)
                    relabelled += 1
                elif needs_label:
                    # An EXISTING topic whose new name could not be
                    # generated keeps the name it has. The local fallback
                    # names a topic after one arbitrary member headline, so
                    # applying it here would actively downgrade a good name
                    # every time the quota ran out — and member_hash is
                    # deliberately left unchanged so the next run retries.
                    kept += 1

                gid_to_topic[gid] = topic_id
                for cid in members:
                    assignments.append((topic_id, cid))
                    grouped_ids.add(cid)

            # Widen coverage: clusters that formed no topic of their own but
            # clearly belong to one get attached here. Without this the topic
            # layer reached only ~53% of clustered articles, because complete
            # linkage excludes a cluster for failing against a single member.
            core_count = len(grouped_ids)
            gid_hash = {p[0]: p[2] for p in plan}
            for topic_id, cid in self._join_leftovers(
                    cluster_ids, matrix, groups, gid_to_topic, grouped_ids,
                    titles, gid_hash):
                assignments.append((topic_id, cid))
                grouped_ids.add(cid)

            # Explicitly clear the rest: a cluster that was in a topic last
            # run and grouped with nothing this run must leave it, not linger.
            assignments += [(None, cid) for cid in cluster_ids if cid not in grouped_ids]

            # ...and clear clusters this pass could not even see. A cluster
            # whose articles have no body, or still carry a previous
            # embedding model, produces no centroid — so it is not in
            # cluster_ids and the line above never touches it. Measured: 72
            # such clusters kept a topic_id assigned under thresholds that no
            # longer exist, propping up 24 topics with no live member at all.
            # An assignment that cannot be re-derived is not evidence.
            stale = [cid for cid in self.db.get_topic_assignments()
                     if cid not in set(cluster_ids)]
            if stale:
                logger.info(f"Clearing {len(stale)} topic assignments that "
                            f"cannot be re-derived (no current embedding)")
                assignments += [(None, cid) for cid in stale]
            self.db.set_cluster_topics(assignments)
            self.db.delete_empty_topics()
            self.db.prune_join_verdicts()

            continued = len(matched)
            new = len(groups) - continued
            logger.info(
                f"Topics: {len(groups)} covering {len(grouped_ids)} clusters "
                f"({core_count} core + {len(grouped_ids) - core_count} joined; "
                f"{continued} continued, {new} new, {relabelled} relabelled"
                + (f", {kept} kept old name" if kept else "") + ")"
            )
            return {'topics': len(groups), 'new_topics': new, 'continued_topics': continued}
        except Exception as e:
            logger.warning(f"Topic grouping failed: {e}")
            return empty

    def _all_centroids(self) -> dict:
        """Full-membership centroid for every cluster in the corpus.
        One query plus numpy grouping — a few hundred centroids over a few
        thousand vectors, so cheap relative to the clustering itself."""
        model_id = embedding_provider.model_id()
        membership = self.db.get_cluster_membership(model_id)
        wanted = [aid for ids in membership.values() for aid in ids]
        vectors = _cached_embeddings(self.db, wanted, model_id)

        out = {}
        for cid, aids in membership.items():
            vecs = [vectors[a] for a in aids if a in vectors]
            if not vecs:
                continue
            c = np.stack(vecs).mean(axis=0)
            out[cid] = c / (np.linalg.norm(c) + 1e-9)
        return out

    def _merge_similar_topics(self, groups: dict, cluster_ids: list,
                              matrix: np.ndarray, titles: dict) -> dict:
        """
        Merge topic groups that complete linkage split but are one story.

        Runs BEFORE topic ids are matched or labels generated, so everything
        downstream — identity matching, member_hash, labelling — sees the
        merged shape and needs no special handling.
        """
        if len(groups) < 2:
            return groups
        index = {c: i for i, c in enumerate(cluster_ids)}
        gids = sorted(groups)
        G = _unit_rows(np.stack([
            np.stack([matrix[index[c]] for c in groups[g]]).mean(axis=0)
            for g in gids
        ]))
        sims = G @ G.T
        iu = np.triu_indices(len(gids), 1)
        pairs = [(int(i), int(j)) for i, j in zip(*iu)
                 if sims[i, j] >= TOPIC_MERGE_FLOOR]
        if not pairs:
            return groups
        pairs.sort(key=lambda p: -sims[p[0], p[1]])
        pairs = pairs[:TOPIC_MERGE_MAX_PAIRS]

        cases = [
            ([t for t in (titles.get(c, '') for c in groups[gids[i]]) if t],
             [t for t in (titles.get(c, '') for c in groups[gids[j]]) if t])
            for i, j in pairs
        ]
        verdicts = _llm_validate_merges(cases)

        # Union-find, so a chain of accepted pairs collapses into one topic
        # rather than depending on the order they were examined.
        parent = list(range(len(gids)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        sizes = {k: len(groups[g]) for k, g in enumerate(gids)}
        merged = 0
        for (i, j), ok in zip(pairs, verdicts):
            if not ok:
                continue
            a, b = find(i), find(j)
            if a == b:
                continue
            if sizes[a] + sizes[b] > TOPIC_MERGE_MAX_SIZE:
                continue        # refuse chains that would build a mega-topic
            parent[a] = b
            sizes[b] += sizes[a]
            merged += 1
        if not merged:
            logger.info(f"Merge check: 0/{len(pairs)} topic pairs merged")
            return groups

        out: dict = {}
        for k, g in enumerate(gids):
            out.setdefault(find(k), []).extend(groups[g])
        logger.info(f"Merge check: {merged}/{len(pairs)} topic pairs merged "
                    f"({len(gids)} -> {len(out)} topics)")
        return out


    def _join_leftovers(self, cluster_ids, matrix, groups, gid_to_topic,
                        core_ids: set, titles: dict, gid_hash: dict) -> list:
        """
        Attach clusters that formed no topic of their own to an existing one.

        Returns [(topic_id, cluster_id)]. Runs AFTER topics are formed and
        labelled, and deliberately does not feed back into either: a joined
        cluster never moves a topic centroid, never changes its member_hash,
        and never influences its title. Topics stay *defined* by the strict
        core, so coverage grows without the meaning of a topic drifting.

        A cheap cosine pre-filter picks at most one candidate topic per
        cluster; Gemini makes the actual call. See config.TOPIC_JOIN_*.
        """
        leftovers = [c for c in cluster_ids if c not in core_ids]
        if not leftovers or not groups:
            return []

        index = {c: i for i, c in enumerate(cluster_ids)}
        gids = sorted(groups)
        topic_centroids = np.stack([
            _unit(np.stack([matrix[index[c]] for c in groups[g]]).mean(axis=0))
            for g in gids
        ])

        # Cosine only narrows the field. It cannot make the decision — the
        # whole leftover distribution sits in 0.83-0.92 under these
        # embeddings — but it cheaply caps how many cases reach the LLM.
        sims = np.stack([matrix[index[c]] for c in leftovers]) @ topic_centroids.T
        cases, pairs, scores, hashes = [], [], [], []
        for row, cid in zip(sims, leftovers):
            best = None
            for gi in np.argsort(-row)[:TOPIC_JOIN_CANDIDATES]:
                if row[gi] < TOPIC_JOIN_SIMILARITY:
                    break           # sorted, so nothing further can qualify
                best = int(gi)
                break
            if best is None:
                continue
            core_titles = [titles.get(c, '') for c in groups[gids[best]]]
            cases.append((titles.get(cid, ''), [t for t in core_titles if t]))
            pairs.append((gid_to_topic[gids[best]], cid))
            scores.append(float(row[best]))
            # Key the remembered answer on BOTH sides of the question: the
            # topic's membership AND the exact headline the model is shown.
            # Keying on the topic alone was not enough, and keying on "was
            # this cluster touched this run" was useless — with solo stories
            # promoted, essentially every cluster is touched every run, so
            # the cache never hit once in two runs.
            hashes.append(hashlib.md5(
                (gid_hash.get(gids[best], '') + '|' + titles.get(cid, '')).encode()
            ).hexdigest())

        if not cases:
            return []

        # Reuse the remembered answer where nothing has changed. The key
        # covers both sides of the question — the topic's membership and the
        # cluster's headline — so any change to either re-asks.
        cached = self.db.get_join_verdicts()
        joined, ask = [], []
        for k, (topic_id, cid) in enumerate(pairs):
            hit = cached.get((cid, topic_id, hashes[k]))
            if hit is not None:
                if hit:
                    joined.append((topic_id, cid))
                continue
            ask.append(k)
        if len(ask) < len(cases):
            logger.info(f"Join check: {len(cases) - len(ask)} of {len(cases)} "
                        f"answered from cache")
        cases = [cases[k] for k in ask]
        pairs = [pairs[k] for k in ask]
        scores = [scores[k] for k in ask]
        hashes = [hashes[k] for k in ask]
        if not cases:
            logger.info(f"Join check: {len(joined)} joined, all from cache")
            return joined

        # Bound the request cost, strongest candidates first. Anything not
        # reached is asked on a later run, and once asked the answer is
        # cached, so the queue drains rather than repeating.
        if len(cases) > TOPIC_JOIN_MAX_CANDIDATES:
            order = sorted(range(len(cases)), key=lambda k: -scores[k])
            keep = sorted(order[:TOPIC_JOIN_MAX_CANDIDATES])
            logger.info(f"Join check: {len(cases)} candidates, testing "
                        f"{len(keep)} this run (rest deferred)")
            # hashes MUST be sliced with the same indices — it is what the
            # verdict is stored under. Truncating cases/pairs alone left it
            # misaligned, so every answer was saved against another
            # candidate's key and the cache could never hit.
            cases = [cases[k] for k in keep]
            pairs = [pairs[k] for k in keep]
            hashes = [hashes[k] for k in keep]
        assert len(pairs) == len(hashes) == len(cases)
        verdicts = _llm_validate_joins(cases)
        fresh = [p for p, ok in zip(pairs, verdicts) if ok]
        self.db.save_join_verdicts([
            (cid, topic_id, h, bool(ok))
            for (topic_id, cid), h, ok in zip(pairs, hashes, verdicts)
        ])
        logger.info(f"Join check: {len(fresh)}/{len(cases)} newly asked accepted "
                    f"({len(joined)} more from cache)")
        return joined + fresh

    def _group_clusters(self, cluster_ids: list, matrix: np.ndarray) -> dict:
        """
        Agglomerative clustering over cluster centroids, cut at a fixed
        similarity rather than a fixed group count — the number of live
        topics isn't knowable in advance and varies run to run.

        Agglomerative rather than HDBSCAN here (the inverse of the choice at
        the event level) because this pass needs a directly interpretable
        "merge while EVERY pair stays above X" rule over a small point set,
        not density estimation: with only tens of centroids, density is
        poorly defined and HDBSCAN would push most of them to noise. Groups
        smaller than TOPIC_MIN_CLUSTERS are dropped here, which is what
        leaves standalone events unassigned.

        linkage='complete', NOT 'average'. Average linkage merges on the
        *mean* pairwise similarity, which permits chaining: A resembles B and
        B resembles C, so A and C land in one topic while resembling each
        other hardly at all. Measured on this corpus, that let a topic hold a
        pair as dissimilar as 0.857 despite a 0.90 threshold — "Fitch affirms
        Commercial Bank" ended up beside electricity tariffs and housing
        rebuilds, chained through intermediate economic stories.

        Complete linkage requires every pair to clear the threshold, so the
        worst pair inside any topic is 0.90 by construction. It costs about
        5% coverage (365 clusters in topics rather than 384) and yields more,
        tighter topics. That trade is deliberate: topic_id=NULL is already a
        meaningful state meaning "no broader story", so an unassigned event
        is a missing answer, while an unrelated event inside a topic is a
        wrong one.
        """
        splitter = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.0 - TOPIC_SIMILARITY_THRESHOLD,
            metric='cosine',
            linkage='complete',
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

        # LLM first, and gated by its own flag rather than the event-label
        # one. A topic spans several events, so falling back to a member's
        # headline names the topic after one arbitrary part of itself — a
        # 10-event business topic ended up called "Intex - InMac Sri Lanka
        # 2026 opens at BMICH". The LLM is the only tier that reliably
        # produces an actual topic name.
        label = ''
        if TOPIC_LABEL_USE_LLM or CLUSTER_LABEL_USE_LLM:
            label = _llm_topic_label(titles)
        if not label:
            label = _keybert_label(titles)      # unavailable under a hosted provider
        if not label:
            label = min(titles, key=len)        # last resort; see above
        return _sanitize_label(label) or 'Untitled topic'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    result = ArticleClusterer().run()
    print(result)
