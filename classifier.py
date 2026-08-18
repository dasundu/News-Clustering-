# classifier.py
"""
Embedding-similarity classification: a second, semantic pass on top of the
keyword filter in config.py.

Why not keyword-only
---------------------
config.classify_category() is a substring match against a fixed word list.
It has two failure modes that matter for a thesis-grade dataset:
  * No tolerance for paraphrase — "cabinet reshuffle" or "GDP contracts"
    mean nothing to it unless those exact words are on the list.
  * No way to weigh a match — an article that mentions "government" once in
    passing scores identically to one that is centrally about government
    policy, because the check is boolean, not a measure of how *much* the
    text is about that topic.

Why not embedding-only
------------------------
Going the other way — dropping the keyword filter and classifying purely by
embedding similarity — would mean embedding (and therefore body-fetching)
every article from every source before knowing whether it's wanted, which is
both slow and impolite to scrape ~20 sites for. The keyword filter stays as
the cheap recall-oriented gate at scrape time (see config.py's widened
POLITICS_KEYWORDS/ECONOMICS_KEYWORDS); this module only refines the category
label afterwards, once an embedding already exists — it never drops a row
that already passed the keyword gate, only relabels it.

Method
------
A handful of representative sentences are hand-written for each of three
classes (economics, politics, general/other) and embedded once with the same
sentence-transformer model already used for article vectors. An article's
existing embedding is then compared to each class's centroid by cosine
similarity (embeddings are L2-normalized, so this is a dot product); the
best match relabels the article only if it beats the current keyword-based
label's implied class by a margin — a plain argmax would flip a lot of
genuinely-borderline articles back and forth for no real gain.

This is a legitimate, standard technique (embeddings-as-zero-shot-classifier)
and, importantly, one you can evaluate: hold out a hand-labeled sample and
report how often the keyword label and the embedding label agree, and which
one matches your manual label more often. That comparison is thesis-worthy
in itself.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Kept short and deliberately varied in phrasing (not just keyword-list
# sentences) so the resulting centroid captures the *topic*, not a bag of
# trigger words repeated back at itself.
_EXEMPLARS = {
    'economics': [
        "Sri Lanka's economy grows as inflation eases and the rupee strengthens",
        "Central Bank cuts interest rates to support economic growth",
        "Government presents new budget with tax reforms and revenue targets",
        "Colombo Stock Exchange rises as foreign investment increases",
        "IMF approves new loan tranche for Sri Lanka's recovery programme",
        "Trade deficit narrows as exports rise and imports slow",
        "Sri Lanka's foreign reserves increase amid economic recovery",
        "Central bank governor comments on monetary policy and interest rates",
        "Company reports quarterly profits and announces new investment",
        "Fuel prices revised as global oil markets shift",
        "Company files for bankruptcy and liquidation amid mounting debt",
        "Court approves a company's debt restructuring and creditor repayment plan",
    ],
    'politics': [
        "President addresses parliament on new policy reforms",
        "Opposition leader criticizes government's handling of the crisis",
        "Parliament passes new constitutional amendment bill",
        "Cabinet reshuffle announced amid political tension",
        "Election commission announces schedule for the next general election",
        "Court orders investigation into corruption allegations against a minister",
        "Minister resigns following public protest over policy",
        "Government and opposition clash over new legislation",
        "Diplomatic talks held between Sri Lanka and a foreign delegation",
        "Police and security forces respond to unrest at a state institution",
    ],
    'general': [
        "Heavy rains and strong winds expected across the country this week",
        "Local football team wins championship final",
        "New film release breaks box office records",
        "University students win international science competition",
        "Wildlife officials rescue an elephant from a residential area",
        "Popular actor announces new film project",
        "Health officials warn of a seasonal flu outbreak",
        "Traffic congestion reported on the main highway after an accident",
        "School holds annual sports meet with record participation",
        "New restaurant opens offering a variety of local cuisine",
    ],
}

# An embedding-label swap only overrides the keyword-based label when it
# wins by at least this much — a plain argmax over three centroids flips a
# lot of genuinely-close calls back and forth run to run for no real benefit.
_MARGIN = 0.03

_centroids = None  # lazily built: {'economics': ndarray, 'politics': ndarray, 'general': ndarray}


def _build_centroids(model):
    global _centroids
    if _centroids is not None:
        return _centroids
    _centroids = {}
    for label, sentences in _EXEMPLARS.items():
        vecs = model.encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        centroid = np.mean(vecs, axis=0)
        _centroids[label] = centroid / (np.linalg.norm(centroid) + 1e-9)
    logger.info(f"Built classifier exemplar centroids for {list(_centroids)}")
    return _centroids


def refine_category(model, embedding: np.ndarray, keyword_category: str) -> str:
    """
    Return the category to store for an article: `keyword_category` unless
    the embedding is clearly closer to a different class's exemplar centroid.
    `keyword_category` should be 'economics' or 'politics' (articles that
    never matched a keyword aren't saved at all, for filter_keywords=True
    sources) — 'general' is only ever an output here, marking articles kept
    unconditionally by a filter_keywords=False source whose actual content
    turned out not to be economics/politics after all.
    """
    centroids = _build_centroids(model)
    sims = {label: float(np.dot(embedding, c)) for label, c in centroids.items()}
    best_label = max(sims, key=sims.get)

    current_sim = sims.get(keyword_category, -1.0)
    if sims[best_label] - current_sim >= _MARGIN:
        return best_label
    return keyword_category
