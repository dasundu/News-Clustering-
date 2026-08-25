# embedding_provider.py
"""
One seam between the pipeline and whatever produces embeddings.

Everything downstream — clustering, topic grouping, the category classifier,
claim search — only ever needs "turn these strings into unit vectors". This
module is the single place that knows *how*, so switching between a local
sentence-transformer and Google's hosted API is a config change rather than
an edit to five files.

Providers
---------
local   The original `paraphrase-multilingual-MiniLM-L12-v2` running in
        process. Fully offline and free, but drags in torch +
        sentence-transformers (~1.2GB) and a 458MB model download, which is
        what makes this project awkward to deploy: the scraper needs a real
        VM, and /search cannot run on a lightweight API host at all.

gemini  Google's `gemini-embedding-001`. No local model, no torch — the
        deployment problem disappears and /search works anywhere. Costs
        about $1.64/year at this project's measured rate (~222 articles a
        day), and scores materially higher on MTEB.

CRITICAL: the two produce *incompatible* vector spaces. Cosine similarity
between models is not comparable, and neither are the thresholds tuned
against them — measured on real data, unrelated text scores ~0.2 under the
local model but ~0.70 under Gemini. Every threshold in config.py must be
re-derived when switching (see scripts/compare_embeddings.py). The
`embedding_model` column records which provider produced each stored vector
so a switch triggers an automatic re-embed rather than silently mixing them.
"""

import logging
import os
import threading
import time
from typing import List

import numpy as np

from config import (
    EMBEDDING_PROVIDER, LOCAL_EMBEDDING_MODEL, GEMINI_EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, GEMINI_EMBED_BATCH, GEMINI_EMBED_SLEEP, REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

# How this module's abstract `task` maps onto Gemini's taskType parameter.
#
# Both document and query deliberately resolve to SEMANTIC_SIMILARITY rather
# than the RETRIEVAL_* pair. Gemini's task types produce *different* embedding
# spaces, and one stored vector has to serve both clustering and search here —
# so a single symmetric space is the only coherent choice without storing two
# vectors per article. It is also the closest analogue to the paraphrase model
# this replaces, which keeps the threshold shift as small as possible.
_GEMINI_TASK = {
    'document': 'SEMANTIC_SIMILARITY',
    'query': 'SEMANTIC_SIMILARITY',
    'classification': 'CLASSIFICATION',
}


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Everything downstream assumes unit vectors — it is what lets the codebase
    treat cosine similarity, dot product and euclidean distance as
    interchangeable. Enforced here, once, for every provider.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-9)).astype('float32')


# ── local provider ────────────────────────────────────────────────────────

_local_model = None
_local_lock = threading.Lock()


def _get_local_model():
    """
    Loaded once per process, not once per call. Lock-guarded because uvicorn
    serves from a thread pool, so two concurrent first-requests could
    otherwise both start a ~458MB model load.

    The dependencies are NOT installed by default — the project ships
    Gemini-only. This path exists for offline work and for deterministic
    evaluation runs, and needs an explicit opt-in install.
    """
    global _local_model
    if _local_model is None:
        with _local_lock:
            if _local_model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise RuntimeError(
                        "EMBEDDING_PROVIDER=local needs sentence-transformers, which is "
                        "not installed (this project runs on the Gemini API by default). "
                        "Install it with:  pip install sentence-transformers keybert . "
                        "Note that switching provider re-embeds the whole corpus and "
                        "invalidates every similarity threshold in config.py — run "
                        "compare_embeddings.py afterwards."
                    ) from e
                logger.info(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
                _local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _local_model


def _encode_local(texts: List[str]) -> np.ndarray:
    vecs = _get_local_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype='float32')


# ── gemini provider ───────────────────────────────────────────────────────

_gemini_session = None
_gemini_lock = threading.Lock()


def _get_gemini_session():
    """
    A requests.Session with retry-and-backoff mounted at the transport layer,
    same approach as scraper._new_session but with API headers rather than
    browser ones. 429 is included in the retry list: rate limiting is the
    expected failure mode here, and a whole pipeline run should not die
    because one batch was throttled.
    """
    global _gemini_session
    if _gemini_session is None:
        with _gemini_lock:
            if _gemini_session is None:
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry

                session = requests.Session()
                retry = Retry(
                    total=5,
                    backoff_factor=2.0,          # 2s, 4s, 8s, 16s, 32s
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=('POST',),
                    respect_retry_after_header=True,
                    raise_on_status=False,
                )
                session.mount('https://', HTTPAdapter(max_retries=retry))
                _gemini_session = session
    return _gemini_session


def _encode_gemini(texts: List[str], task: str) -> np.ndarray:
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=gemini but GEMINI_API_KEY is not set. "
            "Add it to backend/.env, or set EMBEDDING_PROVIDER=local."
        )

    task_type = _GEMINI_TASK.get(task, _GEMINI_TASK['document'])
    url = (f'https://generativelanguage.googleapis.com/v1beta/models/'
           f'{GEMINI_EMBEDDING_MODEL}:batchEmbedContents')
    session = _get_gemini_session()

    out: List[List[float]] = []
    # Chunked because the batch endpoint caps how many items one call may
    # carry; the pipeline routinely hands this 256 texts at a time.
    #
    # Quota is charged per *embedded item*, not per HTTP request — sending
    # 400 texts in four calls still burns 400 units of a per-minute
    # allowance. Observed in practice: back-to-back batches of 400 trip a
    # 429 that transport-level backoff (~62s of retries) cannot ride out.
    # Hence the deliberate pause between chunks.
    chunks = list(range(0, len(texts), GEMINI_EMBED_BATCH))
    for idx, start in enumerate(chunks):
        if idx and GEMINI_EMBED_SLEEP:
            time.sleep(GEMINI_EMBED_SLEEP)
        chunk = texts[start:start + GEMINI_EMBED_BATCH]
        payload = {
            'requests': [
                {
                    'model': f'models/{GEMINI_EMBEDDING_MODEL}',
                    'content': {'parts': [{'text': t or ' '}]},
                    'taskType': task_type,
                    'outputDimensionality': EMBEDDING_DIMENSION,
                }
                for t in chunk
            ]
        }
        response = session.post(
            url,
            headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
            json=payload,
            timeout=max(REQUEST_TIMEOUT, 60),
        )
        if response.status_code == 429:
            raise RuntimeError(
                "Gemini rate limit exceeded and retries were exhausted. Quota is "
                "charged per embedded item, so a large backfill can outrun a "
                "free-tier per-minute allowance. Lower GEMINI_EMBED_BATCH, raise "
                "GEMINI_EMBED_SLEEP, or enable billing. "
                f"Detail: {response.text[:200]}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Gemini embedding failed: HTTP {response.status_code} — "
                f"{response.text[:300]}"
            )
        embeddings = response.json().get('embeddings', [])
        if len(embeddings) != len(chunk):
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} embeddings for {len(chunk)} inputs"
            )
        out.extend(e['values'] for e in embeddings)

    # Gemini returns *unnormalized* vectors at any dimension below 3072
    # (measured: norms around 0.59 at 768), so normalizing here is required,
    # not defensive.
    return _l2_normalize(np.asarray(out, dtype='float32'))


# ── public interface ──────────────────────────────────────────────────────

def encode(texts: List[str], task: str = 'document') -> np.ndarray:
    """
    Embed `texts` into L2-normalized float32 vectors, shape (len(texts), dim).

    `task` is a hint the provider may use to pick a better representation:
      'document'       text being indexed (articles)
      'query'          a search input (a claim)
      'classification' exemplar sentences for the category classifier
    The local provider ignores it — it has no such notion.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIMENSION), dtype='float32')
    if EMBEDDING_PROVIDER == 'gemini':
        return _encode_gemini(texts, task)
    return _encode_local(texts)


def dimension() -> int:
    return EMBEDDING_DIMENSION


def model_id() -> str:
    """
    Recorded in articles.embedding_model. Provider-qualified so the two
    backends can never be mistaken for each other — the mismatch is what
    triggers automatic re-embedding when the provider changes (see
    database.get_articles_without_embedding).
    """
    if EMBEDDING_PROVIDER == 'gemini':
        return f'gemini:{GEMINI_EMBEDDING_MODEL}:{EMBEDDING_DIMENSION}'
    return f'local:{LOCAL_EMBEDDING_MODEL}'


def is_local() -> bool:
    """True when a real model is resident in this process. KeyBERT needs one,
    so cluster labelling checks this before reaching for that tier."""
    return EMBEDDING_PROVIDER != 'gemini'
