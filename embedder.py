# embedder.py
"""
Generates sentence embeddings for articles.

Which backend actually produces the vectors (a local sentence-transformer
or Google's hosted API) is decided by embedding_provider.py — this module
only cares that it gets back L2-normalized float32 vectors. Each stored
vector is tagged with the provider+model that produced it, so switching
providers re-embeds the corpus instead of silently mixing two incompatible
vector spaces.
"""

import hashlib
import logging
import re
import numpy as np
import embedding_provider
from database import ArticleDatabase
from config import EMBED_BATCH_SIZE, EMBED_COMMIT_CHUNK, outlet_name_fragments
from classifier import refine_category

logger = logging.getLogger(__name__)

_OUTLET_STRIP_RE = None  # built lazily from outlet_name_fragments()


def _strip_outlet_names(text: str) -> str:
    """
    Remove outlet display names (e.g. 'Daily FT', 'EconomyNext') from text
    before it's embedded, so the embedding reflects the article's topic
    rather than partly reflecting which outlet wrote it — an outlet's own
    name shows up in a large fraction of its own articles (bylines,
    "reported by", self-references), which would otherwise give the model a
    shortcut correlated with publisher rather than content.
    """
    global _OUTLET_STRIP_RE
    if _OUTLET_STRIP_RE is None:
        names = sorted(outlet_name_fragments(), key=len, reverse=True)
        pattern = '|'.join(re.escape(n) for n in names if n)
        _OUTLET_STRIP_RE = re.compile(pattern, re.IGNORECASE) if pattern else None
    if _OUTLET_STRIP_RE is None:
        return text
    return _OUTLET_STRIP_RE.sub(' ', text)


class ArticleEmbedder:
    def __init__(self):
        self.db = ArticleDatabase()
        # No model is loaded here any more — embedding_provider decides
        # whether that even happens, and defers it until first use.
        self.model_id = embedding_provider.model_id()
        logger.info(f"Embedding provider: {self.model_id}")

    def run(self, batch_size: int = EMBED_BATCH_SIZE) -> int:
        # Picks up articles with no embedding yet *and* articles whose
        # stored embedding_model no longer matches EMBEDDING_MODEL, so
        # changing the model in config.py safely re-embeds everything
        # instead of leaving stale, incomparable vectors mixed in silently.
        articles = self.db.get_articles_without_embedding(self.model_id, limit=batch_size)
        if not articles:
            logger.info("No articles pending embedding")
            return 0

        logger.info(f"Embedding {len(articles)} articles...")

        # Processed and committed in sub-batches rather than one shot. With a
        # hosted provider a rate limit can cut a run short at any point; doing
        # the work incrementally means whatever succeeded is already persisted
        # and the next run resumes from there, instead of throwing away the
        # entire batch. Costs nothing under the local provider.
        done = relabeled = 0
        dim = None
        for start in range(0, len(articles), EMBED_COMMIT_CHUNK):
            chunk = articles[start:start + EMBED_COMMIT_CHUNK]
            texts = [self._build_text(a) for a in chunk]
            try:
                vectors = embedding_provider.encode(texts, task='document')
            except Exception as e:
                logger.warning(
                    f"Embedding stopped after {done}/{len(articles)} articles: {e}"
                )
                break
            dim = vectors.shape[1]

            embedded, recategorised = [], []
            for article, text, vec in zip(chunk, texts, vectors):
                vec32 = vec.astype('float32')
                source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
                # The vector goes to both the BYTEA column and its pgvector
                # mirror that /search reads — the same statement per row, so
                # the two representations can never drift apart.
                embedded.append((vec32.tobytes(), self.model_id, source_hash,
                                 vec32.tolist(), article['id']))

                # Semantic second pass over the keyword-based category (see
                # classifier.py) — refines economics/politics/general, never
                # drops the article itself.
                old_cat = article.get('category') or 'general'
                new_cat = refine_category(vec, old_cat)
                if new_cat != old_cat:
                    recategorised.append((new_cat, article['id']))

            # One batch per chunk rather than a round trip per article. The
            # chunk is still the commit unit, so a rate limit that stops the
            # run keeps every chunk already written.
            self.db.update_embeddings(embedded)
            self.db.update_article_categories(recategorised)
            done += len(embedded)
            relabeled += len(recategorised)

        logger.info(
            f"Embedded {done}/{len(articles)} articles"
            + (f" (dim={dim})" if dim else "")
            + f"; classifier relabeled {relabeled}/{max(done,1)}"
        )
        return done

    @staticmethod
    def _build_text(article: dict) -> str:
        title = _strip_outlet_names(article.get('title', ''))
        body = _strip_outlet_names((article.get('body_cleaned') or '')[:500])
        return f"{title}. {title}. {body}".strip()         # title repeated = more influence

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype='float32')
