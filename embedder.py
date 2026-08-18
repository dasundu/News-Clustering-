# embedder.py
"""
Generates sentence embeddings for articles using sentence-transformers.
Stores normalized float32 vectors, tagged with the model that produced
them, as BYTEA in PostgreSQL.
"""

import hashlib
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from database import ArticleDatabase
from config import EMBEDDING_MODEL, EMBED_BATCH_SIZE, outlet_name_fragments
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
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def run(self, batch_size: int = EMBED_BATCH_SIZE) -> int:
        # Picks up articles with no embedding yet *and* articles whose
        # stored embedding_model no longer matches EMBEDDING_MODEL, so
        # changing the model in config.py safely re-embeds everything
        # instead of leaving stale, incomparable vectors mixed in silently.
        articles = self.db.get_articles_without_embedding(EMBEDDING_MODEL, limit=batch_size)
        if not articles:
            logger.info("No articles pending embedding")
            return 0

        logger.info(f"Embedding {len(articles)} articles...")

        texts = [self._build_text(a) for a in articles]
        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # L2-normalize for cosine-equivalent euclidean
            show_progress_bar=False,
        )

        relabeled = 0
        for article, text, vec in zip(articles, texts, vectors):
            vec32 = vec.astype('float32')
            blob = vec32.tobytes()
            source_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            # embedding_list mirrors the same vector into the pgvector column
            # /search reads — written in the same statement so the two
            # representations can never drift apart (see database.py).
            self.db.update_embedding(article['id'], blob, EMBEDDING_MODEL,
                                      source_hash, embedding_list=vec32.tolist())

            # Semantic second pass over the keyword-based category (see
            # classifier.py) — refines economics/politics/general, never
            # drops the article itself.
            old_cat = article.get('category') or 'general'
            new_cat = refine_category(self.model, vec, old_cat)
            if new_cat != old_cat:
                self.db.update_article_category(article['id'], new_cat)
                relabeled += 1

        logger.info(
            f"Embedded {len(articles)} articles (dim={vectors.shape[1]}); "
            f"classifier relabeled {relabeled}/{len(articles)}"
        )
        return len(articles)

    @staticmethod
    def _build_text(article: dict) -> str:
        title = _strip_outlet_names(article.get('title', ''))
        body = _strip_outlet_names((article.get('body_cleaned') or '')[:500])
        return f"{title}. {title}. {body}".strip()         # title repeated = more influence

    @staticmethod
    def deserialize(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype='float32')
