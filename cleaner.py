# cleaner.py
"""
Cleans raw article text before embedding.
Removes HTML leftovers, normalizes whitespace, filters junk.
"""

import re
import html as _html
import logging
from database import ArticleDatabase

logger = logging.getLogger(__name__)


def fix_encoding(text: str) -> str:
    """Fix UTF-8 mojibake caused by Latin-1 mis-decoding (e.g. â€™ → ')."""
    if not text:
        return text
    try:
        fixed = text.encode('latin-1').decode('utf-8')
        return _html.unescape(fixed)
    except (UnicodeEncodeError, UnicodeDecodeError):
        return _html.unescape(text)


class TextCleaner:
    def __init__(self):
        self.db = ArticleDatabase()

    def run(self) -> int:
        articles = self.db.get_articles_to_clean()
        if not articles:
            logger.info("No articles to clean")
            return 0

        logger.info(f"Cleaning {len(articles)} articles...")
        bodies, titles = [], []
        for article in articles:
            # Fix encoding on both title and body
            fixed_title = fix_encoding(article.get('title', ''))
            bodies.append((self._clean(fix_encoding(article['body'])), article['id']))
            if fixed_title != article.get('title', ''):
                titles.append((fixed_title, article['id']))

        # Written as batches, not a round trip per article (see the batched
        # writers in database.py). Cleaning is deterministic, so a failed
        # batch is simply redone by the next run.
        self.db.update_cleaned_bodies(bodies)
        self.db.update_article_titles(titles)

        logger.info(f"Cleaned {len(bodies)} articles")
        return len(bodies)

    def _clean(self, text: str) -> str:
        if not text:
            return ''
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
