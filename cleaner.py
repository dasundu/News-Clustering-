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
        count = 0
        for article in articles:
            # Fix encoding on both title and body
            fixed_title = fix_encoding(article.get('title', ''))
            cleaned_body = self._clean(fix_encoding(article['body']))
            self.db.update_cleaned_body(article['id'], cleaned_body)
            if fixed_title != article.get('title', ''):
                self.db.update_article_title(article['id'], fixed_title)
            count += 1

        logger.info(f"Cleaned {count} articles")
        return count

    def _clean(self, text: str) -> str:
        if not text:
            return ''
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
