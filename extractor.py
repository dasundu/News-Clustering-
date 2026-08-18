# extractor.py
"""
Full article body extractor using trafilatura.
Visits each article URL and extracts clean text — no site-specific selectors needed.
"""

import hashlib
import logging
import threading
import trafilatura
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from database import ArticleDatabase
from scraper import _new_session
from config import BODY_BATCH_SIZE, EXTRACT_MAX_WORKERS, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_MIN_BODY_LEN = 80

# Caps concurrent requests to any single host, independent of
# EXTRACT_MAX_WORKERS. A batch of 100 articles routinely has 30+ from one
# outlet (e.g. daily_ft), and letting all EXTRACT_MAX_WORKERS threads hit
# that one host at once was observed making an already-slow site slower —
# several requests timed out that would have succeeded run sequentially,
# including some that exhausted every retry attempt. The concurrency this
# module adds should come from working on *different* hosts in parallel,
# not from hammering one host harder; this keeps that distinction real.
_MAX_REQUESTS_PER_HOST = 2
_host_semaphores: dict = {}
_host_semaphores_lock = threading.Lock()


def _host_semaphore(url: str) -> threading.Semaphore:
    host = urlparse(url).hostname or ''
    with _host_semaphores_lock:
        sem = _host_semaphores.get(host)
        if sem is None:
            sem = threading.Semaphore(_MAX_REQUESTS_PER_HOST)
            _host_semaphores[host] = sem
        return sem


class BodyExtractor:
    def __init__(self):
        self.db = ArticleDatabase()
        # A shared, retry-enabled session with a pool sized for
        # EXTRACT_MAX_WORKERS concurrent fetches (see scraper._new_session).
        # trafilatura.fetch_url() manages its own internal connection pool
        # instead, sized far too small for concurrent use — under
        # EXTRACT_MAX_WORKERS threads hitting the same host repeatedly, it
        # was observed constantly discarding and reopening connections
        # rather than reusing them. Fetching the HTML here and handing it to
        # trafilatura.extract() (which accepts raw HTML directly) keeps
        # trafilatura's extraction logic but replaces its transport.
        self.session = _new_session()

    def run(self, batch_size: int = BODY_BATCH_SIZE) -> int:
        articles = self.db.get_articles_without_body(limit=batch_size)
        if not articles:
            logger.info("No articles pending body extraction")
        else:
            logger.info(f"Fetching body for {len(articles)} articles "
                        f"({EXTRACT_MAX_WORKERS} concurrent)...")

            # Each fetch is an independent network call — trafilatura opens
            # its own connection per call rather than sharing a session, so
            # this parallelizes safely with no shared mutable state.
            results = []
            with ThreadPoolExecutor(max_workers=EXTRACT_MAX_WORKERS) as pool:
                futures = {pool.submit(self._fetch, a['url']): a for a in articles}
                for future in as_completed(futures):
                    article = futures[future]
                    body, image_url = future.result()
                    results.append((article, body, image_url))

            # Detect sidebar/widget content: same body >= 2 times from same source
            source_hash_counts: dict = {}
            for article, body, _ in results:
                if body and len(body) >= _MIN_BODY_LEN:
                    h = hashlib.md5(body.encode()).hexdigest()
                    src = article['source']
                    source_hash_counts.setdefault(src, {})
                    source_hash_counts[src][h] = source_hash_counts[src].get(h, 0) + 1

            updated = 0
            discarded = 0
            for article, body, image_url in results:
                if body and len(body) >= _MIN_BODY_LEN:
                    h = hashlib.md5(body.encode()).hexdigest()
                    if source_hash_counts.get(article['source'], {}).get(h, 0) >= 2:
                        logger.debug(f"  Discarded duplicate sidebar body: {article['url']}")
                        body = ''
                        discarded += 1

                self.db.update_body(article['id'], body, image_url)
                if body:
                    updated += 1

            if discarded:
                logger.warning(f"  Discarded {discarded} duplicate bodies (sidebar/widget content)")
            logger.info(f"Body extraction done: {updated}/{len(articles)} successful")

        return len(articles)

    def _fetch(self, url: str):
        """Returns (body_text, image_url). Both may be empty strings."""
        try:
            with _host_semaphore(url):
                r = self.session.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                html = r.text
            if not html:
                return '', ''

            text = trafilatura.extract(
                html,
                favor_recall=True,
                include_comments=False,
                include_tables=False,
            ) or ''

            image_url = _extract_og_image(html)
            return text, image_url
        except Exception as e:
            logger.warning(f"Body fetch failed for {url}: {e}")
            return '', ''


_IMG_PATH_HINTS = ('/uploads/', '/images/', '/media/', '/wp-content/', '/photo', '/news/', '/files/')


def _extract_og_image(html: str) -> str:
    """Extract image URL from HTML: og:image → twitter:image → first likely news photo."""
    try:
        soup = BeautifulSoup(html, 'html.parser')

        # 1. Standard meta tags
        for prop in ('og:image', 'twitter:image'):
            tag = soup.find('meta', attrs={'property': prop}) or \
                  soup.find('meta', attrs={'name': prop})
            if tag and tag.get('content', '').startswith('http'):
                return tag['content'].strip()

        # 2. Fallback: first <img> whose src looks like a news photo (not icon/tracker)
        for img in soup.find_all('img', src=True):
            src = img.get('src', '')
            if src.startswith('http') and any(hint in src for hint in _IMG_PATH_HINTS):
                return src
    except Exception:
        pass
    return ''
