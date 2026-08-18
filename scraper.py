# scraper.py
"""
News scraper for Sri Lankan news outlets.
Discovery priority per outlet: WP REST API → RSS → HTML scraping
"""

import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import xml.etree.ElementTree as ET
from html import unescape
from bs4 import BeautifulSoup
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

from config import NEWS_SOURCES, REQUEST_TIMEOUT, USER_AGENT, classify_category, SCRAPE_MAX_WORKERS
from database import ArticleDatabase

logger = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.google.com/',
}


# Hiru's listing pages read from this endpoint, 12 stories per page.
_HIRU_API = 'https://hirunews.lk/api/fetch_en_news.php'
_HIRU_IMAGE_BASE = 'https://cdn.hirunews.lk/Data/News_Images/'

# Categories carrying Sri Lankan coverage. 'International' is left out: it is
# foreign wire copy, and a sample of 48 of its stories contained no Sri Lankan
# ones. Sports and Entertainment are outside this project's scope.
_HIRU_CATEGORIES = ('General', 'Business')
_HIRU_PAGES = 3


def _new_session() -> requests.Session:
    """
    A requests.Session with automatic retry-with-backoff mounted at the
    transport layer, so every request made through it — from any of the
    ~20 per-outlet scraping methods below, with no changes needed to any of
    them — gets the same retry policy. Retries transient failures (5xx,
    429, connection resets); does *not* retry 403, since a source blocking
    the scraper outright won't start succeeding on attempt 2 (matches the
    sibling project's same distinction between transient and permanent
    failures). pool_maxsize is raised to comfortably cover concurrent
    scraping across SCRAPE_MAX_WORKERS threads sharing one session.
    """
    session = requests.Session()
    session.headers.update(_BROWSER_HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=('GET', 'POST'),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=max(20, SCRAPE_MAX_WORKERS * 2))
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class NewsScraper:

    def __init__(self):
        self.db = ArticleDatabase()
        self.session = _new_session()

    # ── Public API ─────────────────────────────────────────────────────────

    def scrape_all_sources(self) -> Dict:
        """
        Scrapes every configured source concurrently — discovery is
        I/O-bound (waiting on each outlet's own server), so running sources
        in parallel is a straightforward wall-clock win with no change
        needed inside any individual outlet's scraping method.
        """
        results = {'total_found': 0, 'total_new': 0, 'by_source': {}}

        def _scrape_one(source_name: str, config: Dict):
            try:
                logger.info(f"Scraping {source_name}...")
                articles = self._scrape_source(source_name, config)

                if config.get('filter_keywords', False):
                    filtered = []
                    for a in articles:
                        cat = classify_category(a['title'])
                        if cat:
                            a['category'] = cat
                            filtered.append(a)
                    articles = filtered

                new_count = self.db.save_batch(articles) if articles else 0
                logger.info(f"  {source_name}: {len(articles)} found · {new_count} new")
                return source_name, len(articles), new_count
            except Exception as e:
                logger.error(f"Error scraping {source_name}: {e}")
                return source_name, 0, 0

        with ThreadPoolExecutor(max_workers=SCRAPE_MAX_WORKERS) as pool:
            futures = [pool.submit(_scrape_one, name, cfg) for name, cfg in NEWS_SOURCES.items()]
            for future in as_completed(futures):
                source_name, found, new_count = future.result()
                results['total_found'] += found
                results['total_new'] += new_count
                results['by_source'][source_name] = {'found': found, 'new': new_count}

        return results

    def scrape_custom_sources(self) -> Dict:
        results = {'total_found': 0, 'total_new': 0, 'by_source': {}}
        custom_sources = self.db.get_custom_sources()

        def _scrape_one(src: Dict):
            source_name = src['name']
            config = {'url': src['url'], 'type': 'html', 'category': 'general'}
            try:
                logger.info(f"Scraping custom source: {source_name}...")
                articles = self._try_rss_paths(source_name, config) or \
                           self._scrape_html_generic(source_name, config)
                filtered = []
                for a in articles:
                    cat = classify_category(a['title'])
                    if cat:
                        a['category'] = cat
                        filtered.append(a)
                new_count = self.db.save_batch(filtered) if filtered else 0
                self.db.touch_custom_source(src['id'])
                return source_name, len(filtered), new_count
            except Exception as e:
                logger.error(f"Error scraping custom source {source_name}: {e}")
                return source_name, 0, 0

        with ThreadPoolExecutor(max_workers=SCRAPE_MAX_WORKERS) as pool:
            futures = [pool.submit(_scrape_one, src) for src in custom_sources]
            for future in as_completed(futures):
                source_name, found, new_count = future.result()
                results['total_found'] += found
                results['total_new'] += new_count
                results['by_source'][source_name] = {'found': found, 'new': new_count}

        return results

    # ── Router ─────────────────────────────────────────────────────────────

    def _scrape_source(self, source_name: str, config: Dict) -> List[Dict]:
        scraper_map = {
            'economynext':       self._scrape_economynext,
            'lankabusinessonline': self._scrape_lbo,
            'daily_ft':          self._scrape_ft,
            'business_today':    self._scrape_businesstoday,
            'adaderana':         self._scrape_adaderana,
            'newsfirst':         self._scrape_newsfirst,
            'daily_mirror':      self._scrape_dailymirror,
            'the_morning':       self._scrape_themorning,
            'daily_news':        self._scrape_dailynews,
            'the_island':        self._scrape_theisland,
            'sunday_observer':   self._scrape_sundayobserver,
            'colombo_gazette':   self._scrape_colombogazette,
            'ceylon_today':      self._scrape_ceylontoday,
            'sunday_times':      self._scrape_sundaytimes,
            'news_lk':           self._scrape_newslk,
            'helakuru':          self._scrape_helakuru,
            'hirunews':          self._scrape_hirunews,
            'onlanka':           self._scrape_onlanka,
            'asianmirror':       self._scrape_asianmirror,
            'lankanewsweb':      self._scrape_lankanewsweb,
            'lankabusinessnews': self._scrape_lankabusinessnews,
        }
        fn = scraper_map.get(source_name)
        if fn:
            return fn(source_name, config)
        return self._scrape_html_generic(source_name, config)

    # ── Shared discovery helpers ───────────────────────────────────────────

    def _try_wp_api(self, source_name: str, config: Dict,
                    base_url: str = None, max_articles: int = 50) -> List[Dict]:
        """WordPress REST API discovery. Returns articles with real dates and body text."""
        base = (base_url or config['url']).rstrip('/')
        articles = []
        seen = set()
        try:
            for page in range(1, 6):
                url = (
                    f"{base}/wp-json/wp/v2/posts"
                    f"?per_page=100&page={page}"
                    f"&_fields=id,date,title,link,content"
                    f"&orderby=date&order=desc"
                )
                r = self.session.get(url, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    break
                posts = r.json()
                if not isinstance(posts, list) or not posts:
                    break
                for post in posts:
                    link = (post.get('link') or '').strip().rstrip('/')
                    if not link or link in seen or self.db.url_exists(link):
                        continue
                    seen.add(link)
                    title_obj = post.get('title', {})
                    raw_title = title_obj.get('rendered', '') if isinstance(title_obj, dict) else str(title_obj)
                    title = BeautifulSoup(raw_title, 'html.parser').get_text(strip=True)
                    if not title or len(title) < 8:
                        continue
                    content_obj = post.get('content', {})
                    body = ''
                    if isinstance(content_obj, dict):
                        raw = content_obj.get('rendered', '')
                        if raw and len(raw) > 80 and 'read more' not in raw.lower()[-50:]:
                            body = BeautifulSoup(raw, 'html.parser').get_text(separator=' ', strip=True)
                    pub = post.get('date', '') or datetime.now().isoformat()
                    articles.append(self._make_article(title, body, link, source_name, config, pub))
                    if len(articles) >= max_articles:
                        return articles
                if len(posts) < 100:
                    break
        except Exception as e:
            logger.debug(f"WP API failed for {source_name}: {e}")
        return articles

    def _try_rss_paths(self, source_name: str, config: Dict,
                       base_url: str = None, paths: List[str] = None) -> List[Dict]:
        """Try RSS feed paths in order, return articles from the first that works."""
        base = (base_url or config['url']).rstrip('/')
        paths = paths or ['/feed', '/feed/', '/rss', '/feed/rss2', '/?feed=rss2', '/rss.xml']
        for path in paths:
            try:
                r = self.session.get(base + path, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200 or len(r.content) < 200:
                    continue
                root = ET.fromstring(r.content)
                items = root.findall('.//item')
                if not items:
                    continue
                articles = []
                seen = set()
                for item in items[:50]:
                    link = (item.findtext('link') or '').strip()
                    if not link or link in seen or self.db.url_exists(link):
                        continue
                    seen.add(link)
                    title = (item.findtext('title') or '').strip()
                    if not title or len(title) < 8:
                        continue
                    pub = self._parse_date(item.findtext('pubDate') or '')
                    articles.append(self._make_article(title, '', link, source_name, config, pub))
                if articles:
                    logger.debug(f"  RSS {path} returned {len(articles)} articles for {source_name}")
                    return articles
            except Exception as e:
                logger.debug(f"  RSS {path} failed for {source_name}: {e}")
        return []

    def _scrape_html_pattern(self, source_name: str, config: Dict,
                             url: str, pattern: re.Pattern,
                             base_prefix: str = '') -> List[Dict]:
        """Generic HTML scraper using an href regex pattern."""
        try:
            r = self.session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            articles = []
            seen = set()
            for a in soup.find_all('a', href=pattern):
                href = a['href']
                article_url = href if href.startswith('http') else base_prefix + href
                if article_url in seen or self.db.url_exists(article_url):
                    continue
                seen.add(article_url)
                h = a.find(['h1', 'h2', 'h3', 'h4'])
                title = h.get_text(strip=True) if h else a.get_text(strip=True)
                title = re.sub(r'\s+', ' ', title).strip()
                if not title or len(title) < 10:
                    continue
                articles.append(self._make_article(title, '', article_url, source_name, config))
                if len(articles) >= 50:
                    break
            return articles
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
            return []

    # ── Per-outlet scrapers ────────────────────────────────────────────────

    def _scrape_economynext(self, source_name: str, config: Dict) -> List[Dict]:
        """WP REST API → RSS → HTML"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config, paths=['/feed', '/rss', '/feed/rss2'])
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'economynext\.com/[a-z0-9][a-z0-9\-]{10,}'),
            )
        return articles

    def _scrape_lbo(self, source_name: str, config: Dict) -> List[Dict]:
        """Lanka Business Online — WP REST API → HTML"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._scrape_html_generic(source_name, config)
        return articles

    def _scrape_ft(self, source_name: str, config: Dict) -> List[Dict]:
        """ft.lk — h3.newschs links"""
        try:
            r = self.session.get(config['url'], timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            articles = []
            seen = set()
            for h3 in soup.find_all('h3', class_='newschs'):
                parent_a = h3.parent if h3.parent.name == 'a' else h3.find_parent('a')
                if not parent_a:
                    continue
                url = parent_a.get('href', '')
                if not url or url in seen or self.db.url_exists(url):
                    continue
                seen.add(url)
                title = h3.text.strip()
                if title:
                    articles.append(self._make_article(title, '', url, source_name, config))
                if len(articles) >= 50:
                    break
            return articles
        except Exception as e:
            logger.error(f"Error scraping ft.lk: {e}")
            return []

    def _scrape_businesstoday(self, source_name: str, config: Dict) -> List[Dict]:
        """Business Today — WP REST API → HTML"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'businesstoday\.lk/[a-z0-9\-]{15,}'),
                'https://www.businesstoday.lk',
            )
        return articles

    def _scrape_adaderana(self, source_name: str, config: Dict) -> List[Dict]:
        """Ada Derana — RSS → HTML (div.news-story)"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=['/rss.php', '/rss', '/feed', '/hot-news-rss.php']
        )
        if not articles:
            try:
                r = self.session.get(config['url'], timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')
                articles = []
                seen = set()
                for div in soup.find_all('div', class_='news-story'):
                    h3 = div.find('h3')
                    if not h3:
                        continue
                    a = h3.find('a', href=True)
                    if not a:
                        continue
                    url = a['href']
                    if not url.startswith('http'):
                        url = config['url'].rstrip('/') + '/' + url.lstrip('/')
                    if url in seen or self.db.url_exists(url):
                        continue
                    seen.add(url)
                    title = h3.text.strip()
                    if title:
                        articles.append(self._make_article(title, '', url, source_name, config))
                    if len(articles) >= 50:
                        break
            except Exception as e:
                logger.error(f"Error scraping adaderana: {e}")
        return articles

    def _scrape_newsfirst(self, source_name: str, config: Dict) -> List[Dict]:
        """Newsfirst — RSS → date-URL pattern"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=['/feed/rss2', '/feed', '/rss', '/?feed=rss2']
        )
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'^/\d{4}/\d{2}/\d{2}/'),
                config['url'].rstrip('/'),
            )
            for a in articles:
                a['title'] = re.sub(r'\d{2}-\d{2}-\d{4}.*', '', a['title']).strip()
        return articles

    def _scrape_dailymirror(self, source_name: str, config: Dict) -> List[Dict]:
        """Daily Mirror — RSS (8 category feeds) → HTML"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=[
                '/rss.xml', '/rss/breaking-news', '/rss/business',
                '/rss/politics', '/rss/local', '/rss/world', '/feed', '/rss',
            ]
        )
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'/\d+-\d+$'),
                config['url'].rstrip('/'),
            )
        return [a for a in articles if len(a['title']) >= 15]

    def _scrape_themorning(self, source_name: str, config: Dict) -> List[Dict]:
        """The Morning — WP REST API → RSS → __NEXT_DATA__ JSON"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        if not articles:
            import json as _json
            try:
                r = self.session.get(config['url'], timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')
                tag = soup.find('script', id='__NEXT_DATA__')
                if tag:
                    data = _json.loads(tag.string)
                    latest = data.get('props', {}).get('pageProps', {}).get('latestNews', [])
                    seen = set()
                    for item in latest:
                        art_id = item.get('id', '')
                        title = (item.get('title') or '').strip()
                        if not title or len(title) < 10 or not art_id:
                            continue
                        url = f'https://www.themorning.lk/articles/{art_id}'
                        if url in seen or self.db.url_exists(url):
                            continue
                        seen.add(url)
                        articles.append(self._make_article(title, '', url, source_name, config))
                        if len(articles) >= 50:
                            break
            except Exception as e:
                logger.error(f"Error scraping the_morning: {e}")
        return articles

    def _scrape_dailynews(self, source_name: str, config: Dict) -> List[Dict]:
        """Daily News — RSS → HTML"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=['/feed/', '/feed', '/rss.xml', '/rss']
        )
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'dailynews\.lk/\d{4}/\d{2}/\d{2}/'),
                'https://www.dailynews.lk',
            )
        return articles

    def _scrape_theisland(self, source_name: str, config: Dict) -> List[Dict]:
        """The Island — WP REST API → RSS → HTML (must use island.lk not www)"""
        base = 'https://island.lk'
        articles = self._try_wp_api(source_name, config, base_url=base)
        if not articles:
            articles = self._try_rss_paths(source_name, config, base_url=base,
                                           paths=['/feed/', '/feed', '/rss', '/rss.xml'])
        if not articles:
            try:
                r = self.session.get(base, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, 'html.parser')
                articles = []
                seen = set()
                pattern = re.compile(r'island\.lk/[a-z0-9\-]+/?$')
                for a in soup.find_all('a', href=pattern):
                    slug = a['href'].rstrip('/').split('/')[-1]
                    if len(slug) < 20:
                        continue
                    title = a.get_text(strip=True)
                    if not title or len(title) < 10:
                        continue
                    url = a['href'].rstrip('/')
                    if not url.startswith('http'):
                        url = base + '/' + url.lstrip('/')
                    if url in seen or self.db.url_exists(url):
                        continue
                    seen.add(url)
                    articles.append(self._make_article(title, '', url, source_name, config))
                    if len(articles) >= 50:
                        break
            except Exception as e:
                logger.error(f"Error scraping the_island: {e}")
        return articles

    def _scrape_sundayobserver(self, source_name: str, config: Dict) -> List[Dict]:
        """Sunday Observer — RSS → HTML"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=['/feed/', '/feed', '/rss.xml', '/rss']
        )
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'sundayobserver\.lk/\d{4}/\d{2}/\d{2}/'),
            )
        return articles

    def _scrape_colombogazette(self, source_name: str, config: Dict) -> List[Dict]:
        """Colombo Gazette — WP REST API → RSS → HTML"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'^/\d{4}/\d{2}/\d{2}/'),
                'https://colombogazette.com',
            )
        return articles

    def _scrape_ceylontoday(self, source_name: str, config: Dict) -> List[Dict]:
        """Ceylon Today — WP REST API → RSS → date URL pattern"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'/\d{4}/\d{2}/\d{2}/'),
                config['url'].rstrip('/'),
            )
        return articles

    def _scrape_sundaytimes(self, source_name: str, config: Dict) -> List[Dict]:
        """Sunday Times — YYMMDD/category/slug pattern"""
        return self._scrape_html_pattern(
            source_name, config, config['url'],
            re.compile(r'sundaytimes\.lk/\d{6}/[^/]+/.+'),
            'https://sundaytimes.lk',
        )

    def _scrape_onlanka(self, source_name: str, config: Dict) -> List[Dict]:
        """Onlanka — WP REST API → RSS"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        return articles

    def _scrape_asianmirror(self, source_name: str, config: Dict) -> List[Dict]:
        """Asian Mirror — WP REST API → RSS"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        return articles

    def _scrape_lankanewsweb(self, source_name: str, config: Dict) -> List[Dict]:
        """Lanka News Web — WP REST API → RSS"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        return articles

    def _scrape_lankabusinessnews(self, source_name: str, config: Dict) -> List[Dict]:
        """Lanka Business News — WP REST API → RSS"""
        articles = self._try_wp_api(source_name, config)
        if not articles:
            articles = self._try_rss_paths(source_name, config)
        return articles

    def _scrape_newslk(self, source_name: str, config: Dict) -> List[Dict]:
        """News.lk — RSS → HTML"""
        articles = self._try_rss_paths(
            source_name, config,
            paths=['/feed', '/rss.xml', '/rss', '/?format=feed&type=rss']
        )
        if not articles:
            articles = self._scrape_html_pattern(
                source_name, config, config['url'],
                re.compile(r'^/news/[^/]+/.+'),
                'https://www.news.lk',
            )
        return articles

    def _scrape_hirunews(self, source_name: str, config: Dict) -> List[Dict]:
        """
        Hiru News — no RSS or WP API, and the listing page renders its stories
        client-side from a JSON endpoint. Reading that endpoint directly lets
        Sri Lankan coverage be selected by category at the source instead of by
        filtering article text afterwards, which would be lossy: most local
        stories never name the country ("Three arrested over Wattala shooting"),
        so a Sri Lanka keyword test drops around half of genuinely local news.

        The endpoint carries the full story text and a real publish time, so
        bodies are stored here (body_fetched=1) and the extractor step is
        skipped — it would otherwise refetch every article page for a noisier
        version of text already in hand.
        """
        seen_ids = set()
        articles = []

        for category in _HIRU_CATEGORIES:
            for page in range(1, _HIRU_PAGES + 1):
                try:
                    r = self.session.get(
                        _HIRU_API,
                        params={'page': page, 'category': category},
                        headers={'Referer': f"{config['url'].rstrip('/')}/en/news_listing.php"},
                        timeout=REQUEST_TIMEOUT,
                    )
                    r.raise_for_status()
                    items = r.json()
                except Exception as e:
                    logger.warning(f"{source_name}: {category} page {page} failed: {e}")
                    break

                if not items:
                    break

                for item in items:
                    article_id = str(item.get('eng_art_id') or '')
                    slug = (item.get('seourltitle') or '').strip()
                    title = unescape((item.get('eng_title') or '').strip())
                    if not article_id or not slug or len(title) < 10:
                        continue
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)

                    # seourltitle is 'english/<id>/<slug>'; the site serves it under /en/
                    url = f"{config['url'].rstrip('/')}/en/{re.sub(r'^english/', '', slug)}"
                    if self.db.url_exists(url):
                        continue

                    body = unescape((item.get('eng_story') or '').strip())
                    article = self._make_article(
                        title, body, url, source_name, config,
                        self._parse_date(item.get('eng_added_date')),
                    )
                    image = item.get('eng_image') or item.get('art_image')
                    article['image_url'] = _HIRU_IMAGE_BASE + image if image else None
                    article['body_fetched'] = 1 if body else 0
                    articles.append(article)

        return articles

    def _scrape_helakuru(self, source_name: str, config: Dict) -> List[Dict]:
        """
        Helakuru Esana — article pages are client-rendered stubs with no body in
        the HTML, so they can't be scraped or extracted the usual way. The site's
        own AJAX endpoint returns the articles as JSON, including an English
        translation of every headline and body.

        Bodies are captured here (body_fetched=1) because the later extractor
        step would find nothing at the article URLs.
        """
        base = config['url'].rstrip('/')
        landing = f'{base}/esana/'

        r = self.session.get(landing, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()

        m_name = re.search(r"csrfName\s*=\s*'([^']+)'", r.text)
        m_hash = re.search(r"csrfHash\s*=\s*'([^']+)'", r.text)
        if not (m_name and m_hash):
            logger.warning(f"{source_name}: CSRF token not found on {landing}")
            return []

        resp = self.session.post(
            f'{base}/esana/load',
            data={
                'newsLimit': 50,
                'esanaWidget': 'false',
                'category': '',
                m_name.group(1): m_hash.group(1),
            },
            headers={'X-Requested-With': 'XMLHttpRequest', 'Referer': landing},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        articles = []
        for item in (resp.json().get('NEWS') or []):
            url = (item.get('share_url') or '').strip().rstrip('/')
            title = (item.get('titleEn') or '').strip()
            if not url or len(title) < 8 or self.db.url_exists(url):
                continue

            body = self._helakuru_body(item.get('contentEn'))
            article = self._make_article(
                title, body, url, source_name, config,
                self._parse_date(item.get('published')),
            )
            article['image_url'] = item.get('cover') or item.get('thumb') or None
            article['body_fetched'] = 1 if body else 0
            articles.append(article)

        return articles

    @staticmethod
    def _helakuru_body(content) -> str:
        """contentEn is a list of blocks: {'type': 'text'|'image'|'line', 'data': ...}"""
        if not isinstance(content, list):
            return str(content or '').strip()
        parts = [
            str(block.get('data', '')).strip()
            for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        ]
        return ' '.join(p for p in parts if p)

    # ── Generic HTML fallback ──────────────────────────────────────────────

    def _scrape_html_generic(self, source_name: str, config: Dict) -> List[Dict]:
        try:
            r = self.session.get(config['url'], timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            elements = (
                soup.find_all('article') or
                soup.find_all('div', class_='article') or
                soup.find_all('div', class_='news-item') or
                soup.find_all('div', class_='post')
            )
            if not elements:
                logger.warning(f"  No article elements found on {source_name}")
                return []
            articles = []
            for el in elements[:50]:
                try:
                    title_elem = el.find('h1') or el.find('h2') or el.find('h3') or el.find('h4')
                    link_elem = el.find('a', href=True)
                    if not link_elem or not title_elem:
                        continue
                    url = link_elem.get('href', '')
                    if not url:
                        continue
                    if url.startswith('/'):
                        url = config['url'].rstrip('/') + url
                    elif not url.startswith('http'):
                        url = config['url'].rstrip('/') + '/' + url
                    if self.db.url_exists(url):
                        continue
                    title = title_elem.text.strip()
                    if title:
                        articles.append(self._make_article(title, '', url, source_name, config))
                except Exception:
                    continue
            return articles
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
            return []

    # ── Helpers ────────────────────────────────────────────────────────────

    def _make_article(self, title: str, body: str, url: str,
                      source_name: str, config: Dict,
                      published_at: str = None) -> Dict:
        return {
            'title': title[:500],
            'body': body[:50000],
            'url': url,
            'published_at': published_at or datetime.now().isoformat(),
            'source': source_name,
            'category': config.get('category', 'general'),
        }

    def _parse_date(self, date_str: str) -> str:
        if not date_str:
            return datetime.now().isoformat()
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).isoformat()
        except Exception:
            pass
        try:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(date_str).isoformat()
        except Exception:
            pass
        return datetime.now().isoformat()
