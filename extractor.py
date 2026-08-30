# extractor.py
"""
Full article body extractor using trafilatura.
Visits each article URL and extracts clean text — no site-specific selectors needed.
"""

import hashlib
import logging
import re
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

# How much of a body identifies it for duplicate detection. Whole-body hashes
# miss near-duplicates: the same wrongly-extracted block can come back with a
# trailing sentence trimmed and hash differently, which is exactly how wrong
# content used to slip through. The opening is the identifying part.
_FINGERPRINT_CHARS = 300

# A rescued body must share at least this fraction of its title's meaningful
# words. Duplicate detection alone cannot catch a wrongly-extracted block that
# happens to be unique within one batch — measured on economynext, an article
# titled "Singer Finance plans rights issue" was handed a body about
# diplomatic missions and passed every structural check. Title overlap
# separates them cleanly: correct extractions scored 0.62-1.00, that one 0.00.
_MIN_TITLE_OVERLAP = 0.20
_OVERLAP_STOPWORDS = frozenset(
    'the a an of in on at to for and or is was were be been by with from as '
    'its it this that has have had will would say says said sri lanka'.split()
)


def _fingerprint(text: str) -> str:
    """
    Hash a normalised PREFIX of a body, not the whole thing.

    Two extractions of the same wrong block often differ by a trailing
    sentence or some whitespace, which is enough to make whole-body hashes
    disagree — observed on economynext, where the same featured article came
    out as 2076 and 2047 chars and slipped past exact matching. The opening of
    an article is what actually identifies it.

    Module level rather than nested in run(), because the value is now stored
    on the row (articles.body_fingerprint) and compared across runs, so the
    backfill and repair script must produce byte-identical hashes. Two copies
    of this rule that drift apart would silently stop matching.
    """
    return hashlib.md5(
        ' '.join(text[:_FINGERPRINT_CHARS].split()).lower().encode()
    ).hexdigest()


def _title_overlap(title: str, body: str) -> float:
    """Fraction of the title's distinctive words that appear near the start of
    the body. A news article almost always restates its headline early on."""
    def _words(text):
        return {w for w in re.findall(r"[a-z0-9]+", text.lower())
                if w not in _OVERLAP_STOPWORDS and len(w) > 2}
    title_words = _words(title)
    if not title_words:
        return 1.0          # nothing to check against; don't reject on this
    return len(title_words & _words(body[:600])) / len(title_words)

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
                    body, alt_body, image_url = future.result()
                    results.append((article, body, alt_body, image_url))

            # Detect sidebar/widget content: same body >= 2 times from same source
            def _counts(idx):
                counts: dict = {}
                for row in results:
                    text = row[idx]
                    if text and len(text) >= _MIN_BODY_LEN:
                        h = _fingerprint(text)
                        counts.setdefault(row[0]['source'], {})
                        counts[row[0]['source']][h] = counts[row[0]['source']].get(h, 0) + 1
                return counts

            body_counts = _counts(1)   # favor_recall results
            alt_counts = _counts(2)    # favor_precision results

            # Bodies these sources have produced before, from any earlier run.
            # The counts above only see inside THIS batch, which is why a site
            # that serves the wrong article consistently slipped through: each
            # run saw the bad block once and judged it unique.
            known_fp = self.db.get_known_body_fingerprints(
                sorted({a['source'] for a in articles}))

            updated = 0
            rescued = 0
            discarded = 0
            stale = 0
            outcomes = []      # (source, kept_a_body) after discard/rescue
            for article, body, alt_body, image_url in results:
                src = article['source']

                def _is_dup(text, counts):
                    if not text or len(text) < _MIN_BODY_LEN:
                        return False
                    return counts.get(src, {}).get(_fingerprint(text), 0) >= 2

                if _is_dup(body, body_counts):
                    # The default extraction latched onto the same block for
                    # several articles from this source — it is boilerplate or
                    # the wrong article, not this one's body. Before throwing
                    # it away, try the stricter extraction: on sites where
                    # recall grabs a featured/sidebar block, precision usually
                    # picks the real one, and it is per-article rather than
                    # repeated.
                    rescuable = (
                        alt_body
                        and len(alt_body) >= _MIN_BODY_LEN
                        and not _is_dup(alt_body, alt_counts)
                        # Structural checks can't tell "wrong article" from
                        # "this article" when the wrong block appears only
                        # once in the batch. Require the body to actually be
                        # about the headline.
                        and _title_overlap(article.get('title', ''), alt_body) >= _MIN_TITLE_OVERLAP
                    )
                    if rescuable:
                        logger.debug(f"  Rescued with favor_precision: {article['url']}")
                        body = alt_body
                        rescued += 1
                    else:
                        logger.debug(f"  Discarded duplicate sidebar body: {article['url']}")
                        body = ''
                        discarded += 1

                # Cross-run check: has this source already given us this exact
                # opening under a DIFFERENT headline? If so it is that other
                # article's text, not this one's, however plausible it looks.
                # Storing it would put another story's words under this
                # headline AND — because identical text embeds identically —
                # glue the two into one cluster.
                fp = _fingerprint(body) if body and len(body) >= _MIN_BODY_LEN else None
                if fp:
                    seen_title = known_fp.get((src, fp))
                    if seen_title and seen_title.strip().lower() != (
                            article.get('title') or '').strip().lower():
                        logger.debug(f"  Body already seen under another headline "
                                     f"({seen_title[:50]}): {article['url']}")
                        body, fp = '', None
                        stale += 1

                self.db.update_body(article['id'], body, image_url, fingerprint=fp)
                if fp:
                    # Claim it, so a second copy later in this same batch is
                    # judged against it too.
                    known_fp[(src, fp)] = article.get('title') or ''
                outcomes.append((src, bool(body)))
                if body:
                    updated += 1

            if rescued:
                logger.info(f"  Rescued {rescued} bodies via favor_precision fallback")
            if discarded:
                logger.warning(f"  Discarded {discarded} duplicate bodies (sidebar/widget content)")
            if stale:
                logger.warning(f"  Discarded {stale} bodies this source had already "
                               f"served under a different headline")
            logger.info(f"Body extraction done: {updated}/{len(articles)} successful")
            _report_failures(outcomes)

        return len(articles)

    def _fetch(self, url: str):
        """
        Returns (body_text, alt_body_text, image_url).

        Two extractions from the SAME html — no extra request:

          body      favor_recall, the default. Gets the most text and works
                    on the widest range of layouts.
          alt_body  favor_precision, a stricter fallback. Only used when the
                    recall result turns out to be duplicated across articles
                    from one source (see run()).

        Why both: on some sites recall latches onto the wrong block — a
        featured or sidebar article — and returns the *same* text for every
        page. Measured on economynext, that accounted for 417 empty bodies:
        extraction wasn't failing, it was confidently returning the wrong
        article, which duplicate detection then correctly threw away.
        Precision picks the right block there. It is not the default because
        it regresses other sites badly (business_today: 16k chars -> 260).
        """
        try:
            with _host_semaphore(url):
                r = self.session.get(url, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                html = r.text
            if not html:
                return '', '', ''

            text = trafilatura.extract(
                html,
                favor_recall=True,
                include_comments=False,
                include_tables=False,
            ) or ''
            alt = trafilatura.extract(
                html,
                favor_precision=True,
                include_comments=False,
                include_tables=False,
            ) or ''

            image_url = _extract_og_image(html)
            return text, alt, image_url
        except Exception as e:
            logger.warning(f"Body fetch failed for {url}: {e}")
            return '', '', ''


# A source can fail extraction completely and look exactly like a source that
# simply had no new articles: the row is written, body_fetched flips to 1, and
# nothing is logged. the_island ran at a 100% failure rate for weeks that way —
# 192 articles scraped, 192 bodies empty, no error anywhere. Every run now says
# so out loud.
_FAIL_WARN_PCT = 30.0


def _report_failures(outcomes) -> None:
    """
    Per-source empty-body rate for this batch, worst first.

    Takes the outcome AFTER the discard/rescue decision, not the raw fetch
    result — a body that was fetched fine and then thrown away as a duplicate
    is still an article with no text, and hiding that is what let daily_ft sit
    at a 41% failure rate unnoticed.
    """
    per_source: dict = {}
    for src, kept in outcomes:
        got, total = per_source.get(src, (0, 0))
        per_source[src] = (got + (1 if kept else 0), total + 1)

    failing = sorted(
        ((src, total - got, total, 100.0 * (total - got) / total)
         for src, (got, total) in per_source.items() if got < total),
        key=lambda r: -r[3],
    )
    if not failing:
        return
    logger.info("  empty bodies by source:")
    for src, missed, total, pct in failing:
        line = f"     {src:<22} {missed:>3}/{total:<3} {pct:5.1f}%"
        # A third of a source failing is a broken selector or a bot wall, not
        # noise — loud enough to notice in a scrolling log.
        if pct >= _FAIL_WARN_PCT:
            logger.warning(line + "   <-- check this source")
        else:
            logger.info(line)


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
