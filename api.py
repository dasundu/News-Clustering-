# api.py
"""
FastAPI REST API for clustered Sri Lankan news articles — JSON only. The UI
is a separate Next.js app in frontend/ (its own process/server, not served
by this one); see frontend/README or the repo root README's Deployment
section for how the two are run together.
Run (from backend/): python main.py --api
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_serializer
from typing import Any, List, Optional
from datetime import datetime as dt
import threading
import uvicorn
import os
from datetime import datetime
from urllib.parse import urlparse

from database import ArticleDatabase
from config import API_HOST, API_PORT, classify_category, SEARCH_ENABLED, SEARCH_TOP_K

app = FastAPI(title="SL News Clustering API", version="1.0")

# Set PIPELINE_ENABLED=false for a read-only deployment (e.g. a lightweight
# API-only host serving /clusters etc. to other components, with the actual
# scrape/embed/cluster loop running elsewhere against the same database).
# Such a deployment installs requirements-api.txt, not the full
# requirements.txt — it never has scraper/embedder/clusterer's dependencies
# (torch, trafilatura, ...) available, so the routes that import them must be
# turned off explicitly rather than left to fail with an ImportError.
#
# "Read-only" is meant literally: it gates *every* route that mutates state,
# not only the ones that would ImportError. /sources add+delete and the
# maintenance route need no heavy imports and so would happily run on such a
# deployment — but this API has no authentication, so any host reachable by
# another team (or tunnelled out for a demo) must be able to refuse writes
# from anyone who happens to have the URL.
PIPELINE_ENABLED = os.environ.get('PIPELINE_ENABLED', 'true').lower() != 'false'


def _require_writes():
    """Guard for every state-mutating route. Kept as one helper so a new
    write endpoint has an obvious thing to call rather than re-deriving the
    check (and forgetting, as /sources originally did)."""
    if not PIPELINE_ENABLED:
        raise HTTPException(status_code=503, detail='pipeline_disabled_on_this_deployment')

# CORS is wide open deliberately: the Next.js frontend (its own server/port,
# possibly its own container or host entirely) calls this API cross-origin,
# and so can any other consumer — see README's "give my other components an
# API to get clusters" use case.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = ArticleDatabase()


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "SL News Clustering API",
        "docs": "/docs",
        "frontend": "run separately — see frontend/ (Next.js)",
    }


# ── Pipeline runner ────────────────────────────────────────────────────────

_STEPS = [
    {'n': 1, 'name': 'Scraping news sites'},
    {'n': 2, 'name': 'Extracting article bodies'},
    {'n': 3, 'name': 'Cleaning text'},
    {'n': 4, 'name': 'Generating embeddings'},
    {'n': 5, 'name': 'Clustering news events'},
]

_pipeline_state = {
    'running': False,
    'steps': [{**s, 'status': 'pending', 'detail': ''} for s in _STEPS],
    'error': None,
    'finished_at': None,
}
_pipeline_lock = threading.Lock()


def _set_step(n: int, status: str, detail: str = ''):
    with _pipeline_lock:
        _pipeline_state['steps'][n - 1]['status'] = status
        _pipeline_state['steps'][n - 1]['detail'] = detail


# ── Console colours ────────────────────────────────────────────────────────
_C = {
    'reset':  '\033[0m',
    'bold':   '\033[1m',
    'dim':    '\033[2m',
    'cyan':   '\033[96m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'blue':   '\033[94m',
    'white':  '\033[97m',
    'gray':   '\033[90m',
}

def _cprint(color: str, text: str):
    print(f"{_C.get(color,'')}{text}{_C['reset']}", flush=True)

def _step_header(n: int, label: str):
    bar = '─' * 50
    inner = f'  STEP {n}/5  {label}'
    padded = f'{inner:<52}'
    print(flush=True)
    print(f"{_C['cyan']}┌{bar}┐{_C['reset']}", flush=True)
    print(f"{_C['cyan']}│{_C['reset']}{_C['bold']}{padded}{_C['reset']}{_C['cyan']}│{_C['reset']}", flush=True)
    print(f"{_C['cyan']}└{bar}┘{_C['reset']}", flush=True)

def _step_done(detail: str):
    _cprint('green', f'  ✔  {detail}')

def _step_error(detail: str):
    _cprint('red', f'  ✘  {detail}')


def _run_pipeline_background():
    with _pipeline_lock:
        _pipeline_state['running'] = True
        _pipeline_state['error'] = None
        _pipeline_state['finished_at'] = None
        _pipeline_state['steps'] = [{**s, 'status': 'pending', 'detail': ''} for s in _STEPS]

    started = datetime.now()
    print(flush=True)
    _cprint('bold', '╔══════════════════════════════════════════════════════╗')
    _cprint('bold', f'║   NEWS CLUSTERING PIPELINE  —  {started.strftime("%Y-%m-%d %H:%M:%S")}   ║')
    _cprint('bold', '╚══════════════════════════════════════════════════════╝')

    try:
        # ── Step 1 — Scrape ────────────────────────────────────────────────
        _step_header(1, 'Scraping news sites')
        _set_step(1, 'running')
        from scraper import NewsScraper
        scraper = NewsScraper()
        r = scraper.scrape_all_sources()
        c = scraper.scrape_custom_sources()
        total_new   = r['total_new']   + c['total_new']
        total_found = r['total_found'] + c['total_found']
        detail1 = f'{total_found} articles found  |  {total_new} new saved to DB'
        _set_step(1, 'done', f'{total_found} found · {total_new} new')
        _step_done(detail1)

        # ── Step 2 — Extract bodies ────────────────────────────────────────
        _step_header(2, 'Extracting article bodies')
        _set_step(2, 'running')
        from extractor import BodyExtractor
        extracted = BodyExtractor().run()
        detail2 = f'{extracted} article bodies downloaded & stored'
        _set_step(2, 'done', f'{extracted} bodies fetched')
        _step_done(detail2)

        # ── Step 3 — Clean text ────────────────────────────────────────────
        _step_header(3, 'Cleaning & normalising text')
        _set_step(3, 'running')
        from cleaner import TextCleaner
        cleaned = TextCleaner().run()
        detail3 = f'{cleaned} articles cleaned  (encoding fixed, HTML stripped)'
        _set_step(3, 'done', f'{cleaned} articles cleaned')
        _step_done(detail3)

        # ── Step 4 — Embed ────────────────────────────────────────────────
        _step_header(4, 'Generating semantic embeddings')
        _set_step(4, 'running')
        _cprint('gray', '  ↳  Model: paraphrase-multilingual-MiniLM-L12-v2  (384 dims)')
        from embedder import ArticleEmbedder
        embedded = ArticleEmbedder().run()
        detail4 = f'{embedded} articles → 384-dimensional vectors'
        _set_step(4, 'done', f'{embedded} articles embedded')
        _step_done(detail4)

        # ── Step 5 — Cluster ──────────────────────────────────────────────
        _step_header(5, 'Clustering news events  (HDBSCAN)')
        _set_step(5, 'running')
        _cprint('gray', '  ↳  Algorithm: HDBSCAN  |  ε = 0.3  |  min_cluster = 2')
        from clusterer import ArticleClusterer
        result = ArticleClusterer().run()
        clustered = result['total'] - result['noise']
        detail5 = (f"{result['clusters']} clusters formed  |  "
                   f"{clustered} articles grouped  |  "
                   f"{result['noise']} noise/singletons")
        _set_step(5, 'done',
                  f"{result['clusters']} clusters · {clustered} clustered")
        _step_done(detail5)

        # ── Summary ───────────────────────────────────────────────────────
        elapsed = (datetime.now() - started).seconds
        print(flush=True)
        _cprint('bold', '╔══════════════════════════════════════════════════════╗')
        _cprint('green', f'║   PIPELINE COMPLETE  ✔   ({elapsed}s elapsed){" " * (20 - len(str(elapsed)))}║')
        _cprint('bold', f'║   {total_new} new  |  {extracted} extracted  |  {result["clusters"]} clusters{" " * max(0, 22 - len(str(total_new)) - len(str(extracted)) - len(str(result["clusters"])))}║')
        _cprint('bold', '╚══════════════════════════════════════════════════════╝')
        print(flush=True)

    except Exception as e:
        print(flush=True)
        _cprint('red', f'  ✘  PIPELINE ERROR: {e}')
        print(flush=True)
        with _pipeline_lock:
            _pipeline_state['error'] = str(e)
        for step in _pipeline_state['steps']:
            if step['status'] == 'running':
                step['status'] = 'error'
    finally:
        with _pipeline_lock:
            _pipeline_state['running'] = False
            _pipeline_state['finished_at'] = datetime.now().isoformat()


@app.post('/pipeline/run')
def run_pipeline():
    """Start the full pipeline in the background."""
    if not PIPELINE_ENABLED:
        raise HTTPException(status_code=503, detail='pipeline_disabled_on_this_deployment')
    with _pipeline_lock:
        if _pipeline_state['running']:
            return {'status': 'already_running'}
    t = threading.Thread(target=_run_pipeline_background, daemon=True)
    t.start()
    return {'status': 'started'}


@app.get('/pipeline/status')
def pipeline_status():
    """Current pipeline progress."""
    with _pipeline_lock:
        return dict(_pipeline_state)


# ── response models ────────────────────────────────────────────────────────

def _dt_str(v):
    if v is None:
        return None
    if isinstance(v, dt):
        return v.isoformat()
    return str(v)

class ArticleSummary(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    id: int
    title: str
    source: str
    category: Optional[str] = None
    published_at: Optional[Any] = None
    url: str
    cluster_id: Optional[int] = None
    # Defaults matter here: in pydantic v2 `Optional[str]` WITHOUT a default is
    # a *required* field that merely accepts None. Any route returning rows
    # that don't carry the key therefore 500s on response validation —
    # /articles/{id} and /clusters/{id}/articles both did, on every call.
    body_preview: Optional[str] = None

    @field_serializer('published_at')
    def _pub(self, v): return _dt_str(v)

class ArticleFull(ArticleSummary):
    body: Optional[str] = None
    scraped_at: Optional[Any] = None

    @field_serializer('scraped_at')
    def _scr(self, v): return _dt_str(v)

class ArticleInCluster(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    id: int
    title: str
    source: str
    url: str
    published_at: Optional[Any] = None
    body: Optional[str] = None
    image_url: Optional[str] = None

    @field_serializer('published_at')
    def _pub(self, v): return _dt_str(v)

class ClusterSummary(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    cluster_id: int
    event_title: str
    article_count: int
    sources: List[str]
    latest_at: Optional[Any] = None
    cluster_image: Optional[str] = None
    category: Optional[str] = None
    # The topic this event belongs to, if it grouped with related events;
    # null for a standalone event (see clusterer.ClusterTopicBuilder).
    topic_id: Optional[int] = None
    topic_title: Optional[str] = None

    @field_serializer('latest_at')
    def serialize_latest_at(self, v):
        if v is None:
            return None
        if isinstance(v, dt):
            return v.isoformat()
        return str(v)

class ClusterDetail(ClusterSummary):
    cluster_image: Optional[str] = None
    articles: List[ArticleInCluster]

class TopicSummary(BaseModel):
    model_config = {'arbitrary_types_allowed': True}
    topic_id: int
    topic_title: str
    cluster_count: int
    article_count: int
    sources: List[str]
    latest_at: Optional[Any] = None
    topic_image: Optional[str] = None
    category: Optional[str] = None

    @field_serializer('latest_at')
    def serialize_latest_at(self, v):
        if v is None:
            return None
        if isinstance(v, dt):
            return v.isoformat()
        return str(v)

class TopicDetail(BaseModel):
    topic_id: int
    topic_title: str
    cluster_count: int
    clusters: List[ClusterSummary]

class SearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      description="The news claim to look for, in free text")
    top_k: int = Field(SEARCH_TOP_K, ge=1, le=20)

class MatchedArticle(BaseModel):
    id: int
    title: str
    source: str
    url: str
    # null when the article was found only by the keyword channel and so has
    # no dense-similarity score of its own
    score: Optional[float] = None

class MatchedTopic(BaseModel):
    topic_id: int
    topic_title: str
    confidence: float

class SearchMatch(BaseModel):
    cluster_id: int
    event_title: str
    confidence: float
    topic_id: Optional[int] = None
    topic_title: Optional[str] = None
    category: Optional[str] = None
    article_count: int
    sources: List[str]
    latest_at: Optional[Any] = None
    matched_articles: List[MatchedArticle]

class SearchResponse(BaseModel):
    status: str          # 'found' | 'partial' | 'not_found'
    query: str
    best_score: float
    matches: List[SearchMatch]
    matched_topics: List[MatchedTopic]
    interpretation: Optional[str] = None

class Stats(BaseModel):
    total_articles: int
    clustered: int
    unclustered: int
    total_clusters: int
    by_source: dict
    by_category: dict

class SubmitResult(BaseModel):
    status: str
    article_id: Optional[int] = None
    title: Optional[str] = None
    category: Optional[str] = None
    cluster_id: Optional[int] = None

class CustomSourceIn(BaseModel):
    url: str
    name: Optional[str] = None

class CustomSource(BaseModel):
    # psycopg2 hands these back as datetime objects, and pydantic v2 will not
    # silently coerce a datetime into a `str` field — declaring them as such
    # made both POST /sources and (once any custom source existed) GET
    # /sources fail with a ResponseValidationError. Serialized explicitly
    # here, matching how ClusterSummary/TopicSummary handle latest_at.
    model_config = {'arbitrary_types_allowed': True}
    id: int
    name: str
    url: str
    added_at: Optional[Any] = None
    last_scraped: Optional[Any] = None

    @field_serializer('added_at', 'last_scraped')
    def serialize_ts(self, v):
        if v is None:
            return None
        if isinstance(v, dt):
            return v.isoformat()
        return str(v)


# ── helpers ────────────────────────────────────────────────────────────────

def _to_summary(row: dict) -> dict:
    body = row.get('body') or ''
    row['body_preview'] = body[:300] if body else ''
    return row


# ── endpoints ─────────────────────────────────────────────────────────────

@app.post("/search", response_model=SearchResponse)
def search_claim(body: SearchRequest):
    """
    Find event clusters covering a free-text news claim.

    Built for an external agent asking "is this news in your database?" —
    send the text a user described, get back matching `cluster_id`s.

    Hybrid retrieval: dense (embedding cosine) + keyword (full-text search),
    fused by Reciprocal Rank Fusion, then aggregated from articles up to the
    events they belong to.

    `status` is one of:
      * `found`     — confident match; use these cluster_ids
      * `partial`   — related coverage only; candidates returned but NOT
                      asserted as a match
      * `not_found` — nothing resembling this claim is indexed

    All three are HTTP 200 — "we looked and it isn't there" is a successful
    query, not an error.

    IMPORTANT: `not_found` means these ~20 Sri Lankan outlets have no indexed
    coverage resembling the claim. It is **not** evidence the claim is false.
    Consumers doing fact-checking must not treat absence as disproof.
    """
    if not SEARCH_ENABLED:
        # Same guard shape as PIPELINE_ENABLED: this route needs the
        # embedding model in-process to vectorize the query, which the
        # lightweight read-only image (requirements-api.txt) doesn't ship.
        raise HTTPException(status_code=503, detail='search_disabled_on_this_deployment')
    try:
        from searcher import ClaimSearcher
    except ImportError as e:
        raise HTTPException(status_code=503,
                            detail=f'search_dependencies_unavailable: {e}')
    return ClaimSearcher(db).search(body.text, top_k=body.top_k)


@app.get("/topics", response_model=List[TopicSummary])
def list_topics(
    min_clusters: int = Query(2, ge=1, description="Minimum event clusters in a topic"),
):
    """
    List topics — the grouping layer above event clusters. One topic holds
    several distinct-but-related events ("IMF tranche approved", "IMF terms
    debated in parliament"). Events that didn't group with anything have no
    topic and don't appear here; find them via /clusters.
    """
    return db.get_topics(min_clusters=min_clusters)


@app.get("/topics/{topic_id}", response_model=TopicDetail)
def get_topic(topic_id: int):
    """Get a topic and every event cluster filed under it."""
    topic = db.get_topic(topic_id)
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    clusters = db.get_clusters(min_size=1, topic_id=topic_id)
    return {
        'topic_id': topic['topic_id'],
        'topic_title': topic['topic_title'],
        'cluster_count': len(clusters),
        'clusters': clusters,
    }


@app.get("/clusters", response_model=List[ClusterSummary])
def list_clusters(
    min_size: int = Query(2, ge=1),
    category: Optional[str] = Query(None, description="Filter by category: economics or politics"),
    topic_id: Optional[int] = Query(None, description="Only clusters filed under this topic"),
    sort: str = Query('latest', pattern='^(latest|sources|articles)$',
                      description="latest = newest first; sources = most outlets first; "
                                  "articles = most articles first"),
):
    """
    List event clusters.

    `sort=sources` surfaces the most widely-reported events — how many
    *distinct outlets* carried the story, which is a better "biggest story"
    signal than raw article count, since one outlet posting five updates
    isn't five outlets independently corroborating.
    """
    return db.get_clusters(min_size=min_size, category=category,
                           topic_id=topic_id, sort=sort)


@app.get("/clusters/{cluster_id}", response_model=ClusterDetail)
def get_cluster(cluster_id: int):
    """Get event name + all articles grouped under it."""
    articles = db.get_cluster_articles(cluster_id)
    if not articles:
        raise HTTPException(status_code=404, detail="Cluster not found")

    sources = list({a['source'] for a in articles})
    latest_at = max((a['published_at'] or '') for a in articles)
    labels = db.get_cluster_labels()
    event_title = labels.get(cluster_id) or max(articles, key=lambda a: len(a['title']))['title']

    # Hide articles with no extracted body
    articles = [a for a in articles if a.get('body')]

    # Deduplicate: same source + same title → keep only the most recent
    seen_keys: set = set()
    deduped = []
    for a in articles:
        key = (a['source'], a['title'].strip().lower())
        if key not in seen_keys:
            seen_keys.add(key)
            deduped.append(a)
    articles = deduped

    cluster_image = next(
        (a.get('image_url') for a in articles if a.get('image_url')), None
    )

    article_list = [
        {
            'id': a['id'],
            'title': a['title'],
            'source': a['source'],
            'url': a['url'],
            'published_at': a['published_at'],
            'body': a.get('body') or '',
            'image_url': a.get('image_url') or None,
        }
        for a in articles
    ]

    topic = db.get_topic_for_cluster(cluster_id)

    return {
        'cluster_id': cluster_id,
        'event_title': event_title,
        'article_count': len(articles),
        'sources': sources,
        'latest_at': latest_at,
        'cluster_image': cluster_image,
        'topic_id': topic['topic_id'] if topic else None,
        'topic_title': topic['topic_title'] if topic else None,
        'articles': article_list,
    }


@app.get("/clusters/{cluster_id}/articles", response_model=List[ArticleFull])
def cluster_articles(cluster_id: int):
    """Get all full articles in a cluster."""
    articles = db.get_cluster_articles(cluster_id)
    if not articles:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return [_to_summary(a) for a in articles]


@app.get("/articles", response_model=List[ArticleSummary])
def list_articles(
    source: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """List articles with optional filtering by source or category."""
    rows = db.get_articles(source=source, category=category, limit=limit, offset=offset)
    return [_to_summary(r) for r in rows]


@app.get("/articles/{article_id}", response_model=ArticleFull)
def get_article(article_id: int):
    """Get a single article with full body text."""
    article = db.get_article_by_id(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return _to_summary(article)


@app.get("/stats", response_model=Stats)
def stats():
    """System statistics: article counts, cluster counts, source breakdown."""
    cluster_stats = db.get_cluster_stats()
    total, by_source, by_category = db.get_stats()
    return {**cluster_stats, 'by_source': by_source, 'by_category': by_category}


@app.post("/submit", response_model=SubmitResult)
def submit_url(url: str = Query(..., description="Full URL of the news article to add")):
    """Manually submit a single article URL. Fetches, classifies, embeds, and clusters it."""
    if not PIPELINE_ENABLED:
        raise HTTPException(status_code=503, detail='pipeline_disabled_on_this_deployment')
    import trafilatura
    from bs4 import BeautifulSoup

    # Check duplicate
    if db.url_exists(url):
        article_id = db.get_article_id_by_url(url)
        article = db.get_article_by_id(article_id)
        return SubmitResult(
            status='exists',
            article_id=article_id,
            title=article['title'] if article else None,
            category=article['category'] if article else None,
            cluster_id=article['cluster_id'] if article else None,
        )

    # Fetch HTML
    html = trafilatura.fetch_url(url)
    if not html:
        raise HTTPException(status_code=422, detail='fetch_failed')

    # Extract body
    body = trafilatura.extract(
        html, favor_recall=True, include_comments=False, include_tables=False
    ) or ''

    # Extract title: og:title → h1 → <title> → hostname fallback
    soup = BeautifulSoup(html, 'html.parser')
    title = ''
    og = soup.find('meta', property='og:title')
    if og and og.get('content'):
        title = og['content'].strip()
    if not title:
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
    if not title:
        t = soup.find('title')
        if t:
            title = t.get_text(strip=True)
    if not title:
        title = urlparse(url).netloc

    # Classify — manual submit saves even if no keyword match (category='general')
    category = classify_category(title + ' ' + body[:500]) or 'general'
    source = urlparse(url).netloc.replace('www.', '')

    # Save
    article_dict = {
        'title': title[:500],
        'body': body,
        'url': url,
        'published_at': datetime.now().isoformat(),
        'source': source,
        'category': category,
    }
    db.save_article(article_dict)
    article_id = db.get_article_id_by_url(url)
    if article_id:
        db.update_body(article_id, body)  # sets body_fetched=1

    # Run pipeline steps in background — don't block the request
    def _run():
        from cleaner import TextCleaner
        from embedder import ArticleEmbedder
        from clusterer import ArticleClusterer
        TextCleaner().run()
        ArticleEmbedder().run()
        ArticleClusterer().run()

    threading.Thread(target=_run, daemon=True).start()

    return SubmitResult(
        status='ok',
        article_id=article_id,
        title=title,
        category=category,
        cluster_id=None,
    )


# ── Custom sources ─────────────────────────────────────────────────────────

@app.get("/sources/builtin")
def list_builtin_sources():
    """Return the built-in news sources defined in config."""
    from config import NEWS_SOURCES
    return [
        {
            'key': key,
            'name': cfg.get('description', key).split(' — ')[0],
            'url': cfg['url'],
            'category': cfg.get('category', 'general'),
            'priority': cfg.get('priority', 'MEDIUM'),
        }
        for key, cfg in NEWS_SOURCES.items()
    ]


@app.get("/sources", response_model=List[CustomSource])
def list_sources():
    """List all user-added custom news sources."""
    return db.get_custom_sources()


@app.post("/sources", response_model=CustomSource, status_code=201)
def add_source(body: CustomSourceIn):
    """Add a new website to scrape. Articles are keyword-filtered to economics/politics."""
    _require_writes()
    url = body.url.rstrip('/')
    name = body.name or urlparse(url).netloc.replace('www.', '')
    added = db.add_custom_source(name, url)
    if not added:
        raise HTTPException(status_code=409, detail='Source already exists')
    sources = db.get_custom_sources()
    match = next((s for s in sources if s['url'] == url), None)
    if not match:
        raise HTTPException(status_code=500, detail='Failed to retrieve saved source')
    return match


@app.post("/sources/{source_id}/scrape")
def scrape_source_now(source_id: int):
    """Immediately scrape a custom source and run embed+cluster in the background."""
    if not PIPELINE_ENABLED:
        raise HTTPException(status_code=503, detail='pipeline_disabled_on_this_deployment')
    sources = db.get_custom_sources()
    src = next((s for s in sources if s['id'] == source_id), None)
    if not src:
        raise HTTPException(status_code=404, detail='Source not found')

    def _run():
        from scraper import NewsScraper
        from cleaner import TextCleaner
        from embedder import ArticleEmbedder
        from clusterer import ArticleClusterer
        from config import classify_category
        scraper = NewsScraper()
        config = {'url': src['url'], 'type': 'html', 'category': 'general'}
        articles = scraper._try_rss_paths(src['name'], config) or \
                   scraper._scrape_html_generic(src['name'], config)
        filtered = []
        for a in articles:
            cat = classify_category(a['title'])
            if cat:
                a['category'] = cat
                filtered.append(a)
        db.save_batch(filtered)
        db.touch_custom_source(source_id)
        TextCleaner().run()
        ArticleEmbedder().run()
        ArticleClusterer().run()

    threading.Thread(target=_run, daemon=True).start()
    return {'status': 'started', 'source_id': source_id}


@app.delete("/sources/{source_id}")
def delete_source(source_id: int):
    """Remove a custom source."""
    _require_writes()
    deleted = db.delete_custom_source(source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Source not found')
    return {'status': 'deleted'}


@app.post('/maintenance/fix-encoding')
def fix_encoding_endpoint():
    """Fix mojibake in all existing article titles and cluster labels."""
    _require_writes()
    from cleaner import fix_encoding
    fixed_articles = 0
    fixed_labels = 0

    for row in db.get_all_titles():
        fixed = fix_encoding(row['title'])
        if fixed != row['title']:
            db.update_article_title(row['id'], fixed)
            fixed_articles += 1

    for row in db.get_all_cluster_labels():
        fixed = fix_encoding(row['event_title'])
        if fixed != row['event_title']:
            db.save_cluster_label(row['cluster_id'], fixed)
            fixed_labels += 1

    return {'fixed_article_titles': fixed_articles, 'fixed_cluster_labels': fixed_labels}


if __name__ == '__main__':
    uvicorn.run(app, host=API_HOST, port=API_PORT)
