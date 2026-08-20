# config.py
"""
Configuration for news sources, NLP pipeline, and API.
"""

import os
from typing import Optional

# Load .env file if present (before any other config reads)
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()

# ── Keyword classifier ─────────────────────────────────────────────────────

ECONOMICS_KEYWORDS = {
    'economy', 'economic', 'economics', 'finance', 'financial', 'budget',
    'inflation', 'gdp', 'growth', 'trade', 'investment', 'tax', 'revenue',
    'bank', 'banking', 'stock', 'market', 'rupee', 'currency', 'export',
    'import', 'debt', 'deficit', 'surplus', 'fiscal', 'monetary',
    'interest rate', 'loan', 'credit', 'imf', 'world bank', 'business',
    'profit', 'loss', 'earnings', 'shares', 'bond', 'treasury', 'forex',
    'remittance', 'tourism revenue', 'ipr', 'tariff', 'subsidy',
}

POLITICS_KEYWORDS = {
    'parliament', 'parliamentary', 'political', 'politics', 'government',
    'minister', 'ministry', 'president', 'election', 'vote', 'voting',
    'party', 'policy', 'cabinet', 'opposition', ' mp ', 'legislation',
    'bill', 'act ', 'law', 'protest', 'diplomatic', 'diplomacy', 'foreign',
    'corruption', 'rajapaksa', 'wickremesinghe', 'dissanayake', 'sirisena',
    'military', 'defence', 'security forces', 'supreme court', 'court of appeal',
    'human rights', 'constitution', 'referendum', 'sanction',

    # Governance and judicial affairs. Sri Lankan outlets report these without
    # ever using the words above — a gazette or a judicial appointment is
    # policy news that 'parliament'/'minister' alone would miss.
    'gazette', 'magistrate', 'judge', 'judicial', 'judiciary', 'tribunal',
    'attorney general', 'solicitor general', 'ordinance', 'amendment',
    'commission', 'inquiry', 'probe', 'bribery', 'impeach', 'governor',
    'ambassador', 'envoy', 'treaty', 'state minister', 'secretary to the',

    # Law enforcement and custodial affairs. Kept because these carry the
    # national security stories the other outlets cover politically (prison
    # unrest, senior police appointments); the cost is some routine crime.
    'police', 'prison', 'inmate', 'remand', 'custody', 'court',
}


def classify_category(text: str) -> Optional[str]:
    """Return 'economics', 'politics', or None (no keyword match)."""
    t = text.lower()
    if any(kw in t for kw in ECONOMICS_KEYWORDS):
        return 'economics'
    if any(kw in t for kw in POLITICS_KEYWORDS):
        return 'politics'
    return None


# ── News sources ───────────────────────────────────────────────────────────
# filter_keywords=False  →  all articles from this source are kept as-is
# filter_keywords=True   →  only articles matching ECONOMICS/POLITICS keywords
#                            are kept; others are discarded

NEWS_SOURCES = {

    # ── Business / Economics ───────────────────────────────────────────────
    # All articles are economics-focused — keep everything, no filtering
    'economynext': {
        'url': 'https://economynext.com',
        'rss': 'https://economynext.com/feed',
        'type': 'rss',
        'category': 'economics',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'EconomyNext — economic news',
    },
    'lankabusinessonline': {
        'url': 'https://www.lankabusinessonline.com',
        'rss': None,
        'type': 'html',
        'category': 'economics',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'Lanka Business Online — business and economy',
    },
    'daily_ft': {
        'url': 'https://www.ft.lk',
        'rss': None,
        'type': 'html',
        'category': 'economics',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'Daily FT — Sri Lanka business newspaper',
    },
    'business_today': {
        'url': 'https://www.businesstoday.lk',
        'rss': None,
        'type': 'html',
        'category': 'economics',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Business Today Sri Lanka — business and economy',
    },

    # ── General — keyword-filtered (keep only economics + politics) ────────
    # Mixed content sites — scrape all, discard non-matching articles
    'adaderana': {
        'url': 'https://www.adaderana.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'HIGH',
        'description': 'Ada Derana — general breaking news',
    },
    'newsfirst': {
        'url': 'https://english.newsfirst.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'HIGH',
        'description': 'News First — general 24/7 news',
    },
    'daily_mirror': {
        'url': 'https://www.dailymirror.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'HIGH',
        'description': 'Daily Mirror — general news',
    },
    'the_morning': {
        'url': 'https://www.themorning.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'HIGH',
        'description': 'The Morning — general news',
    },
    'daily_news': {
        'url': 'https://www.dailynews.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Daily News — state-owned general news',
    },
    'hirunews': {
        'url': 'https://www.hirunews.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'HIGH',
        'description': 'Hiru News — general news',
    },
    'the_island': {
        'url': 'https://island.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'The Island — general news',
    },
    'sunday_observer': {
        'url': 'https://www.sundayobserver.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Sunday Observer — general news',
    },
    'colombo_gazette': {
        'url': 'https://colombogazette.com',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Colombo Gazette — general news',
    },
    'ceylon_today': {
        'url': 'https://ceylontoday.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Ceylon Today — general news',
    },
    'sunday_times': {
        'url': 'https://sundaytimes.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Sunday Times Sri Lanka — general news',
    },

    # ── Newly researched sources (added Aug 2026) ──────────────────────────
    # All four verified live: valid RSS/WP-API feed, real recent Sri Lankan
    # articles, and a permissive robots.txt (general User-agent: * is
    # Allow: /; the sites' Disallow rules target named AI-training bots
    # specifically, not general crawling — see README for the full check).
    'onlanka': {
        'url': 'https://www.onlanka.com',
        'rss': 'https://www.onlanka.com/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Onlanka — Sri Lanka news and information portal',
    },
    'asianmirror': {
        'url': 'https://asianmirror.lk',
        'rss': 'https://www.asianmirror.lk/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Asian Mirror — politics, international relations and business',
    },
    'lankanewsweb': {
        'url': 'https://lankanewsweb.net',
        'rss': 'https://www.lankanewsweb.net/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Lanka News Web — current affairs and economic analysis',
    },
    'lankabusinessnews': {
        'url': 'https://www.lankabusinessnews.com',
        'rss': 'https://www.lankabusinessnews.com/feed',
        'type': 'rss',
        'category': 'economics',
        'filter_keywords': False,   # business-only wire, same treatment as economynext
        'priority': 'MEDIUM',
        'description': 'Lanka Business News — Sri Lankan business wire',
    },

    # ── Government / Policy ────────────────────────────────────────────────
    # Official government news — keyword-filtered
    'news_lk': {
        'url': 'https://www.news.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'News.lk — government and policy news',
    },

    # ── Sinhala aggregator with English translations ───────────────────────
    # Article pages are client-rendered; scraped via the site's JSON endpoint.
    # Headlines and bodies are taken from the English fields it returns.
    'helakuru': {
        'url': 'https://www.helakuru.lk',
        'rss': None,
        'type': 'json',
        'category': 'general',
        'filter_keywords': True,
        'priority': 'MEDIUM',
        'description': 'Helakuru Esana — breaking news aggregator',
    },
}

_outlet_name_fragments_cache: Optional[set] = None


def outlet_name_fragments() -> set:
    """
    Lowercased outlet display names derived from NEWS_SOURCES, e.g.
    {'daily ft', 'economynext', ...}. Shared by clusterer.py (stripping a
    trailing ' - Daily FT' attribution off a generated label) and
    embedder.py (stripping outlet names out of the text that gets embedded,
    so publisher identity doesn't leak into the semantic vector — an outlet
    name appearing verbatim in most of that outlet's articles would
    otherwise give the embedding model a shortcut that has nothing to do
    with the article's actual topic).
    """
    global _outlet_name_fragments_cache
    if _outlet_name_fragments_cache is None:
        _outlet_name_fragments_cache = {
            cfg.get('description', '').split(' — ')[0].strip().lower()
            for cfg in NEWS_SOURCES.values()
            if cfg.get('description')
        }
    return _outlet_name_fragments_cache


# ── Scraping ───────────────────────────────────────────────────────────────
# Minutes between pipeline runs. Set SCRAPE_INTERVAL in .env to change the
# cadence without touching code. 60 is real-time-enough for a news event
# tracker while giving each run a big-enough window that "found=0" isn't
# just noise from scraping between publishes (see README).
SCRAPE_INTERVAL = int(os.environ.get('SCRAPE_INTERVAL', 60))
MAX_ARTICLES_PER_SOURCE = 25
REQUEST_TIMEOUT = 15        # seconds — 10s was clipping slower sites
                             # (ft.lk observed at 2-3s+ round trip, some
                             # requests timing out at the old 10s ceiling)
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# Sources are scraped concurrently (thread pool) rather than one-at-a-time —
# discovery is I/O-bound (waiting on each outlet's server), so this is a
# straightforward wall-clock win with no change needed to any individual
# outlet's scraping method. Kept modest rather than "one thread per source"
# (21 sources) out of politeness to the sites being scraped and to leave
# headroom in the DB connection pool (ThreadedConnectionPool(1, 10, ...) in
# database.py), which every thread's save_batch() call draws from.
SCRAPE_MAX_WORKERS = 8

# Same reasoning applied to body extraction (extractor.py), which visits one
# article URL per fetch and was previously the slowest sequential step.
EXTRACT_MAX_WORKERS = 8

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# ── Embeddings ────────────────────────────────────────────────────────────
# Which backend turns text into vectors. See embedding_provider.py.
#
#   gemini  Google's hosted API. No local model, no torch — this is what
#           makes the project deployable on a small host and lets /search
#           run anywhere. ~$1.64/year at this project's measured rate.
#   local   paraphrase-multilingual-MiniLM-L12-v2 in process. Offline and
#           free, but ~1.2GB of dependencies plus a 458MB model download.
#
# The two produce INCOMPATIBLE vector spaces — every threshold below is
# tuned per-provider and must be re-derived when switching. Changing this
# also changes model_id(), which makes the next pipeline run re-embed the
# whole corpus automatically instead of mixing spaces.
EMBEDDING_PROVIDER = os.environ.get('EMBEDDING_PROVIDER', 'gemini').strip().lower()

LOCAL_EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
GEMINI_EMBEDDING_MODEL = os.environ.get('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')

# Vector width. 384 is fixed by the local model; Gemini supports 128-3072 via
# Matryoshka truncation. 768 is the default: pgvector's HNSW index refuses
# anything over 2000 dimensions (verified against this project's own Neon
# instance), so 3072 could be stored but never indexed.
EMBEDDING_DIMENSION = (
    384 if EMBEDDING_PROVIDER != 'gemini'
    else int(os.environ.get('EMBEDDING_DIMENSION', 768))
)

# Texts per batchEmbedContents call, and the pause between calls.
#
# MEASURED FREE-TIER LIMIT (from the API's own quota error):
#   quotaId    EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier
#   quotaValue 1000   per DAY, counted per *embedded item* — every text in a
#                     batch spends one unit, not one per HTTP request.
#
# What that means here:
#   * Ongoing operation is fine. This project ingests ~222 articles/day, well
#     inside 1000, with room for /search queries on top.
#   * A full backfill (currently ~3,800 articles) does NOT fit in one day. It
#     completes on its own over ~4 days because embedder.py commits
#     incrementally and each run resumes from where the quota cut it off —
#     or immediately, if billing is enabled.
# Both knobs can be relaxed once billing is on; they only pace the burst.
GEMINI_EMBED_BATCH = int(os.environ.get('GEMINI_EMBED_BATCH', 25))
GEMINI_EMBED_SLEEP = float(os.environ.get('GEMINI_EMBED_SLEEP', 1.0))

# ── NLP pipeline ──────────────────────────────────────────────────────────
BODY_BATCH_SIZE = 100
EMBED_BATCH_SIZE = int(os.environ.get('EMBED_BATCH_SIZE', 256))
# Articles embedded and committed per sub-batch. Keeps progress durable when a
# hosted provider rate-limits mid-run: whatever completed is already saved and
# the next run resumes from it, rather than the whole batch being lost.
EMBED_COMMIT_CHUNK = int(os.environ.get('EMBED_COMMIT_CHUNK', 50))
HDBSCAN_MIN_CLUSTER = 2
HDBSCAN_MIN_SAMPLES = 1

# ── Similarity thresholds ────────────────────────────────────────────────────
# EVERY constant below lives in cosine-similarity space and is therefore
# specific to EMBEDDING_PROVIDER. The two spaces are wildly different — on
# this corpus, unrelated articles score ~0.17 under the local model but
# ~0.75 under Gemini, which is ABOVE the local model's own match threshold.
# Carrying values across providers collapses every article into one cluster,
# silently and with no error.
#
# Current defaults are the GEMINI values, derived by compare_embeddings.py:
# each is the point in the Gemini distribution occupying the same percentile
# the old local value held. For reference, the local-provider equivalents
# were: MATCH 0.60, COHERENCE 0.45, MERGE 0.55, TOPIC_SIM 0.45,
# TOPIC_MATCH 0.40, SEARCH_HIGH/LOW 0.60/0.40.
#
# They are distribution-derived starting points, NOT measured optima. Run
# compare_embeddings.py after any provider change, and re-derive them
# properly once a labeled evaluation set exists.

# ── Incremental clustering ───────────────────────────────────────────────────
# Each run re-fits HDBSCAN only on articles published within this window,
# instead of the whole historical corpus — bounds run time as the database
# grows, and keeps clustering focused on live news events rather than
# resurrecting old ones. New cluster groupings are then matched back to
# existing persistent cluster_ids by centroid similarity (see clusterer.py)
# so "same event -> same cluster_id" holds across runs; only genuinely new
# groupings mint a new id. Articles outside the window keep whatever
# cluster_id they were last assigned and are not touched.
CLUSTER_WINDOW_DAYS = 14

# Cosine similarity (embeddings are L2-normalized, so this is a dot product)
# between a new run's cluster centroid and a previous cluster's centroid
# above which they're treated as the same event. Chosen conservatively above
# HDBSCAN's own cluster_selection_epsilon=0.3 member-to-member merge radius,
# since centroid-to-centroid similarity is naturally higher than the
# member-pair similarities driving that merge.
CLUSTER_MATCH_THRESHOLD = float(os.environ.get('CLUSTER_MATCH_THRESHOLD', 0.90))

# HDBSCAN's cluster_selection_epsilon is picked per run from the data, then
# clamped to this range. Euclidean distance over unit vectors, so it is
# provider-specific like every similarity constant: the local model put
# same-event pairs ~0.68 apart and used [0.15, 0.45]; Gemini compresses that
# to ~0.40, so the clamp scales down proportionally.
EPSILON_CLAMP_LOW = float(os.environ.get('EPSILON_CLAMP_LOW', 0.09))
EPSILON_CLAMP_HIGH = float(os.environ.get('EPSILON_CLAMP_HIGH', 0.26))
# MEASURED: under Gemini this knob is effectively inert. Actual pairwise
# distances in a live window start around 0.55 (p1) with a median of 0.71,
# far above any value in this range, so cluster_selection_epsilon never binds
# — sweeping 0.10 to 0.32 moved the result from 70 clusters to 68. HDBSCAN's
# density selection is doing the work. Do not spend time tuning this before
# checking it still matters; the active levers are min_cluster_size and the
# refinement thresholds.

# ── Cluster quality refinement ───────────────────────────────────────────────
# Post-processing applied to each run's fresh structural HDBSCAN groups,
# before they're matched to persistent cluster ids above. Adapted from a
# sibling project's (media-bias-detection) agglomerative-clustering
# refinement passes, which solve the same underlying problem — grouping
# same-event articles for cross-outlet comparison — with more rigor than a
# bare clustering fit provides. Thresholds below are starting points sized to
# this project's own data (different embedding model/metric/run cadence than
# that project), not copied values, and should be revisited once more runs
# have accumulated to observe actual cluster quality.

# Recursive split trigger #1: a cluster larger than this is assumed to be
# several topically-adjacent stories merged too eagerly, not one real event.
# The largest verified real event cluster so far had ~10 articles; this
# leaves headroom before splitting kicks in.
# Range: [12, 30] — lower splits more aggressively.
CLUSTER_MAX_SIZE = 20

# Recursive split trigger #2: mean cosine similarity of a cluster's members
# to their own centroid, below which the group is treated as too loose to be
# one event and is split in two.
# Range: [0.35, 0.55] — lower accepts looser groupings.
CLUSTER_MIN_COHERENCE = float(os.environ.get('CLUSTER_MIN_COHERENCE', 0.83))

# Recursion depth cap shared by both split triggers above.
CLUSTER_MAX_SPLIT_DEPTH = 3

# A cluster where one outlet supplies more than this fraction of members is
# usually that outlet's own follow-up coverage, not a genuine multi-source
# event; it's split into (that outlet's articles) vs (the rest) so a real
# cross-source story isn't hidden inside a single-outlet cluster. Only
# applied to clusters with >= 4 members — splitting a 2-3 article cluster
# this way tends to just produce noise.
CLUSTER_MAX_DOMINANT_OUTLET_SHARE = 0.6

# After splitting, a cluster left with only one outlet is checked against
# every other single-outlet cluster with no outlet overlap; the pair with
# centroid cosine similarity at or above this — provided merging wouldn't
# exceed CLUSTER_MAX_SIZE or CLUSTER_MAX_SPAN_DAYS — is merged into one
# multi-source cluster instead of being reported as two separate ones.
CLUSTER_MERGE_SIMILARITY_THRESHOLD = float(os.environ.get('CLUSTER_MERGE_SIMILARITY_THRESHOLD', 0.88))

# Hard cap on how many days a single run's *structural* cluster may span.
# This applies only to the fresh grouping computed this run, not to a
# story's overall lifetime — a persistent cluster still keeps growing across
# runs under one id via CLUSTER_MATCH_THRESHOLD above regardless of this
# value, so it doesn't fight the "same event -> same cluster_id" design.
# It exists to stop one run's fit from lumping together two textually
# similar but unrelated events from different weeks inside
# CLUSTER_WINDOW_DAYS. Deliberately looser than scaling the sibling
# project's 4-day/28-day-window ratio down to this project's 14-day window
# would suggest (~2 days) — real Sri Lankan running stories (e.g. an
# extended prison-unrest event) have been observed genuinely spanning
# several days as one story.
CLUSTER_MAX_SPAN_DAYS = 5

# ── Topics: the grouping layer above event clusters ───────────────────────
# Event clustering answers "which articles cover the same event"; topics
# answer "which events belong to the same ongoing story". Several distinct
# IMF events ("tranche approved", "conditions debated in parliament") are
# correctly *separate* clusters — the fact-checking/bias-detection consumer
# needs that separation to compare outlet coverage of one specific event —
# but a reader browsing wants them rolled up under one heading.
#
# Implemented as a second clustering pass over event-cluster centroids
# rather than by loosening the event-level thresholds above, precisely so
# event-level grouping is left untouched (see clusterer.ClusterTopicBuilder).

# Cosine similarity between event-cluster centroids above which two events
# are treated as the same broader topic. Necessarily looser than
# CLUSTER_MATCH_THRESHOLD (0.60): "same ongoing story" is a weaker bar than
# "same event". Starting point — worth re-deriving against a labeled sample
# the same way the event-level thresholds should be.
# MEASURED against centroid pairs, not article pairs — the distinction
# matters and got this wrong once. Cluster centroids are averages, so they
# sit much closer together than individual articles do: p50=0.79, p90=0.84,
# p99=0.89. A threshold of 0.83 is therefore only ~p85 and merged 168
# unrelated events into one "topic" (vehicle imports + a cooperatives act +
# coconut processing + elephants). Sweep at 663 centroids:
#     0.83 -> 66 topics, largest 168   <- was here
#     0.88 -> 105 topics, largest 28
#     0.90 -> 105 topics, largest 13, covering 324 clusters
#     0.94 -> 53 topics, largest 9, covering only 120
# 0.90 keeps topic sizes plausible while still covering ~half the clusters.
TOPIC_SIMILARITY_THRESHOLD = float(os.environ.get('TOPIC_SIMILARITY_THRESHOLD', 0.90))

# Persistent topic identity across runs, matching a fresh topic grouping to
# the previous run's topics by centroid similarity — the same mechanism
# _assign_persistent_ids uses for events, applied one level up. Looser than
# TOPIC_SIMILARITY_THRESHOLD since a topic's centroid drifts as its member
# events change, and losing topic identity is worse than holding a slightly
# drifted one.
# Scaled with TOPIC_SIMILARITY_THRESHOLD, and kept just below it for the
# same reason as at the event level: a topic's centroid drifts as its member
# events change, and losing topic identity is worse than holding a slightly
# drifted one.
TOPIC_MATCH_THRESHOLD = float(os.environ.get('TOPIC_MATCH_THRESHOLD', 0.88))

# How many event clusters must group together before it's presented as a
# topic at all. Below this, a cluster keeps topic_id=NULL — a standalone
# event, not forced into a topic it doesn't belong to.
TOPIC_MIN_CLUSTERS = 2

# ── Cluster labeling ──────────────────────────────────────────────────────
# Layered fallback for picking each cluster's headline, adapted from the
# sibling project's LLM -> most-central-title -> KeyBERT -> sanitized chain.
# The default here is the middle tier alone — this project's own existing
# headline-quality heuristic (60% centroid similarity + 40% hard-news title
# scoring) — since adding a live LLM call to every cluster on every run
# means an external API cost per event, which conflicts with this project's
# "runs indefinitely, fully local, free" design. Set CLUSTER_LABEL_USE_LLM=1
# (with a real GEMINI_API_KEY) to make the LLM tier the first choice.
# KeyBERT stays local either way, and only engages when the heuristic's own
# best pick still looks weak (see CLUSTER_LABEL_QUALITY_FLOOR) — it isn't
# used on every cluster, just the ones the heuristic can't label well.
CLUSTER_LABEL_USE_LLM = os.environ.get('CLUSTER_LABEL_USE_LLM', '').strip().lower() in ('1', 'true', 'yes')
GEMINI_LABEL_MODELS = ['gemini-flash-latest', 'gemini-2.5-flash', 'gemini-3.1-flash-lite']

# Below this _title_quality score (0-1), the heuristic's best real headline
# is considered too weak to use as-is — too short, opinion-toned, no
# hard-news signal — and KeyBERT keyphrase extraction is tried instead
# before falling back to it anyway as a last resort.
CLUSTER_LABEL_QUALITY_FLOOR = 0.3

# ── Claim retrieval (/search) ─────────────────────────────────────────────
# Answers "is this news in your database?" for the external AI-agent
# component: free-text claim in, matching event cluster_ids out. Hybrid
# retrieval — dense (pgvector cosine over article embeddings) plus lexical
# (Postgres full-text search) — fused by Reciprocal Rank Fusion, then
# aggregated from articles up to their event clusters. See searcher.py.

# Needs the embedding model in-process to vectorize the query, so it can't
# run on the lightweight read-only API image (requirements-api.txt ships no
# torch/sentence-transformers). Set SEARCH_ENABLED=false there and the route
# returns a clean 503, exactly like PIPELINE_ENABLED does for /pipeline/run.
SEARCH_ENABLED = os.environ.get('SEARCH_ENABLED', 'true').strip().lower() != 'false'

# Candidates pulled from *each* retrieval channel before fusion. Larger
# widens recall at the cost of a bigger fusion pass; 50 comfortably covers
# top_k after the article->cluster collapse (many candidates share a cluster).
SEARCH_CANDIDATES = int(os.environ.get('SEARCH_CANDIDATES', 50))

# Clusters returned by default.
SEARCH_TOP_K = int(os.environ.get('SEARCH_TOP_K', 5))

# RRF's rank-smoothing constant. 60 is the value from the original paper
# (Cormack et al. 2009) and is deliberately left at it — RRF's appeal is
# having no tuned weight to justify.
SEARCH_RRF_K = int(os.environ.get('SEARCH_RRF_K', 60))

# Two thresholds, not one, so a near-miss can be reported *as* a near-miss:
#   best >= HIGH          -> "found"      (assert the match)
#   LOW <= best < HIGH    -> "partial"    (candidates, not asserted)
#   best <  LOW           -> "not_found"
# Applied to the dense cosine similarity of the best-matching article, which
# is the interpretable, bounded [-1,1] signal — RRF scores are rank-derived
# and not comparable across queries, so they order results but must not be
# the thing a confidence cutoff is read from.
# Starting points to be calibrated against a labeled query set, exactly like
# the clustering thresholds — see the evaluation plan.
SEARCH_MIN_SCORE_HIGH = float(os.environ.get('SEARCH_MIN_SCORE_HIGH', 0.90))
SEARCH_MIN_SCORE_LOW = float(os.environ.get('SEARCH_MIN_SCORE_LOW', 0.82))

# How article-level evidence rolls up to a cluster score: 'max' (one strongly
# matching article is enough to say the event is present) or 'mean3' (mean of
# the top 3, rewarding corroboration across outlets). 'max' by default;
# 'sum' is deliberately not offered since it would just favour big clusters.
SEARCH_AGGREGATION = os.environ.get('SEARCH_AGGREGATION', 'max').strip().lower()

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = '0.0.0.0'
API_PORT = int(os.environ.get('API_PORT', 8000))

# ── Database ───────────────────────────────────────────────────────────────
DATABASE_URL = os.environ['DATABASE_URL']
