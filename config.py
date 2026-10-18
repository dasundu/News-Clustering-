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


# Default ceiling on how many articles one source contributes per run. A
# source can lower it with a `max_articles` key — see the four capped below.
SOURCE_ARTICLE_CAP = int(os.environ.get('SOURCE_ARTICLE_CAP', 50))

# ── News sources ───────────────────────────────────────────────────────────
# filter_keywords=False  →  all articles from this source are kept as-is
# filter_keywords=True   →  only articles matching ECONOMICS/POLITICS keywords
#                            are kept; others are discarded
#
# NOW FALSE EVERYWHERE. The gate tested each HEADLINE independently, and
# outlets word the same event differently — "Fuel prices revised" carries no
# keyword while "...to ease inflation" does. Measured against the existing
# multi-outlet clusters, applying the gate to all sources would have stripped
# outlets from 29% of them and pushed 24% below two outlets entirely, which
# is the difference between an event being usable for cross-outlet
# comparison and not. It also throttled the largest general outlets to
# almost nothing (Newsfirst: 60 articles, against 839 from one business
# site). articles.category still records economics/politics/general, so the
# distinction is kept where it belongs — as a label, not as a gate.

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
        # No new article since 30 Aug 2026. Checked 18 Oct 2026: every page,
        # the feed and the WordPress API return Cloudflare 520, while
        # robots.txt (served from Cloudflare's cache) loads, so the origin is
        # not serving pages to this scraper. Skipped instead of retried every
        # run; its 683 stored articles stay in their clusters. Remove this key
        # to re-enable once the site answers again.
        'blocked': 'Cloudflare 520 on every page since 30 Aug 2026',
    },
    'lankabusinessonline': {
        'max_articles': 12,
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
        'max_articles': 12,
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
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'Ada Derana — general breaking news',
    },
    'newsfirst': {
        'url': 'https://english.newsfirst.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'News First — general 24/7 news',
    },
    'daily_mirror': {
        'url': 'https://www.dailymirror.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'Daily Mirror — general news',
    },
    'the_morning': {
        'url': 'https://www.themorning.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'The Morning — general news',
    },
    'daily_news': {
        'url': 'https://www.dailynews.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Daily News — state-owned general news',
    },
    'hirunews': {
        'url': 'https://www.hirunews.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'HIGH',
        'description': 'Hiru News — general news',
    },
    'the_island': {
        'url': 'https://island.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'The Island — general news',
    },
    'sunday_observer': {
        'url': 'https://www.sundayobserver.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Sunday Observer — general news',
    },
    'colombo_gazette': {
        'url': 'https://colombogazette.com',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Colombo Gazette — general news',
    },
    'ceylon_today': {
        'max_articles': 12,
        'url': 'https://ceylontoday.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Ceylon Today — general news',
    },
    'sunday_times': {
        'url': 'https://sundaytimes.lk',
        'rss': None,
        'type': 'html',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Sunday Times Sri Lanka — general news',
    },

    # ── Newly researched sources (added Aug 2026) ──────────────────────────
    # All four verified live: valid RSS/WP-API feed, real recent Sri Lankan
    # articles, and a permissive robots.txt (general User-agent: * is
    # Allow: /; the sites' Disallow rules target named AI-training bots
    # specifically, not general crawling — see README for the full check).
    'onlanka': {
        'max_articles': 12,
        'url': 'https://www.onlanka.com',
        'rss': 'https://www.onlanka.com/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Onlanka — Sri Lanka news and information portal',
    },
    'asianmirror': {
        'max_articles': 12,
        'url': 'https://asianmirror.lk',
        'rss': 'https://www.asianmirror.lk/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'Asian Mirror — politics, international relations and business',
    },
    'lankanewsweb': {
        'max_articles': 12,
        'url': 'https://lankanewsweb.net',
        'rss': 'https://www.lankanewsweb.net/feed',
        'type': 'rss',
        'category': 'general',
        'filter_keywords': False,
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
        'filter_keywords': False,
        'priority': 'MEDIUM',
        'description': 'News.lk — government and policy news',
        # Never stored a single article. Checked 18 Oct 2026: the site sits
        # behind a Sucuri firewall that answers scripted clients with a
        # JavaScript redirect challenge (HTTP 307), and /feed and /rss.xml are
        # 404. Getting past it needs a headless browser, which this component
        # deliberately avoids (see the thesis, section 7.5). Kept configured so
        # the outlet list stays complete, but skipped rather than retried.
        'blocked': 'Sucuri firewall challenge',
    },

    # ── Sinhala aggregator with English translations ───────────────────────
    # Article pages are client-rendered; scraped via the site's JSON endpoint.
    # Headlines and bodies are taken from the English fields it returns.
    'helakuru': {
        'url': 'https://www.helakuru.lk',
        'rss': None,
        'type': 'json',
        'category': 'general',
        'filter_keywords': False,
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
# 120 rather than 60. Two reasons, both measured: Sri Lankan outlets do not
# publish fast enough for an hourly cycle to find much, and every run spends
# a few Gemini requests on topic maintenance against a 500/day free-tier
# limit. Halving the cadence halves that fixed cost.
SCRAPE_INTERVAL = int(os.environ.get('SCRAPE_INTERVAL', 120))

# A source that stores no new article for this many days is flagged in the
# end-of-run report. EconomyNext started answering every page with Cloudflare
# 520 on 30 Aug 2026 and contributed nothing for seven weeks before anyone
# noticed: each run printed "0 found" for that run alone, with nothing to say
# the silence was weeks long. Three days is well past any outlet's quietest
# normal gap, so the warning means a source is broken, not slow.
STALE_SOURCE_DAYS = int(os.environ.get('STALE_SOURCE_DAYS', 3))
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
# 200, raised from 100 when general news stopped being discarded at scrape
# time. A run started bringing in ~430 new articles instead of ~20, so a
# 100-article extraction cap turned a one-off catch-up into a permanent
# backlog. Extraction is network-bound and already runs EXTRACT_MAX_WORKERS
# in parallel, so doubling it lengthens a run far less than proportionally.
#
# Note the ceiling this does NOT lift: embedding quota is charged per
# ARTICLE (~1,000/day on the free tier), so extracting faster than that just
# moves the queue from "waiting for body" to "waiting for embedding".
BODY_BATCH_SIZE = 200
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

# Give every article HDBSCAN leaves as noise its own single-article cluster,
# instead of leaving it outside the hierarchy entirely.
#
# A story only one outlet carried is still an event — it just has one
# source. Leaving those articles at cluster_id=-1 meant 41% of the corpus
# (1,140 of 3,481 articles) could never appear under a topic at all, since
# topic membership is carried by cluster_labels.topic_id and they had no
# cluster row to carry it.
#
# Chosen over adding articles.topic_id specifically because it needs no
# schema, API or frontend change: every article still reaches its topic
# through exactly one cluster, so the article -> cluster -> topic tree stays
# a tree. The cost is that the cluster count roughly triples, which is why
# labelling now fetches members in one query rather than one per cluster.
CLUSTER_PROMOTE_SOLO = os.environ.get(
    'CLUSTER_PROMOTE_SOLO', 'true').strip().lower() not in ('0', 'false', 'no')

# ── Evicting absorbed outliers ────────────────────────────────────────────
# OFF. Measured on live data: it removes about as many correct articles as
# wrong ones, and no threshold separates them:
#
#     0.939  "China Jails Evergrande Boss for Life"    SAME story, evicted wrongly
#     0.934  "PickMe launches Colombo Connect"         different story, correctly out
#
# 0.005 apart, opposite correct answers. A margin sweep confirmed it: from
# 0.05 to 0.15 the wrong removals and the right ones rise and fall together
# (3/4 and 3/4, then 1/4 and 1/4). There is no setting that keeps one.
#
# The cause is that near-duplicate wire copy inside a cluster pushes the
# median link to ~1.0, lifting the bar high enough to evict a legitimate
# paraphrase of the same event.
#
# Off rather than merely loosened, because the two failure modes cost
# different amounts. An evicted article becomes its own solo cluster, so a
# wrong eviction costs the event an OUTLET — directly shrinking the
# multi-outlet clusters this corpus exists to produce. A wrong keep costs one
# odd headline in a list. Keeping is the cheaper mistake.
#
# The band where arithmetic provably cannot decide (0.86-0.95) holds 41% of
# clustered articles, so the honest fix is an LLM judge over that band, the
# same shape as the topic join/merge checks. Deliberately not done: it would
# put a non-deterministic external call inside EVENT clustering, which today
# is fully reproducible and works whether or not any API is reachable.
# BUILT, MEASURED, AND TURNED OFF. An LLM judge over the borderline band was
# implemented and run against live data. It scored 6/7 on hand-picked cases,
# but on the corpus it failed the only test that matters:
#
#     no eviction        602 clusters with 2+ outlets, 221 with 3+
#     LLM, evicted 142   588 with 2+, 206 with 3+     <- WORSE than doing nothing
#
# Evicting cost 30 multi-outlet events, and the cluster that motivated the
# whole feature (an SLPP article inside a cluster of NPP rally reports) still
# contained it afterwards. It also proved sensitive to which sibling
# headlines it was shown: the same 331 cases produced 0 evictions with peers
# picked by index order and 142 with peers picked by centrality.
#
# Kept as a flag rather than deleted so the experiment can be resumed, but
# the honest conclusion is that "which article belongs to which event" is not
# reliably decidable from headlines alone — by arithmetic OR by an LLM — and
# leaving a wrong article in a cluster is cheaper than removing a right one.
CLUSTER_EVICT_OUTLIERS = os.environ.get(
    'CLUSTER_EVICT_OUTLIERS', 'false').strip().lower() in ('1', 'true', 'yes')

# The band where similarity provably cannot decide, so an LLM is asked
# instead. Outside it the decision is made without a request:
#     >= HIGH  keep    (near-duplicate wire copy; no doubt)
#     <  LOW   evict   (cricket in an economics cluster; no doubt)
# Measured across 1,403 clustered articles: 766 sit above HIGH, 62 below LOW,
# and 575 — 41% — fall in between. That share is the honest reason this is a
# judgement call rather than a threshold.
CLUSTER_EVICT_LLM_LOW = float(os.environ.get('CLUSTER_EVICT_LLM_LOW', 0.86))
CLUSTER_EVICT_LLM_HIGH = float(os.environ.get('CLUSTER_EVICT_LLM_HIGH', 0.95))
CLUSTER_EVICT_LLM_BATCH = 40


# How far below its cluster's median a member's STRONGEST link to any sibling
# may fall before it is evicted. Relative, not absolute: measured across 383
# clusters, the strongest-link distribution runs p5=0.857 to p50=0.949, so a
# fixed cut either spares real outliers in tight clusters or guts loose ones.
#
# 0.05 evicts 11% of members in clusters of 3+, and on the cluster that
# prompted this it removed exactly the two wrong articles (0.869 and 0.907)
# while keeping all four correct ones (0.949-0.975).
CLUSTER_OUTLIER_MARGIN = float(os.environ.get('CLUSTER_OUTLIER_MARGIN', 0.05))

# Absolute backstop, so a cluster whose members are ALL weakly linked (a
# grab-bag rather than an event) still sheds them. One real example held a
# javelin throw, a 400m race and the London Marathon.
CLUSTER_OUTLIER_FLOOR = float(os.environ.get('CLUSTER_OUTLIER_FLOOR', 0.86))

# ── Joining a topic vs forming one ────────────────────────────────────────
# Forming a topic and joining an existing one are different problems and get
# different rules. Forming compares two centroids with no anchor between
# them, so it must be strict. Joining compares one cluster against a topic
# centroid already averaged over several events — a much better-defined
# target — so it can admit clusters complete linkage excluded for failing
# against a single member.
#
# MEASURED, and the first attempt at this was wrong: a *looser cosine* alone
# does not work at all under Gemini embeddings. Unrelated Sri Lankan business
# news sits around 0.92, so the whole leftover distribution is compressed
# into 0.83-0.92 (p50=0.890) and no cut separates good from bad. At 0.925 it
# accepted "CSE & Sampath Bank MoU" -> "People's Leasing partners Toyota"
# while rejecting "SLT-MOBITEL entertainment" -> "SLT-MOBITEL partners vivo"
# at 0.920. Cosine there is measuring domain and register, not subject.
#
# The fix is a second, independent signal that is not another similarity
# number. Cosine survives only as a cheap pre-filter that bounds how many
# candidates reach the expensive check; the actual decision is Gemini's.

# Cosine to a topic centroid below which a leftover cluster is not considered
# at all. Held at TOPIC_SIMILARITY_THRESHOLD deliberately: matching a
# centroid at 0.90 is already looser than complete linkage's "0.90 against
# every member", so this widens the rule without lowering the bar.
TOPIC_JOIN_SIMILARITY = float(os.environ.get('TOPIC_JOIN_SIMILARITY', 0.90))

# Strongest shared term between the candidate's headline and any member's,
# scored by IDF over all cluster headlines. REMOVED — measured and it does
# not work. Rarity is not specificity: across 858 cluster headlines
# "branch" (df=4) scores 0.849 while "treasury" (df=13) scores 0.679, so
# every cut that drops "branch", "strategic" and "2025" drops "treasury"
# and "iran" first. Good and bad shared terms overlap completely. Kept as a
# comment because it is a genuinely tempting idea that has to be measured
# to be rejected.
#
# Gemini decides instead. The project already calls it to *name* topics, so
# asking it to check membership adds no new dependency, and it is the one
# component that can tell "another bank's press release" from "the same
# story". Batched, so ~90 candidates cost 4-5 calls per run rather than 90.
# Failure is closed: if the API is unavailable, nothing joins and coverage
# falls back to the strict core rather than admitting unverified members.
TOPIC_JOIN_LLM_BATCH = 40

# Hard cap on join candidates per run, so the request cost of this stage is
# bounded no matter how large the corpus gets. Clusters touched by THIS run
# are considered first — a cluster that has just changed is the one whose
# topic is most likely wrong — and the rest are ordered by similarity.
# Anything not reached is reconsidered next run, so coverage converges over
# a few cycles instead of re-asking the same settled questions every time.
#
# The budget this protects: the free tier allows 500 requests/day. At a
# 2-hour cadence that is 12 runs, so a run can afford roughly 40 requests.
# Before this cap the join stage alone wanted 18 of them every single run,
# most spent re-confirming clusters that had not changed.
TOPIC_JOIN_MAX_CANDIDATES = 80

# Only the few nearest topics are worth testing; beyond that the cosine gate
# rejects everything anyway.
TOPIC_JOIN_CANDIDATES = 3

# How many event clusters must group together before it's presented as a
# topic at all. Below this, a cluster keeps topic_id=NULL — a standalone
# event, not forced into a topic it doesn't belong to.
TOPIC_MIN_CLUSTERS = 2

# ── Merging over-split topics ─────────────────────────────────────────────
# Complete linkage at TOPIC_SIMILARITY_THRESHOLD is strict on purpose, and
# strictness splits one story into two topics. Measured after the first full
# run with LLM-validated joins:
#     11 events  "Judicial Retirement Amendment Controversy"
#      6 events  "Judicial Retirement Age Debate"
#     25 events  "Sri Lanka Economic Recovery"
#      6 events  "Sri Lanka Economic Outlook"
# Each pair is one story. Lowering the formation threshold to fix this is
# the move already measured and rejected — at 0.83 it merged 168 unrelated
# events into a single topic. So formation stays strict, and topic pairs
# just below it are put to the same judge that validates joins.

# Cosine between topic centroids below which a pair is not even considered
# for merging. HIGH, and measured: topic centroids are averages of averages,
# so they sit even closer together than cluster centroids do. Across 158
# topics, 4,280 pairs clear 0.85 and 691 still clear 0.90 — a low floor here
# does not narrow anything, it just fills the budget below with noise. Real
# duplicates measured 0.956-0.971 ("Sri Lanka Prison Unrest" / "Sri Lanka
# Prison Crisis", and two topics both titled "Business Today Rankings").
TOPIC_MERGE_FLOOR = float(os.environ.get('TOPIC_MERGE_FLOOR', 0.94))

# Hard cap on how many pairs reach the LLM in one run, highest similarity
# first. Bounds cost and run time; anything not reached this run is
# reconsidered next run, so merging converges over a few cycles rather than
# trying to fix everything at once.
TOPIC_MERGE_MAX_PAIRS = 40

# A merge is refused if it would take a topic past this many events. Merging
# is transitive through union-find, so an unguarded chain of individually
# reasonable merges (A~B, B~C, C~D) can collapse into one oversized topic
# where the ends have nothing to do with each other — the same chaining
# failure that ruled out average linkage at the formation stage.
TOPIC_MERGE_MAX_SIZE = 30

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

# Topic labels are a separate decision from event labels, and default ON.
#
# An event label can be a real headline lifted from a member article — that
# reads correctly. A *topic* spans several different events, so naming it
# after one of them is actively misleading: a 10-event topic covering SME
# support, board governance, a trade delegation and an investment summit was
# titled "Intex - InMac Sri Lanka 2026 opens at BMICH".
#
# The extractive tiers cannot fix this. KeyBERT needs the local model (gone
# under the Gemini provider), and a shared-phrase heuristic only works when
# the members happen to share wording — measured across real topics it beat
# the current label about half the time and was worse the other half.
# The LLM produces proper topic names ("Sri Lanka Prison Crisis",
# "Awards and Recognition") and is cheap here because only topics whose
# membership actually changed get relabelled — a handful per run, not all 48.
TOPIC_LABEL_USE_LLM = os.environ.get('TOPIC_LABEL_USE_LLM', 'true').strip().lower() not in ('0', 'false', 'no')
# How many topics are named in a single request. Free-tier quota is counted
# per REQUEST, not per topic, so this is the difference between a run that
# fits in the daily budget and one that does not.
TOPIC_LABEL_BATCH = 15

# Ordered by free-tier daily request quota, highest first — MEASURED against
# the live API, because the order matters far more than it looks:
#     gemini-3.1-flash-lite   500 requests/day
#     gemini-flash-latest      20 requests/day
# The previous order tried the 20/day model first and exhausted it almost
# immediately, after which every later stage in the run fell back to its
# fail-closed path. `gemini-2.5-flash` was also in this list and now returns
# 404, wasting a round trip on every single call.
#
# `gemini-flash-latest` is now dropped for the same reason, 29 Aug 2026.
# Measured across four calls it returned 503 "experiencing high demand" every
# time, taking 5.8s, 28.5s, 38.5s and 55.8s to say so. Every caller below
# walks this list and falls through to the next model on BOTH an HTTP error
# and an unparseable reply — and a reply carrying 14 names where 15 were
# asked for is an ordinary, frequent outcome. So each stumble on the fast
# model was paying ~30s for a dead one, at six call sites, on every run. That
# was the single largest cost in the pipeline: roughly 33 of ~40 minutes,
# against ~7 minutes for all the actual scraping, extraction, embedding and
# clustering combined.
#
# A one-model list is not a downgrade. At 20 requests/day the fallback could
# only ever have served one run in twenty, and every LLM stage already fails
# closed — a topic with no new name keeps the good one it has and retries on
# the next run. Restore a second entry only after timing it against the live
# API; a model that is merely *present* costs real minutes when it is down.
GEMINI_LABEL_MODELS = ['gemini-3.1-flash-lite']

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

# ── corroboration between the two retrieval channels ──────────────────────
# The bands above read confidence off the dense cosine ALONE, which quietly
# wasted the lexical channel: it could reorder results but never rescue a
# verdict. That is backwards for exactly the queries hybrid retrieval exists
# to serve. Measured on the live corpus, the query "Bus Lalith" ranked
# cluster 3435 FIRST in both channels, and was still reported not_found
# because its cosine was 0.8104 — sixteen thousandths under the cutoff.
#
# The cause is structural, not a mis-set number. A two-word proper noun is
# embedded as a very short text and compared against article vectors built
# from a title plus 500 characters of body, so it cannot reach the scores a
# full sentence does: "Sri Lanka gave money to Nepal after the floods"
# measured 0.9379 against the same corpus. Short entity queries live in a
# lower band than claim-shaped ones, permanently.
#
# So agreement between the channels is treated as evidence in its own right.
# A cluster ranked inside the top SEARCH_CORROBORATION_RANK by BOTH the
# vector and the keyword channel has been found twice, by methods with
# opposite blind spots, and its verdict shifts one band down the ladder.
#
# Lowering SEARCH_MIN_SCORE_LOW instead would have been the wrong fix:
# unrelated articles score ~0.75 under this provider, so a cutoff near 0.78
# applied to EVERY query sits almost on the noise floor. Applied only to
# doubly-confirmed matches, it is safe.
SEARCH_CORROBORATION_RANK = int(os.environ.get('SEARCH_CORROBORATION_RANK', 5))

# The floor for a corroborated match. Above the ~0.75 noise floor, below
# SEARCH_MIN_SCORE_LOW; never applied to a cluster only one channel found.
SEARCH_MIN_SCORE_FLOOR = float(os.environ.get('SEARCH_MIN_SCORE_FLOOR', 0.78))

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
