# pipeline.py
"""
Main end-to-end pipeline. Run once manually or called by scheduler.

Flow:
  1. Scrape    — fetch article titles + URLs from 7 news sites
  2. Extract   — visit each article URL, get full body text
  3. Clean     — normalize text, strip noise
  4. Embed     — BERT-family sentence vectors (384-dim)
  5. Cluster   — group articles by news event (HDBSCAN)
"""

import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

W = 60  # output width


def _fmt(seconds: float) -> str:
    """62.4 -> '1m 02s'; 9.7 -> '9.7s'."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    return f'{int(seconds // 60)}m {int(seconds % 60):02d}s'


def _line(char='-'):
    print('  ' + char * (W - 2))


def _report_stale_sources():
    """
    Flags configured sources that have stored nothing new for
    STALE_SOURCE_DAYS. A single run's "0 found" cannot tell a quiet hour from
    an outlet that has been unreachable for weeks; this can. Advisory only:
    a failure here must never stop the pipeline.
    """
    from config import NEWS_SOURCES, STALE_SOURCE_DAYS
    from database import ArticleDatabase
    try:
        last_seen = ArticleDatabase().get_source_last_seen()
    except Exception as e:
        logger.warning(f"Stale-source check skipped: {e}")
        return
    stale = []
    for name, cfg in NEWS_SOURCES.items():
        if cfg.get('blocked'):
            continue
        days = last_seen.get(name)
        if days is None:
            stale.append((name, 'has never stored an article'))
        elif days >= STALE_SOURCE_DAYS:
            stale.append((name, f'no new article for {days:.0f} days'))
    for name, why in sorted(stale):
        logger.warning(f"Stale source: {name} {why}. Check whether it is down or blocking us.")


def _header(run_num=None):
    print('=' * W)
    ts = datetime.now().strftime('%d %b %Y  %H:%M:%S')
    tag = f'  Run #{run_num}  |  ' if run_num else '  '
    print(f'  SL NEWS CLUSTERING  |  {tag}{ts}')
    print('=' * W)


def run_pipeline(run_num=None):
    _header(run_num)

    # Per-stage wall clock. Kept because a single stage can quietly swallow
    # the whole run: scraping once took 40 minutes and never reached stage 2,
    # and a total printed here is the only place that would have shown it.
    # Cheap enough to leave on permanently.
    started = time.time()
    stage_times = []
    mark = started

    # ── Step 1: Scrape ──────────────────────────────────────
    print('\n  [1/5] Scraping news sites ...')
    from scraper import NewsScraper
    scraper = NewsScraper()
    results = scraper.scrape_all_sources()
    custom = scraper.scrape_custom_sources()

    # Merge custom source counts into results
    results['total_found'] += custom['total_found']
    results['total_new'] += custom['total_new']
    results['by_source'].update(custom['by_source'])

    for source, s in results['by_source'].items():
        status = f"found={s['found']:<4}  new={s['new']}"
        if s.get('blocked'):
            flag = f"  [blocked: {s['blocked']}]"
        else:
            flag = '  [no articles]' if s['found'] == 0 else ''
        print(f"        {source:<28} {status}{flag}")
    _line()
    total_status = f"found={results['total_found']:<4}  new={results['total_new']}"
    print(f"        {'Total':<28} {total_status}")
    _report_stale_sources()
    stage_times.append(('1 Scrape', time.time() - mark)); mark = time.time()

    # ── Step 2: Extract body ────────────────────────────────
    print('\n  [2/5] Extracting article bodies ...')
    from extractor import BodyExtractor
    extracted = BodyExtractor().run()
    print(f"        {extracted} bodies fetched")
    stage_times.append(('2 Extract', time.time() - mark)); mark = time.time()

    # ── Step 3: Clean text ──────────────────────────────────
    print('\n  [3/5] Cleaning text ...')
    from cleaner import TextCleaner
    cleaned = TextCleaner().run()
    print(f"        {cleaned} articles cleaned")
    stage_times.append(('3 Clean', time.time() - mark)); mark = time.time()

    # ── Step 4: Embed ───────────────────────────────────────
    # Read from the active provider rather than hardcoded — this banner kept
    # claiming the local model long after embeddings moved to the Gemini API.
    import embedding_provider
    print(f'\n  [4/5] Embedding  ({embedding_provider.model_id()}) ...')
    from embedder import ArticleEmbedder
    embedded = ArticleEmbedder().run()
    print(f"        {embedded} articles embedded")
    stage_times.append(('4 Embed', time.time() - mark)); mark = time.time()

    # ── Step 5: Cluster ─────────────────────────────────────
    print('\n  [5/5] Clustering by news event  (HDBSCAN) ...')
    from clusterer import ArticleClusterer
    result = ArticleClusterer().run()
    print(
        f"        {result['clusters']} clusters  |  "
        f"{result['total'] - result['noise']} clustered  |  "
        f"{result['noise']} noise"
    )
    if result.get('topics'):
        print(
            f"        {result['topics']} topics  |  "
            f"{result['continued_topics']} continued  |  "
            f"{result['new_topics']} new"
        )

    # ── Summary ─────────────────────────────────────────────
    stage_times.append(('5 Cluster', time.time() - mark))

    elapsed = time.time() - started
    print()
    print('=' * W)
    from database import ArticleDatabase
    stats = ArticleDatabase().get_cluster_stats()
    finished = datetime.now()
    print("  PIPELINE FINISHED")
    print(f"  Finished at    : {finished.strftime('%d %b %Y  %H:%M:%S')}")
    print(f"  Took           : {_fmt(elapsed)}")
    _line()
    for name, secs in stage_times:
        share = (100 * secs / elapsed) if elapsed else 0
        bar = '#' * max(1, round(share / 4))
        print(f"    {name:<11} {_fmt(secs):>8}  {share:5.1f}%  {bar}")
    _line()
    print(
        f"  Total articles : {stats['total_articles']}   "
        f"Clustered : {stats['clustered']}   "
        f"Clusters : {stats['total_clusters']}"
    )
    print('=' * W)


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    run_pipeline()
