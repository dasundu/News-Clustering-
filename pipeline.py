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
from datetime import datetime

logger = logging.getLogger(__name__)

W = 60  # output width


def _line(char='-'):
    print('  ' + char * (W - 2))


def _header(run_num=None):
    print('=' * W)
    ts = datetime.now().strftime('%d %b %Y  %H:%M:%S')
    tag = f'  Run #{run_num}  |  ' if run_num else '  '
    print(f'  SL NEWS CLUSTERING  |  {tag}{ts}')
    print('=' * W)


def run_pipeline(run_num=None):
    _header(run_num)

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
        flag = '  [no articles]' if s['found'] == 0 else ''
        print(f"        {source:<28} {status}{flag}")
    _line()
    total_status = f"found={results['total_found']:<4}  new={results['total_new']}"
    print(f"        {'Total':<28} {total_status}")

    # ── Step 2: Extract body ────────────────────────────────
    print('\n  [2/5] Extracting article bodies ...')
    from extractor import BodyExtractor
    extracted = BodyExtractor().run()
    print(f"        {extracted} bodies fetched")

    # ── Step 3: Clean text ──────────────────────────────────
    print('\n  [3/5] Cleaning text ...')
    from cleaner import TextCleaner
    cleaned = TextCleaner().run()
    print(f"        {cleaned} articles cleaned")

    # ── Step 4: Embed ───────────────────────────────────────
    # Read from the active provider rather than hardcoded — this banner kept
    # claiming the local model long after embeddings moved to the Gemini API.
    import embedding_provider
    print(f'\n  [4/5] Embedding  ({embedding_provider.model_id()}) ...')
    from embedder import ArticleEmbedder
    embedded = ArticleEmbedder().run()
    print(f"        {embedded} articles embedded")

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
    print()
    print('=' * W)
    from database import ArticleDatabase
    stats = ArticleDatabase().get_cluster_stats()
    ts = datetime.now().strftime('%d %b %Y  %H:%M:%S')
    print(f"  DONE  |  {ts}")
    print(
        f"  Total articles : {stats['total_articles']}   "
        f"Clustered : {stats['clustered']}   "
        f"Clusters : {stats['total_clusters']}"
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    run_pipeline()
