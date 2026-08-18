# main.py
"""
Entry point for the Sri Lankan News Clustering system.

Usage:
  python main.py          # run scrape+embed+cluster pipeline every SCRAPE_INTERVAL min
  python main.py --api    # start FastAPI server on port 8000
"""

import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Force UTF-8 console output so pipeline banners (box-drawing / ✔ chars)
# don't crash on Windows terminals that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

# Load .env before anything else so HF_TOKEN and DATABASE_URL are available
from config import _load_env
_load_env()
if os.environ.get('HF_TOKEN'):
    os.environ.setdefault('HUGGING_FACE_HUB_TOKEN', os.environ['HF_TOKEN'])

# Suppress sentence-transformers / huggingface direct prints and progress bars
os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

# Only show log messages from our own modules
_OWN_MODULES = {
    '__main__', 'pipeline', 'scraper', 'extractor',
    'cleaner', 'embedder', 'clusterer', 'database', 'api', 'scheduler',
}


class _OwnModulesOnly(logging.Filter):
    def filter(self, record):
        return record.name.split('.')[0] in _OWN_MODULES


_file_handler = logging.FileHandler('scraper.log')
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.addFilter(_OwnModulesOnly())  # console: only our modules

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s  %(name)s  %(message)s',
    handlers=[_stream_handler, _file_handler],
)

logger = logging.getLogger(__name__)


def run_pipeline_loop():
    from scheduler import run_scheduler
    from config import SCRAPE_INTERVAL
    print()
    print('  SL News Clustering — starting scheduler')
    print(f'  Pipeline runs every {SCRAPE_INTERVAL} min  |  Ctrl+C to stop')
    print('  API:  python main.py --api  ->  http://localhost:8000')
    print()
    try:
        run_scheduler(interval_minutes=SCRAPE_INTERVAL)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


def run_api():
    import uvicorn
    from config import API_HOST, API_PORT
    print()
    print(f'  Starting API server   http://{API_HOST}:{API_PORT}')
    print(f'  Swagger docs          http://localhost:{API_PORT}/docs')
    print(f'  Frontend (run separately, see frontend/)  http://localhost:3000')
    print()
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False, log_level="warning")


if __name__ == '__main__':
    if '--api' in sys.argv:
        run_api()
    else:
        run_pipeline_loop()
