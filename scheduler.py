# scheduler.py
"""
Task scheduler — runs the full pipeline every SCRAPE_INTERVAL minutes (config.py).
"""

import logging
import time
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from pipeline import run_pipeline
from config import SCRAPE_INTERVAL

logger = logging.getLogger(__name__)


class NewsScraperScheduler:
    def __init__(self, interval_minutes=SCRAPE_INTERVAL):
        self.scheduler = BackgroundScheduler()
        self.interval_minutes = interval_minutes
        self.run_count = 0

    def start(self):
        self.scheduler.add_job(
            self._run,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id='news_pipeline',
            replace_existing=True,
        )
        self.scheduler.start()
        self._run()  # run immediately on startup

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()

    def _run(self):
        self.run_count += 1
        try:
            run_pipeline(run_num=self.run_count)
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

        next_run = datetime.now() + timedelta(minutes=self.interval_minutes)
        print(f"  Next run at  {next_run.strftime('%d %b %Y  %H:%M:%S')}")
        print('=' * 60)
        print()


def run_scheduler(interval_minutes=SCRAPE_INTERVAL):
    scheduler = NewsScraperScheduler(interval_minutes)
    try:
        scheduler.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\n  Stopping scheduler...')
        scheduler.stop()
        print('  Shutdown complete.')
