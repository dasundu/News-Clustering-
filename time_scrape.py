# time_scrape.py
"""
Time each configured source's discovery step on its own, writing results to a
file as it goes.

Written to a file rather than stdout because piping a long-running run through
`tail` buffers the output and throws away everything but the tail — twice that
hid which stage a killed run was actually in.
"""
import logging
import sys
import time

logging.disable(logging.CRITICAL)

from config import NEWS_SOURCES          # noqa: E402
from scraper import NewsScraper          # noqa: E402

OUT = 'scrape_timings.txt'


def main() -> int:
    s = NewsScraper()
    rows = []
    with open(OUT, 'w', encoding='utf-8') as fh:
        fh.write(f'{"source":<22}{"secs":>8}{"found":>7}  note\n')
        fh.flush()
        for name, cfg in NEWS_SOURCES.items():
            t = time.time()
            err = ''
            try:
                n = len(s._scrape_source(name, cfg))
            except Exception as e:
                n, err = 0, f'{type(e).__name__}: {str(e)[:60]}'
            d = time.time() - t
            rows.append((name, d, n, err))
            note = err or ('0 found' if n == 0 else '')
            if d > 20:
                note = (note + '  <-- SLOW').strip()
            line = f'{name:<22}{d:8.1f}{n:>7}  {note}\n'
            fh.write(line)
            fh.flush()          # so a killed run still leaves what it had
            print(line, end='', flush=True)

        total = sum(r[1] for r in rows)
        slow = sorted(rows, key=lambda r: -r[1])[:5]
        summary = (
            f'\n{"SEQUENTIAL TOTAL":<22}{total:8.1f}s\n'
            f'{"sources":<22}{len(rows):>8}\n'
            f'{"returned 0":<22}{sum(1 for r in rows if r[2] == 0):>8}\n'
            f'{"errored":<22}{sum(1 for r in rows if r[3]):>8}\n'
            f'\nslowest:\n' +
            ''.join(f'  {n:<22}{d:7.1f}s  {c} found  {e}\n' for n, d, c, e in slow)
        )
        fh.write(summary)
        print(summary, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
