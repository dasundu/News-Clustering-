# identity_retention.py
"""
Measures persistent-identity stability across every consecutive pair of
scheduled pipeline runs recorded in scraper.log. Read-only: parses the log,
touches nothing else.

The thesis reports 99.8% retention from two runs. The scheduler has logged
every run it made since 20 Aug 2026, so the same measurement can be repeated
over the whole history instead of one pair.

How runs are reconstructed
--------------------------
scraper.log lines carry no timestamps, but the scheduler's own lines do:

  Scheduler started                         -> a new process (session)
  Running job ... (scheduled at <time>)     -> a scheduled run begins
  Clustering done: A active clusters (C continued, N new), X noise

The first run of each session is triggered directly on start-up, not by the
scheduler, so its time is unknown and the gap since the previous run could be
weeks. It is excluded: a run after a long gap clusters a different 14-day
window and is not a test of identity between consecutive runs. A pair is kept
only if both runs are scheduled runs in the same session, one interval apart.

Two measures, both per consecutive pair (previous run P, next run Q):

  kept share    = C_Q / A_Q   share of Q's active events that kept an ID from P.
                                This is the thesis's definition (1,483 / 1,486).
  carried share = C_Q / A_P   share of P's events still present, under the
                                same ID, in Q. Lower than 100% partly because
                                events age out of the 14-day window; above
                                100% when an event absent from P (briefly
                                below the clustering threshold) returns in Q
                                under its old ID, which matching allows.

New events in Q are not failures: news arrives every hour. "Kept share"
therefore understates stability whenever real news arrived between runs.

Usage:
  python identity_retention.py                  # report on scraper.log
  python identity_retention.py --log other.log
  python identity_retention.py --csv runs.csv   # also write every pair
"""

import argparse
import csv
import re
import statistics
from datetime import datetime, timedelta

RUN_JOB = re.compile(r'Running job "NewsScraperScheduler\._run \(trigger: interval\[(\d+):(\d\d):(\d\d)\].*?'
                     r'\(scheduled at (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
DONE = re.compile(r'Clustering done: (\d+) active clusters \((\d+) continued, (\d+) new\), (\d+) noise')
PROVIDER = re.compile(r'Embedding provider: (\S+)')


def parse_runs(path):
    runs, session, pending, provider = [], 0, None, None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'Scheduler started' in line:
                session += 1
                pending = None          # next run is the unscheduled start-up run
                continue
            m = PROVIDER.search(line)
            if m:
                provider = m.group(1)
                continue
            m = RUN_JOB.search(line)
            if m:
                h, mi, s, when = m.groups()
                pending = (datetime.strptime(when, '%Y-%m-%d %H:%M:%S'),
                           timedelta(hours=int(h), minutes=int(mi), seconds=int(s)))
                continue
            m = DONE.search(line)
            if m:
                active, cont, new, noise = map(int, m.groups())
                when, interval = pending if pending else (None, None)
                runs.append({'session': session, 'time': when, 'interval': interval,
                             'active': active, 'continued': cont, 'new': new,
                             'noise': noise, 'provider': provider})
                pending = None
    return runs


def consecutive_pairs(runs):
    pairs = []
    for p, q in zip(runs, runs[1:]):
        if p['session'] != q['session'] or p['time'] is None or q['time'] is None:
            continue
        gap = q['time'] - p['time']
        if q['interval'] is None or not (0.5 * q['interval'] <= gap <= 1.5 * q['interval']):
            continue
        pairs.append({'time': q['time'], 'gap_min': gap.total_seconds() / 60,
                      'prev_active': p['active'], 'active': q['active'],
                      'continued': q['continued'], 'new': q['new'], 'noise': q['noise'],
                      'provider': q['provider'],
                      'kept': q['continued'] / q['active'] if q['active'] else None,
                      'carried': q['continued'] / p['active'] if p['active'] else None})
    return pairs


def summarise(label, pairs):
    kept = [p['kept'] for p in pairs if p['kept'] is not None]
    carried = [p['carried'] for p in pairs if p['carried'] is not None]
    if not kept:
        print(f'\n  {label}: no consecutive pairs')
        return
    first, last = min(p['time'] for p in pairs), max(p['time'] for p in pairs)
    print(f'\n  {label}')
    print(f'    consecutive pairs     {len(kept)}   ({first:%d %b} to {last:%d %b %Y})')
    print(f'    kept share   mean {statistics.mean(kept):7.1%}   median {statistics.median(kept):7.1%}'
          f'   min {min(kept):7.1%}')
    print(f'    carried      mean {statistics.mean(carried):7.1%}   median {statistics.median(carried):7.1%}'
          f'   min {min(carried):7.1%}')
    bands = [(0.99, '>= 99%'), (0.95, '95-99%'), (0.90, '90-95%'), (0.0, '< 90%')]
    counts, upper = [], 1.01
    for floor, name in bands:
        counts.append(f'{name}: {sum(floor <= k < upper for k in kept)}')
        upper = floor
    print(f'    kept share by band    ' + '   '.join(counts))
    print(f'    pairs meeting the 95% target: {sum(k >= 0.95 for k in kept)}/{len(kept)}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--log', default='scraper.log')
    ap.add_argument('--csv')
    args = ap.parse_args()

    runs = parse_runs(args.log)
    pairs = consecutive_pairs(runs)
    print(f'  {len(runs)} runs in {args.log}, {max((r["session"] for r in runs), default=0)} sessions, '
          f'{len(pairs)} consecutive scheduled pairs')

    summarise('All consecutive pairs', pairs)
    current = [p for p in pairs if (p['provider'] or '').startswith('gemini') and p['noise'] == 0]
    summarise('Current configuration (Gemini embeddings, single-outlet promotion)', current)

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
        print(f'\n  wrote {len(pairs)} pairs to {args.csv}')


if __name__ == '__main__':
    main()
