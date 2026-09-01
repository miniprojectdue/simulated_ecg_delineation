#!/usr/bin/env python3
"""
cache_signals.py  -  convert every raw MedalCare-XL CSV named by the unit tables into a
float32 .npy of shape (12, 5000) under ml_modelling/data/signal_cache/.

Reading a 12x5000 text CSV costs roughly 40 ms. Over 185,022 units that is repeated tens of
thousands of times per epoch, so the cache makes training feasible. 
The cached arrays are memory mapped by the dataset, so the resident set stays
small no matter how many workers are running.

Usage
    python ml_modelling/scripts/cache_signals.py
    python ml_modelling/scripts/cache_signals.py --units ml_modelling/data/finetune_units.csv
    python ml_modelling/scripts/cache_signals.py --workers 8 --limit 100 --verify

Every run is resumable. A record whose .npy already exists and has the right shape is skipped
unless --force is given.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, N_LEADS, log, ml_path, save_json  # noqa: E402

EXPECTED_SAMPLES = 5000


def read_raw_csv(abs_path):
    """Read one raw record. The file is a (12, 5000) matrix with the leads as rows."""
    arr = np.loadtxt(abs_path, delimiter=',', dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError('%s did not parse as a matrix, got shape %s' % (abs_path, arr.shape))
    if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
        arr = arr.T
    if arr.shape[0] != N_LEADS:
        raise ValueError('%s has %d rows, expected %d leads' % (abs_path, arr.shape[0], N_LEADS))
    return np.ascontiguousarray(arr, dtype=np.float32)


def cache_one(job):
    record_id, rel_path, cache_dir, force = job
    dest = os.path.join(cache_dir, record_id + '.npy')
    if not force and os.path.isfile(dest):
        try:
            shape = np.load(dest, mmap_mode='r').shape
            if shape[0] == N_LEADS:
                return record_id, 'skipped', shape[1], ''
        except Exception:
            pass  # a truncated file from an interrupted run, fall through and rewrite it
    src = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT, rel_path)
    try:
        arr = read_raw_csv(src)
    except Exception as exc:
        return record_id, 'failed', 0, str(exc)
    # np.save appends .npy when the name does not already end in it, so the temporary
    # name has to carry the suffix itself or os.replace looks for a file that was never
    # written. Naming it <dest>.tmp<pid>.npy keeps the write atomic and the rename valid.
    tmp = dest + '.tmp%d.npy' % os.getpid()
    np.save(tmp, arr)
    os.replace(tmp, dest)          # atomic, so an interrupted run never leaves a half file
    return record_id, 'written', arr.shape[1], ''


def collect_jobs(units_csvs, cache_dir, force, limit):
    import pandas as pd
    seen = {}
    for path in units_csvs:
        abs_csv = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.isfile(abs_csv):
            raise SystemExit('units table not found at %s' % abs_csv)
        frame = pd.read_csv(abs_csv, usecols=['record_id', 'path_raw'], dtype=str)
        for record_id, rel in zip(frame['record_id'], frame['path_raw']):
            seen.setdefault(record_id, rel)
        log('%s contributed %d rows, %d distinct records so far' % (os.path.basename(path), len(frame), len(seen)))
    jobs = [(rid, rel, cache_dir, force) for rid, rel in sorted(seen.items())]
    if limit:
        jobs = jobs[:limit]
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--units', action='append', default=None,
                        help='a units table. Repeatable. Defaults to the pretrain and finetune tables.')
    parser.add_argument('--cache-dir', default=None, help='defaults to ml_modelling/data/signal_cache')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='cache only the first N records, for a smoke test')
    parser.add_argument('--force', action='store_true', help='rewrite records that are already cached')
    parser.add_argument('--verify', action='store_true', help='reload every cached file and check its shape')
    args = parser.parse_args()

    units = args.units or ['ml_modelling/data/pretrain_units.csv', 'ml_modelling/data/finetune_units.csv']
    cache_dir = args.cache_dir or ml_path('data', 'signal_cache')
    os.makedirs(cache_dir, exist_ok=True)

    jobs = collect_jobs(units, cache_dir, args.force, args.limit)
    log('%d records to consider, cache at %s' % (len(jobs), cache_dir))

    counts = {'written': 0, 'skipped': 0, 'failed': 0}
    failures = []
    lengths = {}

    if args.workers <= 1:
        results = (cache_one(job) for job in jobs)
        for i, (rid, status, n, err) in enumerate(results, 1):
            counts[status] += 1
            lengths[n] = lengths.get(n, 0) + 1
            if status == 'failed':
                failures.append({'record_id': rid, 'error': err})
            if i % 500 == 0:
                log('%d/%d  written %d  skipped %d  failed %d' % (i, len(jobs), counts['written'], counts['skipped'], counts['failed']))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(cache_one, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                rid, status, n, err = future.result()
                counts[status] += 1
                lengths[n] = lengths.get(n, 0) + 1
                if status == 'failed':
                    failures.append({'record_id': rid, 'error': err})
                if i % 500 == 0:
                    log('%d/%d  written %d  skipped %d  failed %d' % (i, len(jobs), counts['written'], counts['skipped'], counts['failed']))

    log('done. written %d, skipped %d, failed %d' % (counts['written'], counts['skipped'], counts['failed']))
    off_length = {k: v for k, v in lengths.items() if k not in (0, EXPECTED_SAMPLES)}
    if off_length:
        log('warning, records whose sample count is not %d: %s' % (EXPECTED_SAMPLES, off_length))

    if args.verify:
        bad = []
        for rid, _, _, _ in jobs:
            dest = os.path.join(cache_dir, rid + '.npy')
            if not os.path.isfile(dest):
                bad.append({'record_id': rid, 'error': 'missing'})
                continue
            try:
                shape = np.load(dest, mmap_mode='r').shape
                if shape[0] != N_LEADS:
                    bad.append({'record_id': rid, 'error': 'shape %s' % (shape,)})
            except Exception as exc:
                bad.append({'record_id': rid, 'error': str(exc)})
        log('verify found %d problem records' % len(bad))
        failures.extend(bad)

    report = {
        'units_tables': units,
        'cache_dir': cache_dir,
        'n_records': len(jobs),
        'counts': counts,
        'sample_length_histogram': lengths,
        'failures': failures[:200],
        'n_failures': len(failures),
    }
    out = ml_path('results', 'cache_signals_report.json')
    save_json(report, out)
    log('report written to %s' % out)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
