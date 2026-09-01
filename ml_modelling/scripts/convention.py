#!/usr/bin/env python3
"""
convention.py  -  re-derive waveform boundaries under the external corpus's own rule.

    python3 ml_modelling/scripts/convention.py --validate
    python3 ml_modelling/scripts/convention.py --apply <units.csv> --out <path.csv>

"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CACHE = os.path.join(ROOT, 'ml_modelling', 'data', 'signal_cache')
MAG_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]      # I, II, V1 to V6, the eight independent leads
FRAC = 0.05


def spatial_magnitude(sig, smooth=5):
    x = np.asarray(sig, dtype=np.float64)[MAG_LEADS]
    m = np.sqrt(((x - np.median(x, axis=1, keepdims=True)) ** 2).sum(axis=0))
    if smooth > 1:
        pad = smooth // 2
        m = np.convolve(np.pad(m, pad, mode='edge'), np.ones(smooth) / smooth, mode='valid')
    return m


def _cross_left(m, peak, level):
    """Last sample before the peak at which the curve is still below the level."""
    i = int(peak)
    while i > 0 and m[i] > level:
        i -= 1
    return i


def _cross_right(m, peak, level, stop):
    i = int(peak)
    stop = int(min(stop, m.size - 1))
    while i < stop and m[i] > level:
        i += 1
    return i


def derive(sig, qrs_hint, t_hint, baseline_window=None):
    """Boundaries under the external rule. Hints bracket the search and need not be accurate."""
    m = spatial_magnitude(sig)
    n = m.size
    # An external recording begins at ventricular activation, so there is no pre-P segment to
    # take a baseline from. The quiet run is the post-T tail instead.
    if baseline_window:
        lo, hi = baseline_window
    else:
        pre = int(qrs_hint[0]) - 5
        lo, hi = (0, pre) if pre >= 12 else (int(min(n - 1, t_hint[1] + 10)), n)
    lo, hi = max(0, int(lo)), min(n, int(hi))
    base = float(np.median(m[lo:hi])) if hi - lo >= 4 else float(np.percentile(m, 10))

    a, b = int(max(0, qrs_hint[0] - 25)), int(min(n, qrs_hint[1] + 25))
    q_peak = a + int(np.argmax(m[a:b])) if b > a else int(qrs_hint[0])
    c, d = int(max(0, t_hint[0] - 30)), int(min(n, t_hint[1] + 30))
    t_peak = c + int(np.argmax(m[c:d])) if d > c else int(t_hint[0])

    q_level = base + FRAC * (m[q_peak] - base)
    t_level = base + FRAC * (m[t_peak] - base)

    # the QRS offset is additionally held at or before the trough between the two peaks
    trough = q_peak + int(np.argmin(m[q_peak:max(q_peak + 2, t_peak)])) if t_peak > q_peak + 1 else n - 1

    return {
        'qrs_onset': _cross_left(m, q_peak, q_level),
        'qrs_offset': min(_cross_right(m, q_peak, q_level, trough), trough),
        't_onset': max(_cross_left(m, t_peak, t_level), trough),
        't_offset': _cross_right(m, t_peak, t_level, n - 1),
        'q_peak_mag': q_peak, 't_peak_mag': t_peak,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', action='store_true')
    p.add_argument('--per-unit', default='ml_modelling/results/context_external/per_unit.csv')
    p.add_argument('--apply', default='')
    p.add_argument('--out', default='')
    a = p.parse_args()

    src = os.path.join(ROOT, a.apply or a.per_unit)
    u = pd.read_csv(src)
    B = ['qrs_onset', 'qrs_offset', 't_onset', 't_offset']
    rows = []
    for rec, s in u.groupby('record_id'):
        f = os.path.join(CACHE, '%s.npy' % rec)
        if not os.path.isfile(f):
            continue
        sig = np.load(f)
        r = s.iloc[0]
        qh = (r.get('qrs_onset_true', np.nan), r.get('qrs_offset_true', np.nan))
        th = (r.get('t_onset_true', np.nan), r.get('t_offset_true', np.nan))
        if not all(np.isfinite(v) for v in qh + th):
            continue
        got = derive(sig, qh, th)
        rows.append(dict(record_id=rec, **{k: got[k] for k in B},
                         **{k + '_ref': float(r[k + '_true']) for k in B}))
    d = pd.DataFrame(rows)
    print('recordings processed %d' % len(d))
    if a.validate:
        print('\nCONTROL, the rule applied to the corpus that defines it')
        print('  %-12s %10s %10s %10s' % ('boundary', 'median err', 'MAE', 'p90'))
        for k in B:
            e = 2.0 * (d[k] - d[k + '_ref'])
            print('  %-12s %8.2f ms %8.2f ms %8.2f ms'
                  % (k, e.median(), e.abs().mean(), np.percentile(e.abs(), 90)))
        agg = 2.0 * pd.concat([(d[k] - d[k + '_ref']) for k in B]).abs().mean()
        print('  %-12s %8.2f ms over all four' % ('aggregate', agg))
        print('\n  a faithful replay sits near zero. Anything above a few milliseconds means the')
        print('  rule implemented here is not the rule that produced the reference.')
    if a.out:
        dest = os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        d.to_csv(dest, index=False)
        print('\nwritten to %s' % a.out)


if __name__ == '__main__':
    main()
