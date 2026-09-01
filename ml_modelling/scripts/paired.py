#!/usr/bin/env python3
"""Paired, record-clustered bootstrap on the three-boundary aggregate."""
import sys, numpy as np, pandas as pd

B = ['qrs_onset', 'qrs_offset', 't_offset']

def load(path):
    d = pd.read_csv(path)
    cols = [b + '_err_ms' for b in B]
    d['agg'] = d[cols].abs().mean(axis=1, skipna=True)
    key = d['record_id'].astype(str) + '|' + d['lead'].astype(str) + '|' + d['beat_id'].astype(str)
    return d.assign(key=key).set_index('key')[['record_id', 'agg']]

def compare(a_path, b_path, label_a, label_b):
    a, b = load(a_path), load(b_path)
    k = a.index.intersection(b.index)
    a, b = a.loc[k], b.loc[k]
    ok = a['agg'].notna() & b['agg'].notna()
    a, b = a[ok], b[ok]
    diff = (b['agg'] - a['agg']).values
    recs = a['record_id'].values
    uniq = np.unique(recs)
    grp = {r: np.where(recs == r)[0] for r in uniq}
    rng = np.random.default_rng(1337)
    boot = np.empty(4000)
    for i in range(4000):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        boot[i] = np.concatenate([diff[grp[r]] for r in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    verdict = 'no difference' if lo <= 0 <= hi else ('favours ' + (label_b if diff.mean() < 0 else label_a))
    print('%-22s %8.2f   %-22s %8.2f   %+7.2f [%+.2f, %+.2f]  %s'
          % (label_a, a['agg'].mean(), label_b, b['agg'].mean(), diff.mean(), lo, hi, verdict))

if __name__ == '__main__':
    compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
