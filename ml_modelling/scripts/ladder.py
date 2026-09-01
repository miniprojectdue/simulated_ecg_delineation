import numpy as np, pandas as pd

OFF = 17.57  # measured T-offset convention gap, ms

n = pd.read_csv('ml_modelling/results/aug_external/per_unit.csv')
b = pd.read_csv('ml_modelling/results/baseline_external/per_unit.csv')
key = ['record_id', 'lead']
n['k'] = n['record_id'].astype(str) + '|' + n['lead'].astype(str)
b['k'] = b['record_id'].astype(str) + '|' + b['lead'].astype(str)
b = b.set_index('k'); n = n.set_index('k')
k = n.index.intersection(b.index)
n, b = n.loc[k], b.loc[k]

def boot(diff, recs, R=4000):
    uniq = np.unique(recs); grp = {r: np.where(recs == r)[0] for r in uniq}
    rng = np.random.default_rng(1337); out = np.empty(R)
    for i in range(R):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        out[i] = np.concatenate([diff[grp[r]] for r in pick]).mean()
    return np.percentile(out, [2.5, 97.5])

def run(label, marks, corrected):
    ne, be = [], []
    for m in marks:
        e = n[m + '_err_ms'].values.astype(float).copy()
        if corrected and m == 't_offset':
            e = e - OFF
        ne.append(np.abs(e)); be.append(np.abs(b['err_' + m].values.astype(float)))
    N = np.nanmean(np.vstack(ne), axis=0); B = np.nanmean(np.vstack(be), axis=0)
    ok = np.isfinite(N) & np.isfinite(B)
    N, B, recs = N[ok], B[ok], n['record_id'].values[ok]
    d = N - B
    lo, hi = boot(d, recs)
    v = 'tie' if lo <= 0 <= hi else ('favours the baseline' if d.mean() > 0 else 'favours the network')
    print('%-46s %7.2f %7.2f  %+7.2f [%+.2f, %+.2f]  %s' % (label, N.mean(), B.mean(), d.mean(), lo, hi, v))

print('%-46s %7s %7s  %s' % ('adjustment', 'network', 'v3', 'paired difference'))
A = ['qrs_onset', 'qrs_offset', 't_offset']
run('as scored, three boundaries', A, False)
run('T-offset convention accounted for', A, True)
run('degenerate QRS onset excluded', ['qrs_offset', 't_offset'], False)
run('both adjustments', ['qrs_offset', 't_offset'], True)
print()
for m in A:
    run('  ' + m + ' alone', [m], False)
run('  t_offset alone, convention accounted for', ['t_offset'], True)
