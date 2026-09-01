
import argparse
import os
import numpy as np
import pandas as pd

_WIN_COLS = {'record_id', 'disease_class', 'win_start_sample', 'win_end_sample'}
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def robust_normalise(w, eps=1e-6, clip=20.0):
    """The robust per-lead normaliser used by dataset.normalise(mode='robust')."""
    x = w.astype(np.float32, copy=True)
    c = np.median(x, axis=1, keepdims=True)
    q75 = np.percentile(x, 75, axis=1, keepdims=True)
    q25 = np.percentile(x, 25, axis=1, keepdims=True)
    s = np.maximum((q75 - q25) / 1.349, eps)
    x = (x - c) / s
    if clip:
        np.clip(x, -clip, clip, out=x)
    return x


def load_signal(record_id, cache=None, signals=None):
    """A 12 x N array for a record, from a .npy cache or a <record>_raw.csv directory."""
    if cache:
        return np.load(os.path.join(cache, str(record_id) + '.npy'))
    s = np.loadtxt(os.path.join(signals, str(record_id) + '_raw.csv'), delimiter=',')
    return s if s.shape[0] == 12 else s.T


def gather(units_csv, cache, signals, per_lead, sample_records, clip, seed):
    df = pd.read_csv(units_csv, usecols=lambda c: c in _WIN_COLS)
    recs = df.drop_duplicates('record_id')
    if sample_records and len(recs) > sample_records:
        if 'disease_class' in recs.columns:
            k = max(1, sample_records // max(1, recs['disease_class'].nunique()))
            recs = recs.groupby('disease_class', group_keys=False).apply(
                lambda g: g.sample(min(len(g), k), random_state=seed))
        else:
            recs = recs.sample(sample_records, random_state=seed)
    per = [[] for _ in range(12)]
    pool = []
    used = 0
    for r in recs.itertuples():
        try:
            sig = load_signal(r.record_id, cache, signals)
        except Exception:
            continue
        n = sig.shape[1]
        a = max(0, int(r.win_start_sample)); b = min(n, int(r.win_end_sample))
        if b - a < 30:
            a, b = 0, n
        z = robust_normalise(sig[:12, a:b], clip=clip)
        if per_lead:
            for i in range(12):
                per[i].append(z[i])
        else:
            pool.append(z.ravel())
        used += 1
    if used == 0:
        raise SystemExit('no signals loaded from %s' % units_csv)
    print('  %s: %d records used' % (os.path.basename(units_csv), used))
    return [np.concatenate(p) for p in per] if per_lead else np.concatenate(pool)


def quantile_vector(vals, L, enforce_increasing):
    q = np.quantile(vals, L).astype(np.float64)
    if enforce_increasing:
        q = np.maximum.accumulate(q)
        d = np.diff(q)
        q[1:] += np.cumsum(np.where(d <= 0, 1e-6, 0.0))
    return q


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-units', required=True, help='training units CSV (the target of the map)')
    p.add_argument('--train-cache', required=True, help='training signal cache (.npy per record)')
    p.add_argument('--domain-units', required=True, help='evaluation-domain units CSV (the source)')
    p.add_argument('--domain-cache', default='', help='evaluation-domain .npy cache dir')
    p.add_argument('--domain-signals', default='', help='evaluation-domain <record>_raw.csv dir')
    p.add_argument('--out', required=True, help='output .csv')
    p.add_argument('--per-lead', action='store_true', help='one mapping per lead (default: global)')
    p.add_argument('--levels', type=int, default=4001)
    p.add_argument('--clip', type=float, default=20.0, help='must match normalisation.clip_sigma')
    p.add_argument('--sample-records', type=int, default=320, help='cap on training records sampled')
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()
    if not (args.domain_cache or args.domain_signals):
        raise SystemExit('pass --domain-cache or --domain-signals')

    L = np.linspace(0.0, 1.0, args.levels)
    print('building quantile-match table (source -> target)')
    tgt = gather(args.train_units, args.train_cache, None, args.per_lead,
                 args.sample_records, args.clip, args.seed)
    src = gather(args.domain_units, args.domain_cache or None, args.domain_signals or None,
                 args.per_lead, 0, args.clip, args.seed)

    import csv
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        if args.per_lead:
            w.writerow(['lead', 'level', 'external', 'training'])
            for i in range(12):
                sq = quantile_vector(src[i], L, True)     # source (external) quantiles
                rq = quantile_vector(tgt[i], L, False)    # target (training) quantiles
                for lv, s, r in zip(L, sq, rq):
                    w.writerow([LEADS[i], '%.6f' % lv, '%.8g' % s, '%.8g' % r])
        else:
            w.writerow(['level', 'external', 'training'])
            sq = quantile_vector(src, L, True)
            rq = quantile_vector(tgt, L, False)
            for lv, s, r in zip(L, sq, rq):
                w.writerow(['%.6f' % lv, '%.8g' % s, '%.8g' % r])
    print('saved %s  (%d levels%s)'
          % (args.out, args.levels, ', per-lead' if args.per_lead else ', global'))


if __name__ == '__main__':
    main()
