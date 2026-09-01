#!/usr/bin/env python3
"""
compare_baseline.py  -  the head to head between delineate_ecg_v3 and the network.

    python3 ml_modelling/scripts/compare_baseline.py \
        --baseline ml_modelling/results/baseline_external/per_unit.csv \
        --model    ml_modelling/results/context_external/per_unit.csv \
        --arm-a    ml_modelling/results/context_armA_external/per_unit.csv \
        --out      ml_modelling/results/baseline_external/comparison.json

Both systems are scored on identical units against an identical reference, so the comparison is
paired unit by unit. Statistics are clustered by recording, since the twelve leads of a recording
share one QRS boundary and are not independent observations, and treating them as independent
would shrink every interval by about the square root of twelve.

Splits are reported and never pooled. The matched classes are the recordings whose morphology has
a counterpart in the training corpus. The ischemia recordings have none, so they are an
out-of-distribution test for the network and ordinary terrain for the baseline. Split sizes are
counted from the data rather than written into this file, so the same script serves whatever
dataset it is pointed at.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SCOREABLE = ['qrs_onset', 'qrs_offset', 't_offset']     # T onset excluded, see the protocol
ALL_FIVE = ['qrs_onset', 'qrs_offset', 't_onset', 't_peak', 't_offset']
MATCHED = ['Healthy', 'AnteriorInfarction', 'InferiorInfarction']
ISCHEMIA = ['AnteriorIschemia', 'InferiorIschemia']


def cluster_stats(values, records, n_boot=5000, seed=1337):
    """Mean and a 95 per cent interval from a bootstrap that resamples whole recordings."""
    df = pd.DataFrame({'v': values, 'r': records}).dropna()
    if df.empty:
        return float('nan'), float('nan'), float('nan'), 0
    per_record = df.groupby('r')['v'].mean()
    point = float(per_record.mean())
    rng = np.random.default_rng(seed)
    arr = per_record.to_numpy()
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    draws = arr[idx].mean(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), int(arr.size)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--arm-a', default=None, help='optional stage-one-only per_unit.csv')
    p.add_argument('--out', default=None, help='where to write comparison.json')
    a = p.parse_args()

    b = pd.read_csv(a.baseline)
    m = pd.read_csv(a.model)
    key = ['record_id', 'lead']
    b = b.sort_values(key).reset_index(drop=True)
    m = m.sort_values(key).reset_index(drop=True)
    if len(b) != len(m) or not (b.record_id.values == m.record_id.values).all() \
            or not (b.lead.values == m.lead.values).all():
        raise SystemExit('the two tables do not cover the same units, refusing to compare')

    cls = b['disease_class'].values
    rec = b['record_id'].values
    n_rec = lambda mask: int(pd.unique(rec[mask]).size)  # noqa: E731
    all_mask = np.ones(len(b), dtype=bool)
    matched, isch = np.isin(cls, MATCHED), np.isin(cls, ISCHEMIA)
    splits = {
        'all %d recordings' % n_rec(all_mask): all_mask,
        'matched, %d recordings' % n_rec(matched): matched,
        'ischemia, %d recordings' % n_rec(isch): isch,
    }

    report = {}
    for split_name, mask in splits.items():
        if not mask.any():
            continue
        print('\n' + '=' * 84)
        print(split_name.upper())
        print('=' * 84)
        print('%-11s | %-24s | %-24s | %s' % ('landmark', 'baseline v3 MAE', 'network MAE', 'difference, net minus v3'))
        entry = {}
        for k in ALL_FIVE:
            eb = np.abs(b['err_' + k].values)[mask]
            em = np.abs(m[k + '_err_ms'].values)[mask]
            pb, lb, ub, nrec = cluster_stats(eb, rec[mask])
            pm, lm, um, _ = cluster_stats(em, rec[mask])
            d, dl, du, _ = cluster_stats(em - eb, rec[mask])
            sig = 'favours v3' if dl > 0 else ('favours network' if du < 0 else 'not significant')
            flag = '' if k in SCOREABLE else '  [not scoreable, see protocol]' if k == 't_onset' else '  [extremum, not a judgement]'
            print('%-11s | %6.2f  [%5.2f, %5.2f] | %6.2f  [%5.2f, %5.2f] | %+6.2f [%+6.2f, %+6.2f]  %s%s'
                  % (k, pb, lb, ub, pm, lm, um, d, dl, du, sig, flag))
            entry[k] = {'baseline_mae_ms': pb, 'baseline_ci': [lb, ub],
                        'network_mae_ms': pm, 'network_ci': [lm, um],
                        'paired_diff_ms': d, 'paired_ci': [dl, du], 'verdict': sig}

        eb = np.nanmean(np.abs(np.column_stack([b['err_' + k].values for k in SCOREABLE])), axis=1)[mask]
        em = np.nanmean(np.abs(np.column_stack([m[k + '_err_ms'].values for k in SCOREABLE])), axis=1)[mask]
        pb, lb, ub, nrec = cluster_stats(eb, rec[mask])
        pm, lm, um, _ = cluster_stats(em, rec[mask])
        d, dl, du, _ = cluster_stats(em - eb, rec[mask])
        sig = 'favours v3' if dl > 0 else ('favours network' if du < 0 else 'not significant')
        print('-' * 84)
        print('%-11s | %6.2f  [%5.2f, %5.2f] | %6.2f  [%5.2f, %5.2f] | %+6.2f [%+6.2f, %+6.2f]  %s'
              % ('BOUNDARY', pb, lb, ub, pm, lm, um, d, dl, du, sig))
        print('   over %s, clustered on %d recordings' % (', '.join(SCOREABLE), nrec))
        entry['_boundary'] = {'baseline_mae_ms': pb, 'network_mae_ms': pm,
                              'paired_diff_ms': d, 'paired_ci': [dl, du], 'verdict': sig,
                              'n_records': nrec, 'landmarks': SCOREABLE}

        print('\n%-14s | %-22s | %-22s | %s' % ('biomarker', 'baseline v3', 'network', 'difference'))
        for bio, (s, e) in {'qrs_duration': ('qrs_onset', 'qrs_offset'),
                            'qt_interval': ('qrs_onset', 't_offset'),
                            't_peak_to_end': ('t_peak', 't_offset')}.items():
            vb = np.abs((b['pred_' + e] - b['pred_' + s]) - (b['true_' + e] - b['true_' + s])).values[mask]
            vm = np.abs((m[e + '_pred'] - m[s + '_pred']) * 2 - (m[e + '_true'] - m[s + '_true']) * 2).values[mask]
            pb2, lb2, ub2, _ = cluster_stats(vb, rec[mask])
            pm2, lm2, um2, _ = cluster_stats(vm, rec[mask])
            d2, dl2, du2, _ = cluster_stats(vm - vb, rec[mask])
            sig2 = 'favours v3' if dl2 > 0 else ('favours network' if du2 < 0 else 'not significant')
            print('%-14s | %6.2f [%5.2f, %5.2f] | %6.2f [%5.2f, %5.2f] | %+6.2f  %s'
                  % (bio, pb2, lb2, ub2, pm2, lm2, um2, d2, sig2))
            entry['bio_' + bio] = {'baseline_mae_ms': pb2, 'network_mae_ms': pm2,
                                   'paired_diff_ms': d2, 'paired_ci': [dl2, du2], 'verdict': sig2}
        report[split_name] = entry

    if a.arm_a and os.path.exists(a.arm_a):
        arm = pd.read_csv(a.arm_a).sort_values(key).reset_index(drop=True)
        print('\n' + '=' * 84)
        print('STAGE ONE ONLY, arm A, boundary aggregate on every recording')
        ea = np.nanmean(np.abs(np.column_stack([arm[k + '_err_ms'].values for k in SCOREABLE])), axis=1)
        pa, la, ua, _ = cluster_stats(ea, arm['record_id'].values)
        print('  arm A %.2f [%.2f, %.2f]' % (pa, la, ua))
        report['armA_boundary_all'] = {'mae_ms': pa, 'ci': [la, ua]}

    out = a.out or os.path.join(os.path.dirname(a.baseline), 'comparison.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, 'w'), indent=2)
    print('\nwritten to %s' % out)


if __name__ == '__main__':
    main()
