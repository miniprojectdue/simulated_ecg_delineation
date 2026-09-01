#!/usr/bin/env python3
"""
run_baseline.py  -  score delineate_ecg_v3 on an external test set through the model's harness.

    python3 ml_modelling/scripts/run_baseline.py \
        --units /path/to/smith2026_test_units_alllead.csv \
        --out   ml_modelling/results/baseline_external

"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_v3 import LANDMARKS_OUT, delineate_v3, normalise_record  # noqa: E402
from biomarkers import BIOMARKER_NAMES, biomarker_errors, summarise_biomarker_errors  # noqa: E402
from common import LANDMARKS  # noqa: E402
from metrics import FiducialAccumulator  # noqa: E402

FS = 500.0
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def load_signal(path):
    """One recording as 12 x N, whichever way round the file stores it."""
    a = np.loadtxt(path, delimiter=',')
    if a.ndim != 2:
        raise ValueError('%s is not a 2-D table' % path)
    if a.shape[0] != len(LEAD_ORDER):
        a = a.T
    if a.shape[0] != len(LEAD_ORDER):
        raise ValueError('%s has %d leads, expected %d' % (path, a.shape[0], len(LEAD_ORDER)))
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--units', required=True, help='the units table to score')
    p.add_argument('--out', required=True, help='directory for metrics.json and per_unit.csv')
    a = p.parse_args()

    units = pd.read_csv(a.units)
    lead_index = {name: i for i, name in enumerate(LEAD_ORDER)}

    cache = {}
    for rec, path in units[['record_id', 'path_raw']].drop_duplicates().itertuples(index=False):
        if not os.path.exists(path):
            raise SystemExit('signal not found for %s at %s' % (rec, path))
        cache[rec] = normalise_record(load_signal(path))
    print('loaded %d recording(s) from the paths in the units table' % len(cache))

    predicted_rows, truth_rows, meta_rows, bio_rows = [], [], [], []
    fallbacks = {}

    for _, row in units.iterrows():
        rec, lead = row['record_id'], row['lead']
        x = cache[rec][lead_index[lead]]
        out = delineate_v3(x, fs_hz=FS)
        for f in out['diag']['fallbacks']:
            fallbacks[f] = fallbacks.get(f, 0) + 1

        # predictions are 500 Hz samples, the reference is in ms, so convert to a common ms grid
        pred = {k: None for k in LANDMARKS}
        for k in LANDMARKS_OUT:
            pred[k] = float(out[k]) * (1000.0 / FS)
        truth = {k: None for k in LANDMARKS}
        for k in LANDMARKS:
            v = row.get(k + '_ms', np.nan)
            truth[k] = None if pd.isna(v) else float(v)

        predicted_rows.append(pred)
        truth_rows.append(truth)
        bio_rows.append(biomarker_errors(pred, truth, fs_hz=1000.0))
        meta_rows.append({
            'record_id': rec, 'lead': lead,
            'disease_class': row.get('disease_class', ''),
            'label_quality': row.get('label_quality', ''),
            'qrs_end_flag': out['qrs_end_flag'],
        })

    os.makedirs(a.out, exist_ok=True)

    def summarise(mask, title):
        acc = FiducialAccumulator(fs_hz=1000.0)          # inputs already in ms
        for keep, pr, t in zip(mask, predicted_rows, truth_rows):
            if keep:
                acc.update_one(pr, t)
        bio = summarise_biomarker_errors([b for keep, b in zip(mask, bio_rows) if keep])
        return {'title': title, 'n_units': int(sum(mask)), 'fiducials': acc.summary(), 'biomarkers': bio}

    all_mask = [True] * len(meta_rows)
    result = summarise(all_mask, 'delineate_ecg_v3, external test set, all leads')
    result['n_records'] = int(units['record_id'].nunique())
    result['fs_hz'] = FS
    result['units_csv'] = os.path.abspath(a.units)
    result['reference_units'] = 'milliseconds on the original 1000 Hz grid'
    result['landmarks_produced'] = LANDMARKS_OUT
    result['fallback_counts'] = fallbacks

    # the six leads the published driver configures
    driver = [m['lead'] in ('V1', 'V2', 'V3', 'V4', 'V5', 'V6') for m in meta_rows]
    result['by_lead_subset_v1_v6'] = summarise(driver, 'driver lead set V1 to V6')

    result['by_lead'] = {}
    for lead in LEAD_ORDER:
        m = [x['lead'] == lead for x in meta_rows]
        if not any(m):
            continue
        s = summarise(m, lead)
        result['by_lead'][lead] = {
            'n_units': s['n_units'],
            'boundary_mae_ms': s['fiducials']['_aggregate']['boundary_mae_ms'],
            'peak_mae_ms': s['fiducials']['_aggregate']['peak_mae_ms'],
        }

    result['by_disease_class'] = {}
    for cls in sorted({x['disease_class'] for x in meta_rows if x['disease_class']}):
        m = [x['disease_class'] == cls for x in meta_rows]
        s = summarise(m, cls)
        result['by_disease_class'][cls] = {
            'n_units': s['n_units'],
            'boundary_mae_ms': s['fiducials']['_aggregate']['boundary_mae_ms'],
            'qt_bias_ms': s['biomarkers']['qt_interval']['bias_ms'],
            'qt_mae_ms': s['biomarkers']['qt_interval']['mae_ms'],
        }

    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)

    per_unit = pd.DataFrame(meta_rows)
    for k in LANDMARKS_OUT:
        per_unit['pred_' + k] = [pr[k] for pr in predicted_rows]
        per_unit['true_' + k] = [t[k] for t in truth_rows]
        per_unit['err_' + k] = [
            (pr[k] - t[k]) if (pr[k] is not None and t[k] is not None) else np.nan
            for pr, t in zip(predicted_rows, truth_rows)]
    for b in BIOMARKER_NAMES:
        per_unit['bioerr_' + b] = [r[b] for r in bio_rows]
    per_unit.to_csv(os.path.join(a.out, 'per_unit.csv'), index=False)

    agg = result['fiducials']['_aggregate']
    print('units %d over %d records' % (result['n_units'], result['n_records']))
    print('boundary MAE %.2f ms | peak MAE %.2f ms' % (agg['boundary_mae_ms'], agg['peak_mae_ms']))
    print('fallbacks:', fallbacks)
    print()
    for k in LANDMARKS_OUT:
        e = result['fiducials'][k]
        print('  %-11s bias %+7.2f  MAE %6.2f  sd %6.2f  within25 %.3f  se %.3f'
              % (k, e['bias_ms'], e['mae_ms'], e['sd_ms'], e['within_25ms'], e['sensitivity']))
    print()
    for b in BIOMARKER_NAMES:
        s = result['biomarkers'][b]
        print('  %-14s bias %+7.2f  MAE %6.2f  coverage %.3f' % (b, s['bias_ms'], s['mae_ms'], s['coverage']))
    print('\nwritten to %s' % a.out)


if __name__ == '__main__':
    main()
