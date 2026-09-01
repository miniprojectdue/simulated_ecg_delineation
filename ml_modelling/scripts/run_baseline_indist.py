#!/usr/bin/env python3
"""
run_baseline_indist.py  -  delineate_ecg_v3 on the held-out MedalCare-XL test set.

This is the mirror of the external run. There the baseline was on its home corpus and the
network was out of distribution. Here the roles reverse.

"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/mnt/user-data/uploads/Delineation/ml_modelling/scripts')

from baseline_v3 import LANDMARKS_OUT, delineate_v3, normalise_record  # noqa: E402
from biomarkers import BIOMARKER_NAMES, biomarker_errors, summarise_biomarker_errors  # noqa: E402
from common import LANDMARKS  # noqa: E402
from metrics import FiducialAccumulator  # noqa: E402

BUNDLE = '/mnt/user-data/uploads/Delineation/_mc_bundle.txt'
UNITS = '/mnt/user-data/uploads/Delineation/ml_modelling/data/finetune_units.csv'
INDIST = '/mnt/user-data/uploads/Delineation/ml_modelling/results/armB_indist/per_unit.csv'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
FS = 500.0
LEAD_ALIGN_LO, LEAD_ALIGN_HI = 100, 180     # ms of segment ahead of the reviewer's QRS onset
SEED = 1337


def load_bundle(path):
    recs, cur, rows = {}, None, []
    for line in open(path):
        if line.startswith('##REC'):
            if cur:
                recs[cur] = np.array(rows, dtype=np.float64)
            cur, rows = line.split()[1].strip(), []
        else:
            rows.append([float(x) for x in line.strip().split(',')])
    if cur:
        recs[cur] = np.array(rows, dtype=np.float64)
    return recs


def main():
    recs = load_bundle(BUNDLE)
    units = pd.read_csv(UNITS)
    test = pd.read_csv(INDIST)[['record_id', 'lead']].drop_duplicates()
    sub = units.merge(test, on=['record_id', 'lead']).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    offsets = rng.integers(LEAD_ALIGN_LO, LEAD_ALIGN_HI + 1, size=len(sub))

    results = {}
    for condition in ('natural', 'p_blanked'):
        pred_rows, truth_rows, bio_rows, meta = [], [], [], []
        for k, (_, r) in enumerate(sub.iterrows()):
            x = recs.get(r.path_raw)
            if x is None:
                continue
            qon = int(round(r.qrs_onset_sample))
            lead_ms = int(offsets[k])
            start = max(0, qon - int(lead_ms * FS / 1000.0))
            stop = min(x.shape[1], int(r.beat_end_sample) + 1)
            if stop - start < 200:
                continue
            seg12 = normalise_record(x[:, start:stop])
            v = seg12[LEADS.index(r.lead)].copy()
            if condition == 'p_blanked':
                cut = max(0, qon - start - int(20 * FS / 1000.0))
                if cut > 3:
                    v[:cut] = float(np.median(v[:cut]))
            out = delineate_v3(v, fs_hz=FS)

            pred = {n: None for n in LANDMARKS}
            for n in LANDMARKS_OUT:
                pred[n] = float(out[n] + start)
            truth = {n: None for n in LANDMARKS}
            for n in LANDMARKS:
                val = r.get(n + '_sample', np.nan)
                truth[n] = None if pd.isna(val) else float(val)
            pred_rows.append(pred)
            truth_rows.append(truth)
            bio_rows.append(biomarker_errors(pred, truth, fs_hz=FS))
            meta.append({'record_id': r.record_id, 'lead': r.lead,
                         'disease_class': r.disease_class, 'lead_in_ms': lead_ms})

        acc = FiducialAccumulator(fs_hz=FS)
        for p, t in zip(pred_rows, truth_rows):
            acc.update_one(p, t)
        fid = acc.summary()
        results[condition] = {
            'n_units': len(meta),
            'n_records': len({m['record_id'] for m in meta}),
            'fiducials': fid,
            'biomarkers': summarise_biomarker_errors(bio_rows),
        }
        frame = pd.DataFrame(meta)
        for n in LANDMARKS_OUT:
            frame['pred_' + n] = [p[n] for p in pred_rows]
            frame['true_' + n] = [t[n] for t in truth_rows]
            frame['err_' + n] = [(p[n] - t[n]) * 1000.0 / FS if (p[n] is not None and t[n] is not None)
                                 else np.nan for p, t in zip(pred_rows, truth_rows)]
        for b in BIOMARKER_NAMES:
            frame['bioerr_' + b] = [r[b] for r in bio_rows]
        os.makedirs('/home/claude/baseline_indist', exist_ok=True)
        frame.to_csv('/home/claude/baseline_indist/per_unit_%s.csv' % condition, index=False)

        agg = fid['_aggregate']
        print('\n%s  (%d units, %d records)' % (condition.upper(), len(meta), results[condition]['n_records']))
        print('  boundary MAE %.2f ms   over qrs_onset, qrs_offset, t_onset, t_offset' % agg['boundary_mae_ms'])
        for n in LANDMARKS_OUT:
            e = fid[n]
            print('    %-11s bias %+7.2f  MAE %7.2f  sens %.3f' % (n, e['bias_ms'], e['mae_ms'], e['sensitivity']))

    json.dump(results, open('/home/claude/baseline_indist/metrics.json', 'w'), indent=2)
    print('\nwritten to baseline_indist/')


if __name__ == '__main__':
    main()
