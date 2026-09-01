#!/usr/bin/env python3
"""
ensemble.py  -  score several checkpoints together, and optionally at several crop offsets.

    python3 ml_modelling/scripts/ensemble.py \
        --checkpoints A/best.pt B/best.pt C/best.pt \
        --units <units.csv> [--split test] --tag <name> [--shifts 0,-8,8]

Neither of the two things this does requires any training. Averaging the class posteriors of
several checkpoints reduces the part of the error that is particular to one optimisation run, and
averaging over small crop offsets reduces the part that comes from where the beat happens to sit
inside the window. Both act on scatter rather than on bias, so neither can repair a systematic
displacement such as a labelling convention.

Every downstream step is the one evaluate.py uses, including the post-processing, the accumulators
and the per-unit table, so a result from here is directly comparable with a result from there.

Note, the stage-two checkpoints share a stage-one initialisation, so they
are correlated and the gain is smaller than independent models would give. The offsets are applied
by rolling the crop, which wraps a few samples from one end to the other, and that is harmless only
while the shift is small and the beat sits away from the crop edges. Shifts beyond about 16 samples
should not be used.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as E                                                        # noqa: E402
from common import LANDMARKS, log, pick_device, save_json                  # noqa: E402
from dataset import BeatWindowDataset, collate, dataset_kwargs, load_units  # noqa: E402
from metrics import (ConfusionAccumulator, FiducialAccumulator, format_report,  # noqa: E402
                     landmarks_from_vector)
from postprocess import batch_fiducials, enforce_order                     # noqa: E402
from biomarkers import BIOMARKER_NAMES, biomarker_errors, summarise_biomarker_errors  # noqa: E402


@torch.no_grad()
def posterior(models, batch, device, shifts):
    """Mean softmax posterior over every model and every crop offset."""
    signal = batch['signal'].to(device, non_blocking=True)
    lead_idx = batch['lead_idx'].to(device, non_blocking=True)
    valid = batch['valid'].to(device, non_blocking=True) if 'valid' in batch else None
    total, n, p_acc = None, 0, []
    for m in models:
        for s in shifts:
            x = torch.roll(signal, shifts=int(s), dims=-1) if s else signal
            v = (torch.roll(valid, shifts=int(s), dims=-1) if (s and valid is not None) else valid)
            out = m(x, lead_idx, valid=v) if getattr(m, 'p_head', None) is not None else m(x, lead_idx)
            lg, p_logit = out if isinstance(out, tuple) else (out, None)
            pr = torch.softmax(lg.float(), dim=1)
            if s:
                pr = torch.roll(pr, shifts=-int(s), dims=-1)   # back into the original alignment
            total = pr if total is None else total + pr
            n += 1
            if p_logit is not None:
                p_acc.append(torch.sigmoid(p_logit.float()))
    p_prob = (torch.stack(p_acc).mean(0).cpu().numpy() if p_acc
              else np.full(signal.shape[0], np.nan))
    return total / max(n, 1), p_prob


def run(models, loader, cfg, device, fs_hz, shifts):
    confusion = ConfusionAccumulator(n_classes=int(cfg['model'].get('n_classes', 4)))
    fiducial = FiducialAccumulator(tolerances_ms=cfg['eval'].get('tolerance_ms', [10, 25, 50, 75, 150]),
                                   primary_ms=cfg['eval'].get('primary_tolerance_ms', 25),
                                   fs_hz=fs_hz)
    biomarker_rows, unit_rows = [], []
    step_ms = 1000.0 / float(fs_hz)
    _rp = float(cfg['eval'].get('read_pad_ms', 0.0) or 0.0)
    read_pad = None if _rp < 0 else int(round(_rp * float(fs_hz) / 1000.0))

    for batch in loader:
        prob, p_prob = posterior(models, batch, device, shifts)
        confusion.update(prob.argmax(dim=1).cpu().numpy(), batch['target'].numpy())
        readable = E.widen(batch['supervised'], batch.get('valid'), read_pad)
        predicted = batch_fiducials(prob, batch['trace'], cfg.get('postprocess'),
                                    supervised=batch['supervised'], readable=readable)
        truth = [landmarks_from_vector(v.numpy()) for v in batch['landmarks']]
        fiducial.update(predicted, truth)

        for i, (p, t) in enumerate(zip(predicted, truth)):
            biomarker_rows.append(biomarker_errors(p, t, fs_hz))
            meta = batch['meta'][i]
            row = {'record_id': meta['record_id'], 'lead': meta['lead'], 'beat_id': meta['beat_id'],
                   'disease_class': meta['disease_class'], 'crop_start': meta['crop_start'],
                   'qrs_pattern_pred': p.get('qrs_pattern', ''),
                   'order_ok_pred': enforce_order(p)['order_ok']}
            anchor = E.qrs_anchor(batch['signal'][i].numpy(), batch['valid'][i].numpy())
            qo, qf = p.get('qrs_onset'), p.get('qrs_offset')
            row['p_head_prob'] = '' if not np.isfinite(p_prob[i]) else round(float(p_prob[i]), 4)
            row['seg_emits_p'] = int(p.get('p_onset') is not None)
            row['qrs_anchor'] = '' if anchor is None else anchor
            row['anchor_in_qrs'] = '' if (anchor is None or qo is None or qf is None) \
                else int(float(qo) <= anchor <= float(qf))
            win_edge = E.edge_of(batch['supervised'][i].numpy())
            read_edge = win_edge if readable is None else E.edge_of(readable[i])
            row['window_end'] = '' if win_edge is None else win_edge + meta['crop_start']
            row['read_end'] = '' if read_edge is None else read_edge + meta['crop_start']
            for name in LANDMARKS:
                pv, tv = p.get(name), t.get(name)
                row['%s_pred' % name] = '' if pv is None else int(round(float(pv))) + meta['crop_start']
                row['%s_true' % name] = '' if tv is None else int(round(float(tv))) + meta['crop_start']
                row['%s_err_ms' % name] = '' if (pv is None or tv is None) \
                    else round((float(pv) - float(tv)) * step_ms, 3)
            for name in BIOMARKER_NAMES:
                value = biomarker_rows[-1][name]
                row['%s_err_ms' % name] = '' if not np.isfinite(value) else round(float(value), 3)
            unit_rows.append(row)

    return confusion.summary(), fiducial.summary(), summarise_biomarker_errors(biomarker_rows), unit_rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--units', required=True)
    p.add_argument('--split', default='')
    p.add_argument('--tag', required=True)
    p.add_argument('--shifts', default='0', help='comma separated crop offsets in samples')
    p.add_argument('--device', default='')
    p.add_argument('--batch-size', type=int, default=0)
    p.add_argument('--max-units', type=int, default=0)
    a = p.parse_args()

    shifts = [int(v) for v in str(a.shifts).split(',') if v.strip() != '']
    if max(abs(v) for v in shifts) > 16:
        raise SystemExit('a shift beyond 16 samples wraps signal across the crop edge, refusing')

    device = pick_device(a.device or 'auto')
    models, cfg = [], None
    for c in a.checkpoints:
        m, cfg_i, _ = E.load_checkpoint(c, device)
        m.eval()
        models.append(m)
        cfg = cfg or cfg_i
    log('loaded %d checkpoints, %d crop offsets, %d forward passes per unit'
        % (len(models), len(shifts), len(models) * len(shifts)))

    frame = load_units(a.units, split=(a.split or None), max_units=a.max_units)
    if frame.empty:
        raise SystemExit('no rows selected from %s' % a.units)
    fs_hz = float(frame['fs_hz'].iloc[0]) if 'fs_hz' in frame.columns else float(cfg['data'].get('fs_hz', 500))
    ds = BeatWindowDataset(frame, augment=None, training=False,
                           **dataset_kwargs(cfg, training=False, seed=0, fs_hz=fs_hz))
    bs = a.batch_size or int(cfg['train'].get('eval_batch_size', 64))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False,
                                         num_workers=int(cfg['data'].get('num_workers', 0) or 0),
                                         collate_fn=collate)
    log('scoring %d units over %d records' % (len(ds), frame['record_id'].nunique()))

    seg, fid, bio, unit_rows = run(models, loader, cfg, device, fs_hz, shifts)
    geometry = E.geometry_report(unit_rows, int(cfg['data']['crop_length']))

    out_dir = os.path.join(E.resolve(cfg['run'].get('log_root', 'ml_modelling/results')), a.tag)
    os.makedirs(out_dir, exist_ok=True)
    save_json({'checkpoints': [E.resolve(c) for c in a.checkpoints], 'shifts': shifts,
               'units_csv': a.units, 'n_units': len(ds),
               'n_records': int(frame['record_id'].nunique()), 'fs_hz': fs_hz,
               'segmentation': seg, 'fiducials': fid, 'biomarkers': bio,
               'by_disease_class': E.by_group(unit_rows, 'disease_class'),
               'by_lead': E.by_group(unit_rows, 'lead'),
               'postprocess': cfg.get('postprocess'),
               'read_pad_ms': float(cfg['eval'].get('read_pad_ms', 0.0) or 0.0),
               'geometry': {k: v for k, v in geometry.items() if k != 'report'}},
              os.path.join(out_dir, 'metrics.json'))
    report = format_report(seg, fid, bio, title='%s, %d checkpoints, shifts %s'
                           % (a.tag, len(models), shifts))
    report = report + '\n\n' + geometry['report']
    open(os.path.join(out_dir, 'report.txt'), 'w').write(report + '\n')
    print(report)
    import csv
    with open(os.path.join(out_dir, 'per_unit.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(unit_rows[0].keys()))
        w.writeheader(); w.writerows(unit_rows)
    log('wrote %s' % out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
