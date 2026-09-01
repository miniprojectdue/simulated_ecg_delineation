#!/usr/bin/env python3
"""evaluate_ensemble.py - the posterior-ensemble ablation of Section 3.9.

Loads several stage-two checkpoints, averages their class posteriors per sample, and scores
the averaged posterior through exactly the evaluation path of evaluate.py: the same dataset,
the same decoding, the same report. The averaged posterior is passed downstream as
log(mean p), which preserves it under any later softmax and gives the same argmax, so the
post-processing sees the ensemble the way Section 3.9 defines it: the mean posterior taken
BEFORE decoding. The auxiliary P-observability logits are averaged the same way.

    python3 ml_modelling/scripts/evaluate_ensemble.py \
        --checkpoints ckptA.pt ckptB.pt ckptC.pt \
        --units ml_modelling/data/finetune_units.csv --split test \
        --tag ablate_ensemble_indist --set data.num_workers=0

All --set overrides are applied to every checkpoint's config; the first checkpoint's config
drives the dataset. Checkpoints must share the input-channel count.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402
from common import N_LEADS  # noqa: E402


class PosteriorEnsemble(nn.Module):
    """Mean class posterior over member models, returned as log-probabilities."""

    def __init__(self, models):
        super(PosteriorEnsemble, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, signal, lead_idx, valid=None):
        probs, p_logits = [], []
        for model in self.models:
            out = model(signal, lead_idx, valid=valid)
            logits, p_logit = out if isinstance(out, tuple) else (out, None)
            probs.append(torch.softmax(logits, dim=1))
            if p_logit is not None:
                p_logits.append(p_logit)
        mean_prob = torch.stack(probs).mean(dim=0).clamp_min(1e-9)
        pseudo_logits = torch.log(mean_prob)
        if len(p_logits) == len(self.models) and p_logits:
            return pseudo_logits, torch.stack(p_logits).mean(dim=0)
        return pseudo_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--units', required=True)
    parser.add_argument('--split', default='', help='restrict to one split value, empty means all rows')
    parser.add_argument('--tag', required=True, help='names the output folder under results/')
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--device', default='')
    parser.add_argument('--max-units', type=int, default=0)
    args = parser.parse_args()

    device = E.pick_device(args.device or 'auto')
    models, cfg = [], None
    for path in args.checkpoints:
        model, model_cfg, _payload = E.load_checkpoint(path, device, None, args.set)
        models.append(model)
        if cfg is None:
            cfg = model_cfg
        elif int(model_cfg.get('model', {}).get('in_channels', N_LEADS)) != \
                int(cfg.get('model', {}).get('in_channels', N_LEADS)):
            raise SystemExit('%s disagrees with the first checkpoint on in_channels' % path)
    ensemble = PosteriorEnsemble(models).to(device).eval()
    E.log('posterior ensemble over %d checkpoints' % len(models))

    out_dir = os.path.join(E.resolve(cfg['run'].get('log_root', 'ml_modelling/results')), args.tag)
    os.makedirs(out_dir, exist_ok=True)

    frame = E.load_units(args.units, split=(args.split or None), max_units=args.max_units)
    if frame.empty:
        raise SystemExit('no rows selected from %s' % args.units)
    fs_hz = float(frame['fs_hz'].iloc[0]) if 'fs_hz' in frame.columns else float(cfg['data'].get('fs_hz', 500))

    dataset = E.BeatWindowDataset(frame, augment=None, training=False,
                                  **E.dataset_kwargs(cfg, training=False, seed=0, fs_hz=fs_hz))
    want = int(cfg.get('model', {}).get('in_channels', N_LEADS))
    got = int(dataset[0]['signal'].shape[0])
    if got != want:
        raise SystemExit('the loader delivers %d channels and the checkpoints expect %d' % (got, want))
    batch_size = args.batch_size or int(cfg['train'].get('eval_batch_size', 64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=int(cfg['data'].get('num_workers', 0) or 0),
                                         collate_fn=E.collate)
    E.log('scoring %d units over %d records from %s'
          % (len(dataset), frame['record_id'].nunique(), args.units))

    seg, fid, bio, unit_rows = E.run(ensemble, loader, cfg, device, fs_hz)
    geometry = E.geometry_report(unit_rows, int(cfg['data']['crop_length']))

    metrics = {
        'checkpoints': [E.resolve(p) for p in args.checkpoints],
        'ensemble': 'mean posterior before post-processing',
        'units_csv': args.units,
        'n_units': len(dataset),
        'n_records': int(frame['record_id'].nunique()),
        'fs_hz': fs_hz,
        'segmentation': seg,
        'fiducials': fid,
        'biomarkers': bio,
        'by_disease_class': E.by_group(unit_rows, 'disease_class'),
        'by_lead': E.by_group(unit_rows, 'lead'),
        'postprocess': cfg.get('postprocess'),
        'geometry': {k: v for k, v in geometry.items() if k != 'report'},
    }
    E.save_json(metrics, os.path.join(out_dir, 'metrics.json'))

    report = E.format_report(seg, fid, bio, title='%s (posterior ensemble of %d) on %s'
                             % (args.tag, len(models), os.path.basename(args.units)))
    report = report + '\n\n' + geometry['report']
    with open(os.path.join(out_dir, 'report.txt'), 'w') as fh:
        fh.write(report + '\n')
    print(report)

    import csv
    per_unit = os.path.join(out_dir, 'per_unit.csv')
    with open(per_unit, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(unit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unit_rows)
    E.log('wrote %s, %s and %s' % (os.path.join(out_dir, 'metrics.json'),
                                   os.path.join(out_dir, 'report.txt'), per_unit))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
