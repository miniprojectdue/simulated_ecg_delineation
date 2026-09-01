"""ablation_conditioning.py - the two lead-conditioning ablations of Section 3.9.

Every unit is decoded twelve times, once under each lead query, with the input tensor held
fixed. only the FiLM conditioning index changes. Two tables come out of the same forwards:

  query sweep      how much the predicted landmarks MOVE across the twelve queries
                   (per-landmark spread in ms, and the fraction of units on which the
                   prediction changes at all) - conditioning that carried no information
                   would leave every landmark identical across queries.
  mismatched query the localisation error against each unit's own reference under the
                   CORRECT query against the mean over the eleven INCORRECT queries -
                   conditioning that carries lead-specific information makes the correct
                   query better than the wrong ones.

    python3 ml_modelling/scripts/ablation_conditioning.py \
        --checkpoint ml_modelling/checkpoints/finetune_toffset_fix_tailweight/best_geometry.pt \
        --units ml_modelling/data/finetune_units.csv --split test \
        --tag ablate_conditioning_indist --set data.num_workers=0

Writes per_unit_query.csv (one row per unit per query) and summary.txt under results/<tag>/.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E  # noqa: E402

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BOUNDARY = [n for n in E.LANDMARKS if n.endswith('_onset') or n.endswith('_offset')]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--units', required=True)
    parser.add_argument('--split', default='')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--device', default='')
    parser.add_argument('--max-units', type=int, default=0)
    args = parser.parse_args()

    device = E.pick_device(args.device or 'auto')
    model, cfg, _payload = E.load_checkpoint(args.checkpoint, device, None, args.set)
    out_dir = os.path.join(E.resolve(cfg['run'].get('log_root', 'ml_modelling/results')), args.tag)
    os.makedirs(out_dir, exist_ok=True)

    frame = E.load_units(args.units, split=(args.split or None), max_units=args.max_units)
    if frame.empty:
        raise SystemExit('no rows selected from %s' % args.units)
    fs_hz = float(frame['fs_hz'].iloc[0]) if 'fs_hz' in frame.columns else float(cfg['data'].get('fs_hz', 500))
    step_ms = 1000.0 / fs_hz

    dataset = E.BeatWindowDataset(frame, augment=None, training=False,
                                  **E.dataset_kwargs(cfg, training=False, seed=0, fs_hz=fs_hz))
    batch_size = args.batch_size or int(cfg['train'].get('eval_batch_size', 64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=int(cfg['data'].get('num_workers', 0) or 0),
                                         collate_fn=E.collate)
    E.log('scoring %d units over %d records under all twelve queries'
          % (len(dataset), frame['record_id'].nunique()))

    read_pad_ms = float(cfg['eval'].get('read_pad_ms', 0.0) or 0.0)
    read_pad = None if read_pad_ms < 0 else int(round(read_pad_ms * fs_hz / 1000.0))

    rows = []
    with torch.no_grad():
        for batch in loader:
            signal = batch['signal'].to(device, non_blocking=True)
            valid_t = batch['valid'].to(device, non_blocking=True) if 'valid' in batch else None
            readable = E.widen(batch['supervised'], batch.get('valid'), read_pad)
            truth = [E.landmarks_from_vector(v.numpy()) for v in batch['landmarks']]
            correct = batch['lead_idx'].numpy()

            for q in range(len(LEADS)):
                lead_q = torch.full_like(batch['lead_idx'], q).to(device)
                out = model(signal, lead_q, valid=valid_t)
                logits = out[0] if isinstance(out, tuple) else out
                predicted = E.batch_fiducials(logits, batch['trace'], cfg.get('postprocess'),
                                              supervised=batch['supervised'], readable=readable)
                for i, (p, t) in enumerate(zip(predicted, truth)):
                    meta = batch['meta'][i]
                    row = {'record_id': meta['record_id'], 'lead': meta['lead'],
                           'disease_class': meta['disease_class'],
                           'query_lead': LEADS[q], 'matched': int(q == int(correct[i]))}
                    for name in E.LANDMARKS:
                        pv, tv = p.get(name), t.get(name)
                        row['%s_pred' % name] = '' if pv is None else round(float(pv), 1)
                        row['%s_err_ms' % name] = '' if (pv is None or tv is None) \
                            else round((float(pv) - float(tv)) * step_ms, 3)
                    rows.append(row)

    per_unit = os.path.join(out_dir, 'per_unit_query.csv')
    with open(per_unit, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- summaries, computed from the rows just written --------------------------------------
    import pandas as pd
    d = pd.DataFrame(rows).replace('', np.nan)
    n_units = len(d) // len(LEADS)
    lines = ['Conditioning ablations  %s  (%d units, twelve queries each)' % (args.tag, n_units),
             'checkpoint %s' % args.checkpoint, '']

    lines.append('Mismatched-query test: MAE (ms) under the correct query against the mean over')
    lines.append('the eleven incorrect queries. A conditioning path carrying lead-specific')
    lines.append('information makes the matched column smaller.')
    lines.append('%-12s %10s %12s %8s' % ('landmark', 'matched', 'mismatched', 'ratio'))
    for name in E.LANDMARKS:
        col = pd.to_numeric(d['%s_err_ms' % name], errors='coerce')
        matched = col[d.matched == 1].abs().mean()
        mismatched = col[d.matched == 0].abs().mean()
        if np.isfinite(matched) and np.isfinite(mismatched):
            ratio = mismatched / matched if matched > 0 else np.inf
            lines.append('%-12s %10.2f %12.2f %8.2f' % (name, matched, mismatched, ratio))
    for label, names in (('boundaries', BOUNDARY), ('all landmarks', E.LANDMARKS)):
        cm = pd.concat([pd.to_numeric(d.loc[d.matched == 1, '%s_err_ms' % n], errors='coerce')
                        for n in names]).abs().mean()
        cw = pd.concat([pd.to_numeric(d.loc[d.matched == 0, '%s_err_ms' % n], errors='coerce')
                        for n in names]).abs().mean()
        lines.append('%-12s %10.2f %12.2f %8.2f' % (label, cm, cw, cw / cm if cm > 0 else np.inf))
    lines.append('')

    lines.append('Query sweep: per-landmark spread of the PREDICTED position across the twelve')
    lines.append('queries, input held fixed. SD is in ms, averaged over units; changed is the')
    lines.append('fraction of units on which any query moves the landmark at all.')
    lines.append('%-12s %10s %12s %10s' % ('landmark', 'mean SD', 'median SD', 'changed'))
    key = d.record_id.astype(str) + '|' + d.lead.astype(str)
    for name in E.LANDMARKS:
        pred = pd.to_numeric(d['%s_pred' % name], errors='coerce')
        g = pred.groupby(key)
        sd = g.std() * step_ms
        moved = g.apply(lambda v: v.dropna().nunique() > 1)
        if sd.notna().any():
            lines.append('%-12s %10.2f %12.2f %9.1f%%'
                         % (name, sd.mean(), sd.median(), 100.0 * moved.mean()))
    text = '\n'.join(lines)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as fh:
        fh.write(text + '\n')
    print(text)
    E.log('wrote %s and %s' % (per_unit, os.path.join(out_dir, 'summary.txt')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
