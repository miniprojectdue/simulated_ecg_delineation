#!/usr/bin/env python3
"""
evaluate.py  -  score a trained checkpoint on a units table.

The intended target is the external MonoAlg3D set, which lives outside this repository so that
no part of it can leak into training. Point it at any table that carries the same columns as
ml_modelling/data/pretrain_units.csv.

    python ml_modelling/scripts/evaluate.py \
        --checkpoint ml_modelling/checkpoints/finetune_unet1d/best.pt \
        --units /path/to/monoalg3d_units.csv \
        --tag finetuned_on_external

Outputs, all under ml_modelling/results/<tag>/
    metrics.json        every number, machine readable
    report.txt          the formatted table that goes into the results chapter
    per_unit.csv        one row per unit with the signed error of each landmark, for the plots
                        and for any per disease class breakdown

The checkpoint carries the config it was trained with, so the crop length, the normalisation
and the post-processing thresholds are recovered rather than guessed. Anything passed with
--set overrides them, which is how a post-processing sensitivity sweep is run without retraining.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biomarkers import BIOMARKER_NAMES, biomarker_errors, summarise_biomarker_errors  # noqa: E402
from common import LANDMARKS, ROOT, apply_overrides, load_config, log, pick_device, save_json, N_LEADS  # noqa: E402
from dataset import BeatWindowDataset, dataset_kwargs, collate, load_units  # noqa: E402
from metrics import (  # noqa: E402
    ConfusionAccumulator, FiducialAccumulator, format_report, landmarks_from_vector,
)
from model import build_model  # noqa: E402
from postprocess import batch_fiducials, enforce_order  # noqa: E402


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_checkpoint(path, device, config_override=None, sets=None):
    payload = torch.load(resolve(path), map_location=device)
    cfg = payload.get('config')
    if config_override:
        cfg = load_config(config_override, sets)
    elif cfg is None:
        raise SystemExit('%s carries no config, so pass --config as well' % path)
    else:
        cfg = apply_overrides(dict(cfg), sets)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    log('loaded %s, trained to epoch %s with %s %s'
        % (path, payload.get('epoch'), payload.get('monitor'), payload.get('value')))
    return model, cfg, payload



MAG_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def qrs_anchor(trace_all, valid):
    """A ventricular anchor read from the raw signal, using no region prediction.

    The steepest point of the spatial magnitude curve over the observed samples. It is derived
    from the same curve the network receives as its thirteenth channel, so it is independent of
    the network's computation rather than of its input, and that distinction belongs in the
    write-up. Its purpose is to notice when the predicted QRS is not where the ventricular
    activation is, which is the signature of an identity rotation and is invisible to an
    ordering check.
    """
    x = np.asarray(trace_all, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < max(MAG_LEADS) + 1:
        return None
    v = np.asarray(valid, dtype=bool)
    if v.sum() < 8:
        return None
    if x.shape[0] >= 13:
        # channel 13 already holds the curve. Normalisation is a per-channel affine map, so the
        # steepest point of the scaled curve is the steepest point of the raw one.
        m = x[12]
    else:
        base = np.median(x[MAG_LEADS][:, v], axis=1, keepdims=True)
        m = np.sqrt(((x[MAG_LEADS] - base) ** 2).sum(axis=0))
    d = np.abs(np.diff(m))
    # A transition from the last valid sample into padding is not a ventricular slope.
    d[~(v[:-1] & v[1:])] = -1.0
    return int(np.argmax(d))


def widen(supervised, valid, pad_samples):
    """The mask a predicted region may be followed into, given the indexed beat's window.

    The window has two jobs and they are not the same job. It says which beat is being asked
    about, and until now it also decided how far that beat's waves were allowed to run, since
    the region search was given the window and nothing else. A wave that ends past the window
    edge was therefore cut at the edge and the cut was reported as the model's error. On the
    external corpus the window ended twenty samples past the last annotated landmark and the
    T offset error piled up at exactly forty milliseconds on the majority of units, which is
    that edge and not a property of the model.

    Widening the crop instead is not a fix. The crop is placed by centring the window in it, so
    a wider window moves the beat away from the position the network was trained to expect, and
    the prediction moves with it.

    pad_samples None means the whole of the observed record, which is the right setting for a
    diagnostic. It removes every limit on how far a region may be followed and leaves the window
    doing only the job it can defend, which is naming the beat. Any remaining prediction sitting
    on the record edge is then the model failing to close a wave rather than a limit we imposed.

    Returns None when no widening is asked for, which is the earlier behaviour exactly.
    """
    if pad_samples is None:
        if valid is None:
            return None
        v = valid.numpy() if hasattr(valid, 'numpy') else np.asarray(valid)
        return v.astype(bool)
    if pad_samples <= 0:
        return None
    sup = supervised.numpy() if hasattr(supervised, 'numpy') else np.asarray(supervised)
    sup = sup.astype(bool)
    out = np.zeros_like(sup)
    length = sup.shape[1]
    for i in range(sup.shape[0]):
        idx = np.flatnonzero(sup[i])
        if idx.size == 0:
            continue
        lo = max(0, int(idx[0]) - pad_samples)
        hi = min(length - 1, int(idx[-1]) + pad_samples)
        out[i, lo:hi + 1] = True
    if valid is not None:
        v = valid.numpy() if hasattr(valid, 'numpy') else np.asarray(valid)
        out &= v.astype(bool)
    return out


def edge_of(mask_row):
    """The last sample the mask admits, or None when it admits nothing."""
    idx = np.flatnonzero(np.asarray(mask_row).astype(bool))
    return None if idx.size == 0 else int(idx[-1])


@torch.no_grad()
def run(model, loader, cfg, device, fs_hz):
    confusion = ConfusionAccumulator(n_classes=int(cfg['model'].get('n_classes', 4)))
    fiducial = FiducialAccumulator(tolerances_ms=cfg['eval'].get('tolerance_ms', [10, 25, 50, 75, 150]),
                                   primary_ms=cfg['eval'].get('primary_tolerance_ms', 25),
                                   fs_hz=fs_hz)
    biomarker_rows, unit_rows = [], []
    step_ms = 1000.0 / float(fs_hz)
    read_pad_ms = float(cfg['eval'].get('read_pad_ms', 0.0) or 0.0)
    # A negative value means the whole observed record, with no limit at all on how far a region
    # may be followed. That is the setting a diagnostic wants, since any limit we impose comes
    # back as error and cannot be told apart from the model's own.
    read_pad = None if read_pad_ms < 0 else int(round(read_pad_ms * float(fs_hz) / 1000.0))
    if read_pad is None:
        log('reading regions over the whole observed record. The window only names the beat.')
    elif read_pad > 0:
        log('reading regions up to %d samples past the window on each side, clipped to the '
            'observed record. The window still decides which beat is answered.' % read_pad)

    identity_scope = str(cfg.get('eval', {}).get('identity_scope', 'window')).lower()
    if identity_scope not in ('window', 'valid'):
        raise ValueError('eval.identity_scope %r is not one of window or valid' % identity_scope)
    if identity_scope == 'valid':
        log('selecting the single beat over the observed record, without using its label window.')

    for batch in loader:
        signal = batch['signal'].to(device, non_blocking=True)
        lead_idx = batch['lead_idx'].to(device, non_blocking=True)
        valid_t = batch['valid'].to(device, non_blocking=True) if 'valid' in batch else None
        out = model(signal, lead_idx, valid=valid_t)
        logits, p_logit = out if isinstance(out, tuple) else (out, None)
        p_prob = (torch.sigmoid(p_logit).cpu().numpy() if p_logit is not None
                  else np.full(len(signal), np.nan))

        confusion.update(logits.argmax(dim=1).cpu().numpy(), batch['target'].numpy())
        readable = widen(batch['supervised'], batch.get('valid'), read_pad)
        identity = batch['valid'] if identity_scope == 'valid' else batch['supervised']
        predicted = batch_fiducials(logits, batch['trace'], cfg.get('postprocess'),
                                    supervised=identity, readable=readable)
        truth = [landmarks_from_vector(v.numpy()) for v in batch['landmarks']]
        fiducial.update(predicted, truth)

        for i, (p, t) in enumerate(zip(predicted, truth)):
            biomarker_rows.append(biomarker_errors(p, t, fs_hz))
            meta = batch['meta'][i]
            row = {
                'record_id': meta['record_id'], 'lead': meta['lead'], 'beat_id': meta['beat_id'],
                'disease_class': meta['disease_class'], 'crop_start': meta['crop_start'],
                'qrs_pattern_pred': p.get('qrs_pattern', ''),
                'order_ok_pred': enforce_order(p)['order_ok'],
            }
            # the three independent statements about this unit, kept side by side so a
            # disagreement between them is visible in the per-unit table rather than inferred
            anchor = qrs_anchor(batch['signal'][i].numpy(), batch['valid'][i].numpy())
            qo, qf = p.get('qrs_onset'), p.get('qrs_offset')
            row['p_head_prob'] = '' if not np.isfinite(p_prob[i]) else round(float(p_prob[i]), 4)
            row['seg_emits_p'] = int(p.get('p_onset') is not None)
            row['qrs_anchor'] = '' if anchor is None else anchor
            row['anchor_in_qrs'] = '' if (anchor is None or qo is None or qf is None) \
                else int(float(qo) <= anchor <= float(qf))
            # Where the search was allowed to stop. A predicted offset sitting exactly on one of
            # these is a truncation rather than a reading, and the count of them is printed with
            # the report so a capped run cannot be mistaken for a measured one.
            win_edge = edge_of(batch['supervised'][i].numpy())
            read_edge = win_edge if readable is None else edge_of(readable[i])
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


def geometry_report(unit_rows, crop_length):
    """Two ways a run can be invalid while every printed metric still looks like a measurement.

    The first is placement. The crop is built by centring the labelled window in it, so a window
    that is the beat span puts the beat where every training unit put it. A window with a long
    empty stretch on one side pushes the beat off to the other, the network answers where it was
    trained to look rather than where the beat is, and the errors come back in the hundreds of
    milliseconds. That happened on this corpus and nothing in the report said so. The quantity
    that separates the two cases is the slack, meaning how much of the window lies past the last
    landmark. The training corpus runs at a median of 16 samples and a beat span window at 20,
    while the whole record window that broke the run sat at 319.

    The second is truncation. A predicted offset that lands exactly on the last sample the search
    was allowed to see is a cut, not a reading, and averaging cuts into an error gives a number
    that is a property of the table.
    """
    offsets, slack = [], []
    on_window = beyond_window = on_read = beyond_read = n_t = 0
    for r in unit_rows:
        a, b = r.get('qrs_onset_true', ''), r.get('t_offset_true', '')
        if a != '' and b != '':
            middle = (float(a) + float(b)) / 2.0 - float(r['crop_start'])
            offsets.append(middle - crop_length / 2.0)
        if b != '' and r.get('window_end', '') != '':
            slack.append(float(r['window_end']) - float(b))
        off = r.get('t_offset_pred', '')
        if off == '':
            continue
        n_t += 1
        if r.get('window_end', '') != '':
            delta = float(off) - float(r['window_end'])
            on_window += int(delta == 0)
            beyond_window += int(delta > 0)
        if r.get('read_end', '') != '':
            delta = float(off) - float(r['read_end'])
            on_read += int(delta == 0)
            beyond_read += int(delta > 0)
    out = {'n_t_offset': n_t, 'on_window_edge': on_window,
           'past_window_edge': beyond_window, 'on_read_edge': on_read,
           'past_read_edge': beyond_read,
           'window_slack_samples': float(np.median(slack)) if slack else float('nan'),
           'beat_centre_offset_samples': float(np.median(offsets)) if offsets else float('nan'),
           'beat_centre_offset_iqr': (float(np.subtract(*np.percentile(offsets, [75, 25])))
                                      if len(offsets) > 3 else float('nan'))}
    lines = ['geometry and truncation',
             '  window runs %.0f samples past the last landmark (median)'
             % out['window_slack_samples'],
             '  beat centre sits %.0f samples from the crop centre, IQR %.0f'
             % (out['beat_centre_offset_samples'], out['beat_centre_offset_iqr']),
             '  predicted T offset on / past window edge  %d / %d of %d'
             % (on_window, beyond_window, n_t),
             '  predicted T offset on / past read edge    %d / %d of %d'
             % (on_read, beyond_read, n_t)]
    if slack and out['window_slack_samples'] > 150:
        lines.append('  WARNING the window carries a long post-landmark tail. Under the default '
                     'window-centred placement this changes crop geometry; use a label-free '
                     'record placement before interpreting transfer performance.')
    if n_t and on_read > 0.05 * n_t:
        lines.append('  WARNING %d T offsets are sitting on the edge of what the search could '
                     'see, so their error is the edge and not a reading. Raise eval.read_pad_ms.'
                     % on_read)
    out['report'] = '\n'.join(lines)
    return out


def by_group(unit_rows, key, landmarks=None):
    """Boundary MAE and count per disease class or per lead, for the breakdown tables."""
    landmarks = landmarks or [n for n in LANDMARKS if n.endswith('_onset') or n.endswith('_offset')]
    groups = {}
    for row in unit_rows:
        bucket = groups.setdefault(row.get(key, ''), [])
        values = [row['%s_err_ms' % name] for name in landmarks if row['%s_err_ms' % name] != '']
        bucket.extend(float(v) for v in values)
    return {name: {'n_landmarks': len(values),
                   'bias_ms': float(np.mean(values)) if values else float('nan'),
                   'mae_ms': float(np.mean(np.abs(values))) if values else float('nan')}
            for name, values in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--units', required=True, help='the units table to score, normally the external set')
    parser.add_argument('--split', default='', help='restrict to one split value, empty means all rows')
    parser.add_argument('--tag', default='', help='names the output folder, defaults to the checkpoint name')
    parser.add_argument('--config', default='', help='use this config instead of the one in the checkpoint')
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value')
    parser.add_argument('--batch-size', type=int, default=0)
    parser.add_argument('--device', default='')
    parser.add_argument('--max-units', type=int, default=0)
    parser.add_argument('--record-split', default='',
                        help='a csv of record_id,external_split, used with --record-split-value')
    parser.add_argument('--record-split-value', default='',
                        help="'dev' or 'holdout', restricts scoring to those records")
    args = parser.parse_args()

    device = pick_device(args.device or 'auto')
    model, cfg, payload = load_checkpoint(args.checkpoint, device, args.config or None, args.set)

    tag = args.tag or (os.path.basename(os.path.dirname(resolve(args.checkpoint))) + '_eval')
    out_dir = os.path.join(resolve(cfg['run'].get('log_root', 'ml_modelling/results')), tag)
    os.makedirs(out_dir, exist_ok=True)

    iso = cfg.get('eval', {}).get('isolate_beat_ms')
    if iso is not None and float(iso) >= 0:
        log('DIAGNOSTIC, every unit is scored under the external observation geometry. Its '
            'neighbouring beats are flattened away and %.0f ms of quiet tail is left behind it. '
            'The reference landmarks are untouched.' % float(iso))

    frame = load_units(args.units, split=(args.split or None), max_units=args.max_units)
    if args.record_split and args.record_split_value:
        import pandas as _pd
        keep = _pd.read_csv(args.record_split)
        keep = set(keep.loc[keep.external_split == args.record_split_value, 'record_id'].astype(str))
        frame = frame[frame['record_id'].astype(str).isin(keep)].reset_index(drop=True)
        log('restricted to the %s half, %d records' % (args.record_split_value, len(keep)))
    if frame.empty:
        raise SystemExit('no rows selected from %s' % args.units)
    fs_hz = float(frame['fs_hz'].iloc[0]) if 'fs_hz' in frame.columns else float(cfg['data'].get('fs_hz', 500))

    dataset = BeatWindowDataset(
        frame, augment=None, training=False,
        **dataset_kwargs(cfg, training=False, seed=0, fs_hz=fs_hz))

    # Fail here rather than inside a convolution. The model's channel count comes from the
    # checkpoint and the loader's comes from the config, and a key that reaches one and not the
    # other used to surface as an unreadable shape error several frames deep.
    want = int(cfg.get('model', {}).get('in_channels', N_LEADS))
    got = int(dataset[0]['signal'].shape[0])
    if got != want:
        raise SystemExit('the loader delivers %d channels and the checkpoint expects %d. '
                         'Check data.magnitude_channel against model.in_channels in the config '
                         'carried by this checkpoint.' % (got, want))
    batch_size = args.batch_size or int(cfg['train'].get('eval_batch_size', 64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=int(cfg['data'].get('num_workers', 0) or 0),
                                         collate_fn=collate)
    log('scoring %d units over %d records from %s'
        % (len(dataset), frame['record_id'].nunique(), args.units))

    seg, fid, bio, unit_rows = run(model, loader, cfg, device, fs_hz)
    geometry = geometry_report(unit_rows, int(cfg['data']['crop_length']))

    metrics = {
        'checkpoint': resolve(args.checkpoint),
        'checkpoint_epoch': payload.get('epoch'),
        'units_csv': args.units,
        'n_units': len(dataset),
        'n_records': int(frame['record_id'].nunique()),
        'fs_hz': fs_hz,
        'segmentation': seg,
        'fiducials': fid,
        'biomarkers': bio,
        'by_disease_class': by_group(unit_rows, 'disease_class'),
        'by_lead': by_group(unit_rows, 'lead'),
        'postprocess': cfg.get('postprocess'),
        'read_pad_ms': float(cfg['eval'].get('read_pad_ms', 0.0) or 0.0),
        'identity_scope': str(cfg['eval'].get('identity_scope', 'window')),
        'record_start_at_sample': cfg['eval'].get('record_start_at_sample'),
        'isolate_beat_ms': (None if cfg['eval'].get('isolate_beat_ms') is None
                            else float(cfg['eval']['isolate_beat_ms'])),
        'geometry': {k: v for k, v in geometry.items() if k != 'report'},
    }
    save_json(metrics, os.path.join(out_dir, 'metrics.json'))

    report = format_report(seg, fid, bio, title='%s on %s' % (tag, os.path.basename(args.units)))
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
    log('wrote %s, %s and %s' % (os.path.join(out_dir, 'metrics.json'),
                                 os.path.join(out_dir, 'report.txt'), per_unit))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
