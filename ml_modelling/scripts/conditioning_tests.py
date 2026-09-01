#!/usr/bin/env python3
"""
conditioning_tests.py  -  do the FiLM lead query and the lead embedding carry information?

    python3 ml_modelling/scripts/conditioning_tests.py \
        --checkpoint ml_modelling/checkpoints/finetune_unet1d/best.pt \
        --units <external units csv> --tag primary

Removing the conditioning and retraining answers whether the model is better with the query than
without it. It does not answer whether the modulation path of Equation 3.14 is used at all. A
zero-initialised projection that never moves away from the identity would leave the two arms
indistinguishable without that being visible in either. These two tests address the mechanism
directly, cost no training, and exploit a property the external set has and the in-distribution
partition does not.

The external boundary reference is one criterion-defined set per recording, shared by all twelve
leads. So the twelve units of a recording can be given an identical input tensor and differ only
in which lead the query names. Any variation in the output is produced by the conditioning and by
nothing else.

    Test 1, query sweep. Hold the input fixed, sweep the query across all twelve leads, and
    measure the standard deviation of each predicted landmark across the twelve. A path the
    network ignores gives exactly zero at every landmark. A non-zero spread proves the query
    reaches the output. The ordering across landmarks is a second and independent check, since
    the T offset is the most lead-dependent boundary of the beat and the QRS onset among the
    least, so a mechanism carrying real lead information should show more spread at the former.

    Test 2, mismatched query. Score each unit with the query naming its own lead and again with a
    query naming each of the eleven others. Conditioning that carries lead-specific information
    must degrade under a wrong query. Conditioning that is decorative will not. This is the
    adversarial form and the one that is hard to pass by accident. The degradation is resolved by
    lead pair, since if the embedding has learned something corresponding to lead geometry then
    electrically adjacent leads should cost less than distant ones.

The two tests belong on different surfaces and the reason is the same property, read twice.

    Test 1 needs the external set. Its twelve units per recording can be handed a genuinely
    identical input tensor, so the output spread is attributable to the query alone.

    Test 2 needs the in-distribution set. There every unit carries its own per-lead reference,
    so naming the wrong lead is genuinely wrong and the degradation means something. On the
    external set the boundary reference is shared across all twelve leads, so a wrong query is
    not wrong for a boundary and the test would measure nothing. Running test 2 there produces an
    antisymmetric lead-pair matrix whose mean cancels to zero, which is the signature of the
    reference being shared rather than of conditioning being absent.

Use --test sweep on the external units and --test mismatch on the in-distribution units.

"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
BOUNDARIES = ['qrs_onset', 'qrs_offset', 't_onset', 't_offset']


def repo_root(start=HERE):
    d = start
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, 'config', 'paths.yaml')):
            return d
        d = os.path.dirname(d)
    raise SystemExit('could not find the repository root above %s' % start)


def _mk(root, out_root, tag):
    d = os.path.join(root, out_root, tag + '_conditioning')
    os.makedirs(d, exist_ok=True)
    return d


def merge_out(root, out_root, tag, payload):
    """Update conditioning.json in place rather than replacing it.

    The two tests run as separate invocations and belong on different surfaces, so writing the
    whole file each time silently discards whichever ran first. Only the keys this invocation
    actually computed are touched.
    """
    d = _mk(root, out_root, tag)
    path = os.path.join(d, 'conditioning.json')
    existing = {}
    if os.path.isfile(path):
        try:
            existing = json.load(open(path))
        except ValueError:
            existing = {}
    existing.update(payload)
    json.dump(existing, open(path, 'w'), indent=2)
    return d


def main():
    root = repo_root()
    sys.path.insert(0, HERE)

    import dataset as D
    from common import LANDMARKS, load_config, log, pick_device
    from model import build_model
    from postprocess import batch_fiducials

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--units', required=True, help='the external units table')
    p.add_argument('--tag', default='conditioning')
    p.add_argument('--device', default='')
    p.add_argument('--max-records', type=int, default=0)
    p.add_argument('--out-root', default='ml_modelling/results')
    p.add_argument('--test', choices=['sweep', 'mismatch', 'both'], default='both',
                   help='sweep belongs on the external set, mismatch on the in-distribution set')
    a = p.parse_args()

    device = torch.device(a.device) if a.device else pick_device('auto')
    payload = torch.load(os.path.join(root, a.checkpoint) if not os.path.isabs(a.checkpoint)
                         else a.checkpoint, map_location=device, weights_only=False)
    cfg = payload.get('config')
    if cfg is None:
        raise SystemExit('checkpoint carries no config')
    model = build_model(cfg).to(device)
    model.load_state_dict(payload['model'])
    model.eval()
    log('loaded %s on %s' % (a.checkpoint, device))

    frame = D.load_units(a.units)
    if a.max_records:
        keep = sorted(frame['record_id'].unique())[:a.max_records]
        frame = frame[frame['record_id'].isin(set(keep))].reset_index(drop=True)
    data = cfg['data']
    ds = D.BeatWindowDataset(
        frame, augment=None, training=False,
        **D.dataset_kwargs(cfg, training=False, seed=0, allow_csv_fallback=True))
    step_ms = 1000.0 / float(data.get('fs_hz', 500.0))
    pcfg = cfg.get('postprocess', {})

    def predict(item, lead_override):
        """Run one item with the query forced to a chosen lead. Returns landmark positions."""
        x = item['signal'].unsqueeze(0).to(device)
        idx = torch.tensor([int(lead_override)], dtype=torch.long, device=device)
        val = item['valid'].unsqueeze(0).to(device) if 'valid' in item else None
        with torch.no_grad():
            # The dual-head model returns a pair. Passing the pair straight on would reach the
            # numpy branch of batch_fiducials and fail on the device rather than on the shape.
            out_m = model(x, idx, valid=val)
        logits = out_m[0] if isinstance(out_m, tuple) else out_m
        sup = item['supervised'].unsqueeze(0).numpy()
        out = batch_fiducials(logits, item['trace'].unsqueeze(0).numpy(), pcfg, sup)[0]
        return out

    # ---------------- Test 1, query sweep on a fixed input -----------------------------------
    by_record = {}
    for i in range(len(ds)):
        rec = str(ds.records[i]['record_id'])
        by_record.setdefault(rec, []).append(i)

    sweep = {name: [] for name in LANDMARKS}
    for rec, idxs in (by_record.items() if a.test in ('sweep', 'both') else []):
        item = ds[idxs[0]]                       # one input tensor for the whole recording
        got = {name: [] for name in LANDMARKS}
        for lead in range(len(LEADS)):
            out = predict(item, lead)
            for name in LANDMARKS:
                v = out.get(name)
                if v is not None:
                    got[name].append(float(v))
        for name in LANDMARKS:
            if len(got[name]) >= 2:
                sweep[name].append(float(np.std(got[name], ddof=1)) * step_ms)

    print('\n' + '=' * 70)
    print('TEST 1  query sweep, identical input, twelve different queries')
    print('=' * 70)
    print('%-12s %10s %10s %10s' % ('landmark', 'median SD', 'mean SD', 'n records'))
    sweep_summary = {}
    for name in LANDMARKS:
        v = np.array(sweep[name], dtype=float)
        if v.size:
            sweep_summary[name] = {'median_sd_ms': float(np.median(v)),
                                   'mean_sd_ms': float(np.mean(v)), 'n': int(v.size)}
            print('%-12s %10.2f %10.2f %10d' % (name, np.median(v), np.mean(v), v.size))
        else:
            print('%-12s %10s %10s %10d' % (name, '--', '--', 0))
    nonzero = [k for k, s in sweep_summary.items() if s['median_sd_ms'] > 0]
    print('\n  landmarks with a non-zero across-query spread: %d of %d'
          % (len(nonzero), len(sweep_summary)))
    print('  a conditioning path the network ignores would give exactly zero everywhere')

    # ---------------- Test 2, mismatched query ------------------------------------------------
    matched, mismatched = [], []
    pair = np.full((12, 12), np.nan)
    pair_n = np.zeros((12, 12))
    for i in (range(len(ds)) if a.test in ('mismatch', 'both') else []):
        item = ds[i]
        true_lead = int(item['lead_idx'])
        marks = item['landmarks'].numpy()
        base = predict(item, true_lead)

        def boundary_mae(out):
            errs = []
            for name in BOUNDARIES:
                j = LANDMARKS.index(name)
                t = marks[j]
                v = out.get(name)
                if v is not None and np.isfinite(t):
                    errs.append(abs(float(v) - float(t)) * step_ms)
            return float(np.mean(errs)) if errs else np.nan

        m0 = boundary_mae(base)
        if not np.isfinite(m0):
            continue
        matched.append(m0)
        for other in range(len(LEADS)):
            if other == true_lead:
                continue
            m1 = boundary_mae(predict(item, other))
            if np.isfinite(m1):
                mismatched.append(m1 - m0)
                d = m1 - m0
                pair[true_lead, other] = d if np.isnan(pair[true_lead, other]) else pair[true_lead, other] + d
                pair_n[true_lead, other] += 1

    matched = np.array(matched)
    mismatched = np.array(mismatched)
    if mismatched.size == 0:
        print('\n(mismatch test not run)')
        d = merge_out(root, a.out_root, a.tag, {
            'checkpoint': a.checkpoint,
            'sweep_units': a.units,
            'query_sweep': sweep_summary,
        })
        print('written to %s' % os.path.relpath(d, root))
        return
    print('\n' + '=' * 70)
    print('TEST 2  mismatched query, the same unit scored under a wrong lead')
    print('=' * 70)
    print('  boundary MAE, matched query      %7.2f ms  (n=%d)' % (matched.mean(), matched.size))
    print('  degradation under a wrong query  %+7.2f ms  (n=%d comparisons)'
          % (mismatched.mean(), mismatched.size))
    print('  proportion of wrong queries that made it worse: %.1f%%' % (100 * (mismatched > 0).mean()))
    boot = np.random.default_rng(1337).choice(mismatched, size=(4000, mismatched.size))
    lo, hi = np.percentile(boot.mean(axis=1), [2.5, 97.5])
    print('  95%% interval on the degradation  [%+.2f, %+.2f]' % (lo, hi))
    print('  a decorative conditioning path would give a degradation of zero')

    with np.errstate(invalid='ignore'):
        pair_mean = np.where(pair_n > 0, pair / np.maximum(pair_n, 1), np.nan)
    print('\n  degradation by lead pair, rows are the true lead, columns the query given')
    print('      ' + ''.join('%7s' % l for l in LEADS))
    for i, l in enumerate(LEADS):
        print('%5s ' % l + ''.join('%7s' % ('--' if np.isnan(pair_mean[i, j]) else '%.1f' % pair_mean[i, j])
                                   for j in range(12)))

    payload = {
        'checkpoint': a.checkpoint,
        'mismatch_units': a.units,
        'mismatched': {
            'matched_boundary_mae_ms': float(matched.mean()),
            'degradation_ms': float(mismatched.mean()),
            'ci95': [float(lo), float(hi)],
            'fraction_worse': float((mismatched > 0).mean()),
            'n_units': int(matched.size), 'n_comparisons': int(mismatched.size),
            'by_lead_pair': [[None if np.isnan(v) else float(v) for v in row] for row in pair_mean],
            'leads': LEADS,
        },
    }
    if sweep_summary:
        payload['query_sweep'] = sweep_summary
        payload['sweep_units'] = a.units
    out_dir = merge_out(root, a.out_root, a.tag, payload)
    print('\nwritten to %s' % os.path.relpath(out_dir, root))


if __name__ == '__main__':
    main()
