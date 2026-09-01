#!/usr/bin/env python3
"""
preflight.py  -  prove every setting in the config reached the code, before a long run starts.

    python3 ml_modelling/scripts/preflight.py --config ml_modelling/configs/pretrain.yaml

A smoke test proves the plumbing runs. It does not prove that any particular setting arrived,
and a key that never reaches the code fails silently and reproduces a different run over many
hours. This checks the four things that can go wrong that way and refuses if any has.

    model         parameter count and the presence of both attention paths
    input         thirteen channels, and the thirteenth is the magnitude curve
    target        the transitions are Gaussian at the configured widths rather than instantaneous
    augmentation  each structural transform fires at roughly its configured probability

"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--n', type=int, default=256, help='items to draw for the rate checks')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--set', action='append', default=[], metavar='section.key=value',
                   dest='overrides', help='same override syntax as train.py')
    a = p.parse_args()

    import dataset as D
    import losses as L
    import model as M
    from common import load_config

    try:
        from common import apply_overrides
        cfg = apply_overrides(load_config(a.config), a.overrides) if a.overrides \
            else load_config(a.config)
    except ImportError:
        cfg = load_config(a.config)
        for item in a.overrides:
            key, _, value = item.partition('=')
            node, parts = cfg, key.split('.')
            for k in parts[:-1]:
                node = node.setdefault(k, {})
            low = value.strip().lower()
            node[parts[-1]] = (True if low == 'true' else False if low == 'false'
                               else int(value) if value.lstrip('-').isdigit()
                               else float(value) if value.replace('.', '', 1).lstrip('-').isdigit()
                               else value)
    fail = []
    print('config  %s\n' % a.config)

    # ---- 1. model -----------------------------------------------------------------------
    net = M.build_model(cfg)
    n = M.count_parameters(net)['total']
    att = cfg.get('model', {}).get('attention') or {}
    print('MODEL')
    print('  parameters                %d' % n)
    print('  bottleneck attention      %s' % ('present, %d heads, neighborhood %s, position %s, mask invalid %s'
          % (net.bottleneck_attn.heads, net.bottleneck_attn.neighborhood,
             net.bottleneck_attn.position_encoding, net.bottleneck_attn.mask_invalid)
          if net.bottleneck_attn is not None else 'ABSENT'))
    print('  skip gates                %s' % ('%d gates' % len(net.skip_gates)
          if net.skip_gates is not None else 'ABSENT'))
    if att.get('bottleneck') and net.bottleneck_attn is None:
        fail.append('the config asks for bottleneck attention and the model has none')
    if att.get('skip_gates') and net.skip_gates is None:
        fail.append('the config asks for skip gates and the model has none')

    # ---- 2. the loader ------------------------------------------------------------------
    train_loader = D.build_loaders(cfg)[0]
    ds = train_loader.dataset
    item = ds[0]
    sig = item['signal'].numpy()
    want_ch = int(cfg['model'].get('in_channels', 12))
    print('\nINPUT')
    print('  channels delivered        %d, model expects %d' % (sig.shape[0], want_ch))
    if sig.shape[0] != want_ch:
        fail.append('the loader delivers %d channels and the model expects %d'
                    % (sig.shape[0], want_ch))
    if bool(cfg['data'].get('magnitude_channel', False)):
        if not getattr(ds, 'magnitude_channel', False):
            fail.append('data.magnitude_channel is set and the loader did not receive it')
        else:
            # Channel order is twelve leads, then the magnitude curve, then the validity mask.
            # Reaching for the last channel picks up the mask, which is constant one over the
            # observed samples and has no correlation with anything, so the index is explicit.
            v = item['valid'].numpy().astype(bool)
            last, leads = sig[12][v], sig[:12][:, v]
            cors = []
            for i in range(12):
                if leads[i].std() > 1e-9 and last.std() > 1e-9:
                    cors.append(abs(float(np.corrcoef(last, leads[i])[0, 1])))
            same = max(cors) if cors else 0.0
            if not cors:
                fail.append('every lead was constant over the observed samples, so the '
                            'magnitude channel could not be checked')
            if last.std() <= 1e-9:
                fail.append('the thirteenth channel is constant, so it is not the magnitude curve')
            print('  thirteenth channel        max |correlation| with any single lead %.3f, '
                  'std %.4f over %d observed samples' % (same, last.std(), int(v.sum())))
            if same > 0.995:
                fail.append('the thirteenth channel is a copy of a lead, not the magnitude curve')

    # ---- 3. the target ------------------------------------------------------------------
    crit = L.build_loss(cfg)
    print('\nTARGET')
    sigma = getattr(crit, 'sigma', None)
    print('  boundary_sigma            %s' % (sigma if sigma else 'ABSENT, the target is one-hot'))
    print('  boundary_weight           %.1f over a band of %d samples'
          % (crit.boundary_weight, crit.boundary_band))
    if cfg.get('loss', {}).get('boundary_sigma') and not sigma:
        fail.append('loss.boundary_sigma is set in the config and did not reach the objective')
    if sigma:
        tgt = torch.stack([ds[i]['target'] for i in range(4)])
        q = L.soft_target(tgt, sigma, min_sigma=crit.sigma_floor)
        v = tgt != -100
        softened = ((q.max(1).values < 0.999) & v).float().mean()
        agree = (q.argmax(1)[v] == tgt[v]).float().mean()
        print('  samples softened          %.1f%% of the supervised window' % (100 * softened))
        print('  argmax still the label    %.2f%%' % (100 * agree))
        if softened < 0.01:
            fail.append('the soft target is indistinguishable from the hard one')
        if agree < 0.999:
            fail.append('smoothing moved the argmax away from the label, which would add bias')
        if crit.boundary_weight > 1.0:
            print('  NOTE the upweighted band is still on alongside the smoothing')

    # ---- 3b. the fourteenth channel and the auxiliary head --------------------------------
    if bool(cfg['data'].get('validity_channel', False)):
        print('\nVALIDITY CHANNEL')
        if not getattr(ds, 'validity_channel', False):
            fail.append('data.validity_channel is set and the loader did not receive it')
        else:
            # find a crop that is actually truncated, otherwise the assertion is vacuous
            probe, seen_partial = item, False
            for j in range(min(200, len(ds))):
                cand = ds[j]
                if not bool(cand['valid'].numpy().all()):
                    probe, seen_partial = cand, True
                    break
            ch = probe['signal'].numpy()[-1]
            mask = probe['valid'].numpy().astype(np.float32)
            vals = np.unique(ch)
            print('  fourteenth channel        values %s over a crop that is %.1f%% observed'
                  % (np.round(vals, 4).tolist()[:5], 100 * mask.mean()))
            if not set(np.round(vals, 6).tolist()).issubset({0.0, 1.0}):
                fail.append('the fourteenth channel is not binary, so it was normalised')
            if float(np.abs(ch - mask).max()) > 1e-6:
                fail.append('the fourteenth channel does not agree with the validity mask')
            if not seen_partial:
                fail.append('no truncated crop was found in the first 200 items, so the '
                            'validity channel is constant and carries no information')
    if (cfg.get('model', {}).get('p_head') or {}).get('enabled'):
        print('\nAUXILIARY HEAD')
        print('  p_head                    %s' % ('present' if net.p_head is not None else 'ABSENT'))
        if net.p_head is None:
            fail.append('model.p_head.enabled is set and the model has no head')
        else:
            b = torch.stack([ds[i]['signal'] for i in range(4)])
            v = torch.stack([ds[i]['valid'] for i in range(4)])
            li = torch.stack([ds[i]['lead_idx'] for i in range(4)])
            net.eval()
            with torch.no_grad():
                out = net(b, li, valid=v)
            if not isinstance(out, tuple) or out[1].shape != (4,):
                fail.append('the dual-head model did not return a per-item P logit')
            else:
                print('  forward returns            logits %s and p_logit %s'
                      % (tuple(out[0].shape), tuple(out[1].shape)))
        if float(cfg.get('loss', {}).get('p_head_weight', 0.0) or 0.0) <= 0:
            fail.append('the head is enabled and loss.p_head_weight is zero, so it gets no gradient')

    # ---- 4. the structural transforms ---------------------------------------------------
    struct = cfg.get('structural_augment') or {}
    print('\nAUGMENTATION')
    if not struct:
        print('  structural_augment        ABSENT')
    elif not getattr(ds, 'structural', None):
        fail.append('structural_augment is in the config and did not reach the loader')
    elif not ds.training:
        fail.append('the training flag is off, so no augmentation would fire')
    else:
        rng = np.random.default_rng(a.seed)
        idx = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False)
        fired, bad = {}, 0
        tail_draws, repolarisation_draws, repolarisation_targets = [], [], []
        for i in idx:
            it = ds[int(i)]
            applied = it['meta'].get('structural') or {}
            for k in applied:
                fired[k] = fired.get(k, 0) + 1
            if 'isolate_tail_ms' in applied:
                tail_draws.append(float(applied['isolate_tail_ms']))
            if 'repolarisation_alpha' in applied:
                repolarisation_draws.append(float(applied['repolarisation_alpha']))
            if 'repolarisation_target_ms' in applied:
                repolarisation_targets.append(float(applied['repolarisation_target_ms']))
            t = it['target'].numpy()
            if not set(np.unique(t)).issubset({-100, 0, 1, 2, 3}):
                bad += 1
        m = len(idx)
        print('  %-22s %8s %9s %10s' % ('transform', 'fired', 'rate', 'configured'))
        for key, conf in (('isolate_beat', 'isolate_beat_p'), ('left_edge_ms', 'left_edge_p'),
                          ('p_absent', 'p_absent_p'), ('p_mask', 'p_mask_p'),
                          ('repolarisation_alpha', 'repolarisation_stretch_p')):
            c = fired.get(key, 0)
            want = float(struct.get(conf, 0.0) or 0.0)
            print('  %-22s %8d %9.3f %10.2f' % (key, c, c / m, want))
            if want > 0.05 and c / m < want * 0.4:
                fail.append('%s fired at %.3f against a configured %.2f' % (key, c / m, want))
        if tail_draws:
            print('  isolate tail draw         %.1f to %.1f ms over %d isolated units'
                  % (min(tail_draws), max(tail_draws), len(tail_draws)))
        if repolarisation_draws:
            print('  repolarisation alpha      %.3f to %.3f over %d resampled units'
                  % (min(repolarisation_draws), max(repolarisation_draws),
                     len(repolarisation_draws)))
            if struct.get('repolarisation_target_ms'):
                if not any(x < 0.98 for x in repolarisation_draws):
                    fail.append('repolarisation_target_ms never shortened a T descending limb')
                if not any(x > 1.02 for x in repolarisation_draws):
                    fail.append('repolarisation_target_ms never lengthened a T descending limb')
        if repolarisation_targets:
            print('  repolarisation target     %.1f to %.1f ms over %d resampled units'
                  % (min(repolarisation_targets), max(repolarisation_targets),
                     len(repolarisation_targets)))
        # The four cells, and the two conditional rates that have to sit away from zero and one.
        cells = {}
        for i in idx:
            it = ds[int(i)]
            cells[it['meta']['cell']] = cells.get(it['meta']['cell'], 0) + 1
        if cells:
            m2 = sum(cells.values())
            print('\n  structural cells')
            for k, lab in (('A', 'normal'), ('B', 'truncated + P'),
                           ('C', 'full, P masked'), ('D', 'QRS edge')):
                print('    %s %-16s %6d %8.3f' % (k, lab, cells.get(k, 0), cells.get(k, 0) / m2))
            trunc = cells.get('B', 0) + cells.get('D', 0)
            untrunc = cells.get('A', 0) + cells.get('C', 0)
            r_t = cells.get('D', 0) / trunc if trunc else 0.0
            r_u = cells.get('C', 0) / untrunc if untrunc else 0.0
            print('    P(no P | truncated)     %.3f' % r_t)
            print('    P(no P | not truncated) %.3f' % r_u)
            if cfg.get('structural_augment', {}).get('context'):
                if min(cells.get('A', 0), cells.get('B', 0), cells.get('C', 0), cells.get('D', 0)) == 0:
                    fail.append('at least one of the four structural cells never fired, so the '
                                'auxiliary head can read its target off the validity channel')
                if not (0.05 < r_t < 0.95) or not (0.05 < r_u < 0.95):
                    fail.append('P observability is nearly determined by truncation, at %.3f and '
                                '%.3f, which defeats the auxiliary task' % (r_t, r_u))
        c = fired.get('t_offset_jitter_ms', 0)
        print('  %-22s %8d %9.3f %10s' % ('t_offset_jitter_ms', c, c / m,
                                          struct.get('t_offset_jitter_ms', 0)))
        if bad:
            fail.append('%d of %d targets hold a class outside {-100, 0, 1, 2, 3}' % (bad, m))

        if bool(cfg.get('augment', {}).get('epoch_resample', False)):
            probes = [int(i) for i in idx[:min(16, len(idx))]]
            ds.set_epoch(0)
            before = [(ds[i]['meta'].get('structural'), int(ds[i]['crop_start'])) for i in probes]
            ds.set_epoch(1)
            after = [(ds[i]['meta'].get('structural'), int(ds[i]['crop_start'])) for i in probes]
            ds.set_epoch(0)
            changed = sum(a0 != a1 for a0, a1 in zip(before, after))
            print('  epoch resampling          changed %d of %d probe assignments'
                  % (changed, len(probes)))
            if changed == 0:
                fail.append('augment.epoch_resample is on but epoch zero and one are identical')

    print('\n' + '=' * 72)
    if fail:
        print('PREFLIGHT FAILED, %d problem(s)' % len(fail))
        for f in fail:
            print('  - %s' % f)
        raise SystemExit(1)
    print('PREFLIGHT PASSED. Every setting in the config reached the code.')


if __name__ == '__main__':
    main()
