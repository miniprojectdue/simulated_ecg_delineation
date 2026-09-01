#!/usr/bin/env python3
"""
train.py  -  one driver for both stages of the semisupervised schedule.

Stage one, pretraining.
    python ml_modelling/scripts/train.py --config ml_modelling/configs/pretrain.yaml

Stage two, fine-tuning on the human-reviewed units.
    python ml_modelling/scripts/train.py --config ml_modelling/configs/finetune.yaml


Three fine-tuning strategies are supported,

    freeze: none          every weight moves. optim.encoder_lr_mult below one gives the
                          discriminative schedule, where the encoder moves more slowly than
                          the decoder and the head. This is the default.
    freeze: encoder       the encoder and the bottleneck are frozen and only the decoder and
                          the head adapt. The cheapest strategy and the least able to overfit.
    freeze: encoder_bn    every weight moves but the batch normalisation statistics in the
                          encoder are held at their pretrained values, which is the usual
                          remedy when the fine-tuning batches are too small to estimate them.

Outputs go to ml_modelling/checkpoints/<run name>/ and ml_modelling/results/<run name>/.
"""
import argparse
import contextlib
import copy
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    ROOT, Tee, environment_report, load_config, log, pick_device, save_json, set_seed,
)
from dataset import build_geometry_validation_loaders, build_loaders  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from losses import build_loss  # noqa: E402
from metrics import ConfusionAccumulator, FiducialAccumulator, landmarks_from_vector  # noqa: E402
from model import build_model, count_parameters  # noqa: E402
from postprocess import batch_fiducials  # noqa: E402


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def apply_freeze(model, policy):
    """Put the model into the requested fine-tuning regime and report what moved."""
    policy = (policy or 'none').lower()
    for param in model.parameters():
        param.requires_grad = True
    if policy in ('none', ''):
        pass
    elif policy == 'encoder':
        for param in model.encoder_parameters():
            param.requires_grad = False
    elif policy == 'encoder_bn':
        pass  # handled at train time by holding the encoder normalisation layers in eval mode
    else:
        raise SystemExit('train.freeze %r is not one of none, encoder or encoder_bn' % policy)
    return count_parameters(model)


def freeze_encoder_batchnorm(model):
    """Hold the encoder normalisation statistics at their pretrained values."""
    for module in list(model.encoders.modules()) + list(model.bottleneck.modules()):
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.GroupNorm)):
            module.eval()


def build_optimizer(model, cfg):
    o = cfg['optim']
    encoder_mult = float(o.get('encoder_lr_mult', 1.0) or 1.0)
    lr = float(o['lr'])
    decay = float(o.get('weight_decay', 0.0) or 0.0)
    betas = tuple(float(b) for b in (o.get('betas') or [0.9, 0.999]))

    encoder_ids = {id(p) for p in model.encoder_parameters()}
    encoder_params = [p for p in model.parameters() if id(p) in encoder_ids and p.requires_grad]
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
    groups = []
    if encoder_params:
        groups.append({'params': encoder_params, 'lr': lr * encoder_mult, 'name': 'encoder'})
    if other_params:
        groups.append({'params': other_params, 'lr': lr, 'name': 'decoder_and_head'})
    if not groups:
        raise SystemExit('every parameter is frozen, so there is nothing to train')

    name = str(o.get('optimizer', 'adamw')).lower()
    if name == 'adamw':
        return torch.optim.AdamW(groups, lr=lr, weight_decay=decay, betas=betas)
    if name == 'adam':
        return torch.optim.Adam(groups, lr=lr, weight_decay=decay, betas=betas)
    if name == 'sgd':
        return torch.optim.SGD(groups, lr=lr, weight_decay=decay, momentum=0.9, nesterov=True)
    raise SystemExit('optim.optimizer %r is not implemented' % name)


def lr_lambda_factory(cfg, steps_per_epoch):
    """Linear warmup then cosine decay, expressed as a multiplier on each group's own rate."""
    o = cfg['optim']
    epochs = int(cfg['train']['epochs'])
    warmup_steps = max(0, int(float(o.get('warmup_epochs', 0) or 0) * steps_per_epoch))
    total_steps = max(1, epochs * steps_per_epoch)
    min_ratio = float(o.get('min_lr', 0.0) or 0.0) / max(float(o['lr']), 1e-12)
    schedule = str(o.get('scheduler', 'cosine')).lower()

    def fn(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if schedule in ('none', 'constant'):
            return 1.0
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if schedule == 'cosine':
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        if schedule == 'linear':
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
        raise SystemExit('optim.scheduler %r is not implemented' % schedule)

    return fn


def load_initial_weights(model, path, device, strict=False):
    """Start from a pretrained checkpoint. This is what makes stage two a fine-tune."""
    abs_path = resolve(path)
    if not os.path.isfile(abs_path):
        raise SystemExit('train.init_from names %s, which does not exist. Run stage one first.' % abs_path)
    payload = torch.load(abs_path, map_location=device)
    state = payload.get('model', payload)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    log('initialised from %s, %d missing and %d unexpected keys'
        % (abs_path, len(missing), len(unexpected)))
    if missing:
        log('  missing %s' % list(missing)[:8])
    if unexpected:
        log('  unexpected %s' % list(unexpected)[:8])
    return payload


CELLS = ('A normal', 'B truncated + P', 'C full, P masked', 'D QRS edge')


def _forward(model, batch, device):
    """One forward pass that tolerates both the single-head and the dual-head model."""
    signal = batch['signal'].to(device, non_blocking=True)
    lead_idx = batch['lead_idx'].to(device, non_blocking=True)
    valid = batch['valid'].to(device, non_blocking=True) if 'valid' in batch else None
    # Validity is consumed by repaired bottleneck attention as well as by the optional P head.
    # Passing it unconditionally keeps those two features independent in configuration.
    out = model(signal, lead_idx, valid=valid)
    return (out if isinstance(out, tuple) else (out, None))


def _aux_loss(p_logit, batch, device, weight):
    if p_logit is None or weight <= 0 or 'p_observable' not in batch:
        return None
    y = batch['p_observable'].to(device, non_blocking=True)
    return weight * F.binary_cross_entropy_with_logits(p_logit, y)


def _cell_lines(summary, split='validation'):
    """Render the per-cell diagnostic. Empty when the auxiliary task is not in use."""
    if not summary or len(summary) < 2:
        return []          # the validation split runs with augmentation off, so it is all cell A
    out = ['  P head and segmentation by structural cell, %s split' % split]
    for name, v in summary.items():
        acc = v.get('p_head_correct')
        out.append('    %-18s n %5d   P head correct %s   segmentation emits P %5.1f%%'
                   % (name, v['n'], '%5.1f%%' % (100 * acc) if acc is not None else '   --  ',
                      100 * (v.get('segmentation_emits_P') or 0.0)))
    return out


class CellTracker(object):
    """P-head and segmentation behaviour split by structural-augmentation cell.

    A single pooled accuracy would read near perfect while the head was doing nothing but
    reading the validity mask, since two of the four cells are separable that way. Reporting the
    cells apart is what makes the shortcut visible, and it has to be visible during the run
 
    """

    def __init__(self):
        self.n = [0] * 4
        self.right = [0] * 4
        self.pred_p = [0] * 4
        self.seg_p = [0] * 4

    def update(self, p_logit, logits, batch):
        if 'cell_idx' not in batch:
            return
        cell = batch['cell_idx'].numpy()
        # Restricted to the supervised window. Taking any() over the whole crop counts a P
        # predicted in the padding, which carries no gradient, and a P predicted on a
        # neighbouring beat, which genuinely has one and lies outside the labelled window, so
        # the unrestricted version reports close to 100 per cent for every cell and says
        # nothing about whether the model asserts a P where the label denies one.
        pred_p_map = (logits.argmax(dim=1) == 1)
        if 'supervised' in batch:
            pred_p_map = pred_p_map & batch['supervised'].to(pred_p_map.device)
        seg_has_p = pred_p_map.any(dim=1).detach().cpu().numpy()
        if p_logit is not None:
            pred = (p_logit.detach().float().sigmoid() >= 0.5).cpu().numpy()
            true = batch['p_observable'].numpy() >= 0.5
        else:
            pred = true = None
        for i, c in enumerate(cell):
            c = int(c)
            self.n[c] += 1
            self.seg_p[c] += int(seg_has_p[i])
            if pred is not None:
                self.right[c] += int(bool(pred[i]) == bool(true[i]))
                self.pred_p[c] += int(pred[i])

    def summary(self):
        return {CELLS[c]: {'n': self.n[c],
                           'p_head_correct': (self.right[c] / self.n[c]) if self.n[c] else None,
                           'p_head_says_present': (self.pred_p[c] / self.n[c]) if self.n[c] else None,
                           'segmentation_emits_P': (self.seg_p[c] / self.n[c]) if self.n[c] else None}
                for c in range(4) if self.n[c]}

    def lines(self):
        out = []
        for c in range(4):
            if not self.n[c]:
                continue
            out.append('    %-18s n %5d   P head correct %5.1f%%   segmentation emits P %5.1f%%'
                       % (CELLS[c], self.n[c], 100 * self.right[c] / self.n[c],
                          100 * self.seg_p[c] / self.n[c]))
        return out


@torch.no_grad()
def evaluate_split(model, loader, criterion, device, cfg, collect_fiducials=True, max_batches=0,
                   full_read=False):
    """Validation pass. Returns the loss, the segmentation summary and the fiducial summary."""
    model.eval()
    confusion = ConfusionAccumulator(n_classes=int(cfg['model'].get('n_classes', 4)))
    fs_hz = float(cfg['data'].get('fs_hz', 500))
    fiducial = FiducialAccumulator(tolerances_ms=cfg['eval'].get('tolerance_ms', [10, 25, 50, 75, 150]),
                                   primary_ms=cfg['eval'].get('primary_tolerance_ms', 25),
                                   fs_hz=fs_hz)
    total_loss, total_batches = 0.0, 0
    lam = float(cfg.get('loss', {}).get('p_head_weight', 0.0) or 0.0)
    tracker = CellTracker()

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        target = batch['target'].to(device, non_blocking=True)
        tail = batch['tail'].to(device, non_blocking=True) if 'tail' in batch else None
        logits, p_logit = _forward(model, batch, device)
        loss, _ = criterion(logits, target, tail=tail)
        aux = _aux_loss(p_logit, batch, device, lam)
        if aux is not None:
            loss = loss + aux
        tracker.update(p_logit, logits, batch)
        total_loss += float(loss.detach())
        total_batches += 1

        predicted = logits.argmax(dim=1)
        confusion.update(predicted.cpu().numpy(), target.cpu().numpy())

        if collect_fiducials:
            readable = batch.get('valid') if full_read else None
            predicted_fiducials = batch_fiducials(logits, batch['trace'], cfg.get('postprocess'),
                                                  supervised=batch['supervised'],
                                                  readable=readable)
            truth_fiducials = [landmarks_from_vector(v.numpy()) for v in batch['landmarks']]
            fiducial.update(predicted_fiducials, truth_fiducials)

    model.train()
    return {
        'loss': total_loss / max(1, total_batches),
        'segmentation': confusion.summary(),
        'fiducials': fiducial.summary() if collect_fiducials else {},
        'cells': tracker.summary(),
    }


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg,
                    epoch, global_step, freeze_policy, teacher=None):
    model.train()
    if freeze_policy == 'encoder_bn':
        freeze_encoder_batchnorm(model)
    clip = float(cfg['optim'].get('grad_clip_norm', 0.0) or 0.0)
    accumulate = max(1, int(cfg['train'].get('accumulate_steps', 1) or 1))
    log_every = int(cfg['train'].get('log_every_steps', 50) or 50)
    use_amp = bool(cfg['train'].get('amp', False)) and device.type == 'cuda'

    running = {'loss': 0.0, 'ce': 0.0, 'dice': 0.0, 'p': 0.0}
    lam = float(cfg.get('loss', {}).get('p_head_weight', 0.0) or 0.0)
    tracker = CellTracker()   # the four cells only exist on the training split
    seen = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        target = batch['target'].to(device, non_blocking=True)
        tail = batch['tail'].to(device, non_blocking=True) if 'tail' in batch else None

        teacher_logits = None
        if teacher is not None:
            with torch.no_grad():
                teacher_logits, _ = _forward(teacher, batch, device)

        autocast = torch.autocast(device_type='cuda') if use_amp else contextlib.nullcontext()
        with autocast:
            logits, p_logit = _forward(model, batch, device)
            loss, parts = criterion(logits, target, tail=tail, teacher_logits=teacher_logits)
            aux = _aux_loss(p_logit, batch, device, lam)
            if aux is not None:
                loss = loss + aux
                parts['p'] = float(aux.detach())
        tracker.update(p_logit, logits, batch)

        scaled = loss / accumulate
        if scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % accumulate == 0:
            if clip:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        for key in running:
            running[key] += float(parts.get(key, 0.0))
        seen += 1

        if log_every and (step + 1) % log_every == 0:
            rate = seen / max(1e-9, time.time() - started)
            log('  epoch %d  step %d/%d  loss %.4f  ce %.4f  dice %.4f  lr %.2e  %.1f batches/s'
                % (epoch, step + 1, len(loader), running['loss'] / seen, running['ce'] / seen,
                   running['dice'] / seen, optimizer.param_groups[0]['lr'], rate))

    stats = {key: value / max(1, seen) for key, value in running.items()}
    stats['cells'] = tracker.summary()
    return stats, global_step


def monitored_value(stats, key):
    if key == 'val_loss':
        return stats['loss']
    if key == 'val_mean_iou':
        return stats['segmentation']['mean_iou_waves']
    if key == 'val_mean_f1':
        return stats['segmentation']['mean_f1_waves']
    if key == 'val_boundary_mae':
        return stats['fiducials'].get('_aggregate', {}).get('boundary_mae_ms', float('nan'))
    if key == 'val_qrs_boundary_mae':
        # Few-shot adaptation with T supervision withheld (loss.adapt_qrs_local or
        # loss.mask_t_supervision). The full boundary aggregate scores the T boundaries
        # against the global adaptation labels the loss deliberately ignores, which both
        # stops training at the first epoch and selects the wrong checkpoint. This monitor
        # pools the two QRS boundaries only.
        values = [stats['fiducials'].get(name, {}).get('mae_ms', float('nan'))
                  for name in ('qrs_onset', 'qrs_offset')]
        values = [v for v in values if v == v]
        return sum(values) / len(values) if values else float('nan')
    raise SystemExit('train.monitor %r is not one of val_loss, val_mean_iou, val_mean_f1, '
                     'val_boundary_mae or val_qrs_boundary_mae' % key)


def is_better(value, best, mode):
    if not np.isfinite(value):
        return False
    if best is None or not np.isfinite(best):
        return True
    return value > best if mode == 'max' else value < best


def geometry_validation_score(stats_by_tail, cfg, natural_mean_iou=None):
    """Worst-tail T score, including missing detections and a collapsed response slope."""
    spec = dict(cfg.get('train', {}).get('geometry_validation') or {})
    missing_penalty = float(spec.get('missing_penalty_ms', 150.0))
    slope_penalty = float(spec.get('slope_penalty_ms', 50.0))
    minimum_iou = spec.get('min_natural_mean_iou')
    if minimum_iou is not None and (natural_mean_iou is None
                                    or float(natural_mean_iou) < float(minimum_iou)):
        return float('inf')
    scores = []
    for stats in stats_by_tail.values():
        t = stats.get('fiducials', {}).get('t_offset', {})
        mae = float(t.get('mae_ms', float('nan')))
        coverage = float(t.get('detection_coverage', float('nan')))
        slope = float(t.get('response_slope', float('nan')))
        if not (np.isfinite(mae) and np.isfinite(coverage) and np.isfinite(slope)):
            scores.append(float('inf'))
            continue
        scores.append(mae + missing_penalty * (1.0 - coverage)
                      + slope_penalty * abs(1.0 - slope))
    return max(scores) if scores else float('nan')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='section.key=value',
                        help='override a config value from the command line. Repeatable.')
    parser.add_argument('--resume', default='', help='resume from a checkpoint written by this script')
    parser.add_argument('--smoke', action='store_true',
                        help='one short epoch over a handful of batches, to prove the plumbing')
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg['train']['epochs'] = 1
        cfg['data']['max_units'] = max(64, int(cfg['train']['batch_size']) * 4)
        cfg['data']['num_workers'] = 0
        cfg['run']['name'] = cfg['run']['name'] + '_smoke'
        geometry_spec = cfg['train'].get('geometry_validation')
        if geometry_spec:
            # A one-epoch plumbing check is not expected to clear the real
            # quality gate, but it should still exercise checkpoint writing.
            geometry_spec['min_natural_mean_iou'] = 0.0

    run_name = cfg['run']['name']
    ckpt_dir = os.path.join(resolve(cfg['run'].get('out_root', 'ml_modelling/checkpoints')), run_name)
    result_dir = os.path.join(resolve(cfg['run'].get('log_root', 'ml_modelling/results')), run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    tee = Tee(os.path.join(result_dir, 'train_log.txt'))
    sys.stdout = tee
    try:
        set_seed(int(cfg['run'].get('seed', 0)), bool(cfg['run'].get('deterministic', True)))
        device = pick_device(cfg['run'].get('device', 'auto'))
        log('run %s on %s' % (run_name, device))
        log('environment %s' % environment_report())

        train_loader, val_loader = build_loaders(cfg)
        geometry_loaders = build_geometry_validation_loaders(cfg)

        model = build_model(cfg).to(device)
        if cfg['train'].get('init_from'):
            load_initial_weights(model, cfg['train']['init_from'], device)
        freeze_policy = str(cfg['train'].get('freeze', 'none')).lower()
        params = apply_freeze(model, freeze_policy)
        log('model %s, %d parameters of which %d trainable, freeze policy %s'
            % (cfg['model']['name'], params['total'], params['trainable'], freeze_policy))

        criterion = build_loss(cfg).to(device)
        teacher = None
        if float(cfg.get('loss', {}).get('adapt_distill_weight', 0.0) or 0.0) > 0:
            # Frozen copy of the initialisation weights, used by the adaptation distillation
            # term to hold the post-QRS behaviour at its zero-shot state. Built before the
            # first optimizer step so it is exactly the init_from checkpoint. Not compatible
            # with --resume, which would rebuild the teacher from mid-run weights.
            teacher = copy.deepcopy(model).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            log('distillation teacher frozen from the initialisation weights, '
                'weight %.2f' % float(cfg['loss']['adapt_distill_weight']))
        optimizer = build_optimizer(model, cfg)
        steps_per_epoch = max(1, len(train_loader) // max(1, int(cfg['train'].get('accumulate_steps', 1))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(cfg, steps_per_epoch))
        use_amp = bool(cfg['train'].get('amp', False)) and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        start_epoch, global_step, best, best_geometry, bad_epochs = 1, 0, None, None, 0
        history = []
        if args.resume:
            payload = torch.load(resolve(args.resume), map_location=device)
            model.load_state_dict(payload['model'])
            optimizer.load_state_dict(payload['optimizer'])
            if payload.get('scheduler'):
                scheduler.load_state_dict(payload['scheduler'])
            start_epoch = int(payload.get('epoch', 0)) + 1
            global_step = int(payload.get('global_step', 0))
            best = payload.get('best')
            best_geometry = payload.get('best_geometry')
            history = payload.get('history', [])
            log('resumed from %s at epoch %d' % (args.resume, start_epoch))

        monitor = cfg['train'].get('monitor', 'val_mean_iou')
        mode = cfg['train'].get('monitor_mode', 'max')
        patience = int(cfg['train'].get('early_stopping_patience', 0) or 0)
        save_json(cfg, os.path.join(result_dir, 'config_resolved.json'))

        for epoch in range(start_epoch, int(cfg['train']['epochs']) + 1):
            if hasattr(train_loader.dataset, 'set_epoch'):
                # Epoch one deliberately retains the historical index-only draw. Subsequent
                # epochs change it when augment.epoch_resample is enabled.
                train_loader.dataset.set_epoch(epoch - 1)
            epoch_started = time.time()
            train_stats, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, device, cfg,
                epoch, global_step, freeze_policy, teacher=teacher)
            val_stats = evaluate_split(model, val_loader, criterion, device, cfg,
                                       max_batches=8 if args.smoke else 0)
            geometry_stats = {}
            for tail_ms, geometry_loader in geometry_loaders.items():
                stats = evaluate_split(model, geometry_loader, criterion, device, cfg,
                                       max_batches=2 if args.smoke else 0, full_read=True)
                geometry_stats[str(tail_ms)] = stats
                t = stats['fiducials'].get('t_offset', {})
                log('  geometry tail %g ms  T MAE %.2f ms  bias %+.2f  coverage %.3f  '
                    'slope %.3f  r %.3f'
                    % (tail_ms, t.get('mae_ms', float('nan')),
                       t.get('bias_ms', float('nan')),
                       t.get('detection_coverage', float('nan')),
                       t.get('response_slope', float('nan')),
                       t.get('response_r', float('nan'))))
            geometry_score = geometry_validation_score(
                geometry_stats, cfg,
                natural_mean_iou=val_stats['segmentation'].get('mean_iou_waves'))
            value = monitored_value(val_stats, monitor)
            aggregate = val_stats['fiducials'].get('_aggregate', {})
            log('epoch %d  train loss %.4f  val loss %.4f  val mean IoU %.4f  boundary MAE %.2f ms  %.1f s'
                % (epoch, train_stats['loss'], val_stats['loss'],
                   val_stats['segmentation']['mean_iou_waves'],
                   aggregate.get('boundary_mae_ms', float('nan')), time.time() - epoch_started))

            for line in _cell_lines(train_stats.get('cells'), 'training'):
                log(line)

            record = {'epoch': epoch, 'train': train_stats, 'val_loss': val_stats['loss'],
                      'val_segmentation': val_stats['segmentation'],
                      'val_fiducial_aggregate': aggregate, 'monitor': monitor, 'value': value,
                      'geometry_validation': geometry_stats,
                      'geometry_score': geometry_score,
                      'lr': [g['lr'] for g in optimizer.param_groups]}
            history.append(record)
            save_json(history, os.path.join(result_dir, 'history.json'))

            payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'epoch': epoch,
                       'global_step': global_step, 'best': best, 'history': history,
                       'best_geometry': best_geometry,
                       'geometry_validation': geometry_stats,
                       'geometry_score': geometry_score,
                       'config': cfg, 'monitor': monitor, 'value': value}
            should_stop = False
            if is_better(value, best, mode):
                best = value
                bad_epochs = 0
                payload['best'] = best
                torch.save(payload, os.path.join(ckpt_dir, 'best.pt'))
                save_json({'epoch': epoch, 'monitor': monitor, 'value': value,
                           'segmentation': val_stats['segmentation'],
                           'fiducials': val_stats['fiducials']},
                          os.path.join(result_dir, 'best_validation.json'))
                log('  new best %s %.6f, checkpoint written' % (monitor, value))
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    should_stop = True

            if geometry_loaders and np.isfinite(geometry_score) \
                    and (best_geometry is None or geometry_score < best_geometry):
                best_geometry = geometry_score
                payload['best_geometry'] = best_geometry
                torch.save(payload, os.path.join(ckpt_dir, 'best_geometry.pt'))
                save_json({'epoch': epoch, 'score': geometry_score,
                           'definition': ('maximum across tails of T-offset MAE plus configured '
                                          'missing-detection and response-slope penalties'),
                           'tails': geometry_stats},
                          os.path.join(result_dir, 'best_geometry_validation.json'))
                log('  new best geometry score %.3f, best_geometry.pt written' % geometry_score)

            # Save resumable state after both selectors have updated their best values.
            payload['best'] = best
            payload['best_geometry'] = best_geometry
            torch.save(payload, os.path.join(ckpt_dir, 'last.pt'))
            if cfg['train'].get('save_every_epoch'):
                torch.save(payload, os.path.join(ckpt_dir, 'epoch_%03d.pt' % epoch))
            if should_stop:
                log('early stopping after %d epochs without an improvement in %s'
                    % (bad_epochs, monitor))
                break

        log('training finished, best %s %s' % (monitor, best))
        log('checkpoints in %s' % ckpt_dir)
        log('to evaluate, run')
        log('  python ml_modelling/scripts/evaluate.py --checkpoint %s --units <external units csv>'
            % os.path.join(ckpt_dir, 'best.pt'))
    finally:
        sys.stdout = tee.stdout
        tee.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
