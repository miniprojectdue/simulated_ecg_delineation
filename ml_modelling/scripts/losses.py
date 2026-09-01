"""
losses.py  -  the segmentation objective.

Cross entropy with class weights for the class imbalance, soft Dice for the region shape
cross entropy is indifferent to, and extra weight on every sample within
loss.boundary_band_samples of a class transition, by a factor of loss.boundary_weight (two in
stage one, three in stage two). This is the Equation 3.11 formulation. Targets are the hard
one-hot label everywhere.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IGNORE_INDEX, N_CLASSES  # noqa: E402

BACKGROUND, P_WAVE, QRS, T_WAVE = 0, 1, 2, 3


def boundary_weight_map(target, band=10, weight=2.0, ignore_index=IGNORE_INDEX):
    """Per-sample weights that rise near a class transition in the target."""
    if weight is None or weight <= 1.0 or band <= 0:
        return torch.ones_like(target, dtype=torch.float32)
    valid = target != ignore_index
    left, right = target[:, :-1], target[:, 1:]
    change = (left != right) & valid[:, :-1] & valid[:, 1:]
    marks = torch.zeros_like(target, dtype=torch.float32)
    marks[:, :-1] += change.float()
    marks[:, 1:] += change.float()
    marks = marks.clamp(max=1.0).unsqueeze(1)
    kernel = torch.ones(1, 1, 2 * band + 1, device=target.device, dtype=marks.dtype)
    spread = F.conv1d(marks, kernel, padding=band).squeeze(1) > 0
    return torch.where(spread, torch.full_like(marks.squeeze(1), float(weight)),
                       torch.ones_like(marks.squeeze(1)))


def masked_cross_entropy(logits, target, class_weights=None, sample_weights=None,
                         label_smoothing=0.0, ignore_index=IGNORE_INDEX):
    """Cross entropy over the supervised samples only, with optional per-sample weights."""
    per_sample = F.cross_entropy(logits, target, weight=class_weights, reduction='none',
                                 ignore_index=ignore_index, label_smoothing=label_smoothing)
    mask = (target != ignore_index).float()
    if sample_weights is not None:
        per_sample = per_sample * sample_weights
        denominator = (mask * sample_weights).sum().clamp(min=1.0)
    else:
        denominator = mask.sum().clamp(min=1.0)
    return (per_sample * mask).sum() / denominator


def soft_dice(logits, target, n_classes=N_CLASSES, ignore_index=IGNORE_INDEX, eps=1.0,
              include_background=False):
    """Soft Dice averaged over the classes that actually occur in the batch."""
    mask = (target != ignore_index)
    if not bool(mask.any()):
        return logits.sum() * 0.0
    probs = torch.softmax(logits, dim=1)
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    one_hot = F.one_hot(safe_target, num_classes=n_classes).permute(0, 2, 1).float()
    mask_f = mask.unsqueeze(1).float()
    probs = probs * mask_f
    one_hot = one_hot * mask_f

    start = 0 if include_background else 1
    dims = (0, 2)
    intersection = (probs[:, start:] * one_hot[:, start:]).sum(dims)
    cardinality = probs[:, start:].sum(dims) + one_hot[:, start:].sum(dims)
    present = one_hot[:, start:].sum(dims) > 0
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    if bool(present.any()):
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()


class DelineationLoss(nn.Module):
    """Weighted sum of the three ingredients, reported as a dict so training can log each."""

    def __init__(self, cross_entropy_weight=1.0, dice_weight=0.5, class_weights=None,
                 boundary_weight=1.0, boundary_band_samples=10, label_smoothing=0.0,
                 tail_background_weight=1.0, mask_t_supervision=False,
                 adapt_qrs_local=False, adapt_qrs_local_margin_samples=10,
                 adapt_distill_weight=0.0,
                 n_classes=N_CLASSES, ignore_index=IGNORE_INDEX):
        super(DelineationLoss, self).__init__()
        self.ce_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)
        self.boundary_weight = float(boundary_weight or 1.0)
        self.boundary_band = int(boundary_band_samples or 0)
        self.label_smoothing = float(label_smoothing or 0.0)
        # Extra weight on the supervised isolated-tail background samples the dataset marks
        # with its ``tail`` mask. Motivation, measured on the current checkpoint: 80% of
        # isolated units overrun the true T offset, and the T-vs-background logit margin in
        # the overrun samples is a median 0.52 nats, which is what marginal calls look like
        # when the mistake costs 0.2x through the background class weight. A value of 5.0
        # exactly cancels that 0.2 and restores parity with the wave classes on the tail.
        # 1.0 leaves behaviour identical to before this key existed.
        self.tail_weight = float(tail_background_weight or 1.0)
        # Few-shot adaptation only. The external adaptation labels are the global V2/V5-derived
        # reference applied to every lead. Cross-lead deviation from that reference is 34-47 ms
        # at the 98th percentile for the QRS landmarks but 168-172 ms for the T landmarks, so
        # the adaptation targets are near-correct per lead for QRS and badly wrong per lead for
        # T. When this flag is set, samples labelled T in the target are moved to ignore_index
        # before any term is computed: the finetune adapts the QRS behaviour and leaves the T
        # behaviour at its initialisation (zero-shot) state instead of dragging it onto the
        # global-label convention. Background samples after the reference T offset remain
        # supervised, which is harmless because the global offset sits at or past the per-lead
        # one. Never set this for pretraining or the in-distribution finetune.
        self.mask_t_supervision = bool(mask_t_supervision)
        # Few-shot adaptation only, and stricter than mask_t_supervision. The adapt_ischemia3
        # run showed that masking the T class alone is not enough: the reference labels
        # background between the QRS offset and the late global T onset, and supervising those
        # samples dragged the predicted T onset from 4.6 ms early to 60.9 ms late on the
        # inferior-infarction class. When this flag is set, supervision stops at the last
        # QRS-labelled sample plus adapt_qrs_local_margin_samples (enough background to teach
        # the QRS-offset transition): every later sample is moved to ignore_index, so nothing
        # about the ST segment, the T wave or the tail is trained on the untrustworthy global
        # convention. Rows containing no QRS are left fully supervised. Never set this for
        # pretraining or the in-distribution finetune.
        self.adapt_qrs_local = bool(adapt_qrs_local)
        self.adapt_qrs_local_margin = int(adapt_qrs_local_margin_samples or 0)
        # Few-shot adaptation only, on top of adapt_qrs_local. The adapt_ischemia4 run showed
        # that leaving the post-QRS region entirely unsupervised is not enough either: with
        # nothing anchoring it, twenty-five epochs of feature drift raised the held-out
        # T-offset scatter from 29 to 87 ms and made the model emit false P waves on a corpus
        # where zero-shot correctly emits none. When this weight is positive, the samples the
        # QRS-local cutoff removed from supervision are instead trained to match the frozen
        # initialisation checkpoint's own predictions (a KL term against the teacher's
        # posterior), so the T and P behaviour is actively held at its zero-shot state while
        # the QRS boundaries adapt to the labels. train.py builds the teacher automatically
        # when this is set. Never set this for pretraining or the in-distribution finetune.
        self.adapt_distill_weight = float(adapt_distill_weight or 0.0)
        self.n_classes = int(n_classes)
        self.ignore_index = int(ignore_index)
        if class_weights:
            self.register_buffer('class_weights', torch.tensor(list(class_weights), dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, target, tail=None, teacher_logits=None):
        distill_mask = None
        if self.adapt_qrs_local:
            qrs = target == QRS
            has_qrs = qrs.any(dim=1)
            if bool(has_qrs.any()):
                length = target.size(1)
                idx = torch.arange(length, device=target.device)
                last_qrs = (length - 1) - torch.flip(qrs, dims=[1]).to(torch.uint8).argmax(dim=1)
                cutoff = last_qrs + self.adapt_qrs_local_margin
                keep = (idx.unsqueeze(0) <= cutoff.unsqueeze(1)) | ~has_qrs.unsqueeze(1)
                distill_mask = (target != self.ignore_index) & ~keep
                target = torch.where(keep, target,
                                     torch.full_like(target, self.ignore_index))
        if self.mask_t_supervision:
            target = torch.where(target == T_WAVE,
                                 torch.full_like(target, self.ignore_index), target)
        weights = self.class_weights
        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)
        sample_weights = None
        if self.boundary_weight > 1.0 and self.boundary_band > 0:
            sample_weights = boundary_weight_map(target, band=self.boundary_band,
                                                 weight=self.boundary_weight,
                                                 ignore_index=self.ignore_index)
            sample_weights = sample_weights.to(dtype=logits.dtype)
        if self.tail_weight > 1.0 and tail is not None and bool(tail.any()):
            # Combined with the boundary band by maximum rather than product, so a tail sample
            # inside the T-offset band carries the larger of the two emphases instead of both
            # stacked.
            tail_map = torch.where(tail.to(torch.bool) & (target == BACKGROUND),
                                   torch.full_like(target, self.tail_weight, dtype=logits.dtype),
                                   torch.ones_like(target, dtype=logits.dtype))
            sample_weights = (tail_map if sample_weights is None
                              else torch.maximum(sample_weights, tail_map))

        ce = masked_cross_entropy(logits, target, class_weights=weights,
                                  sample_weights=sample_weights,
                                  label_smoothing=self.label_smoothing,
                                  ignore_index=self.ignore_index)
        dice = soft_dice(logits, target, n_classes=self.n_classes,
                         ignore_index=self.ignore_index)

        total = self.ce_weight * ce + self.dice_weight * dice
        parts = {'ce': float(ce.detach()), 'dice': float(dice.detach())}
        if (self.adapt_distill_weight > 0 and teacher_logits is not None
                and distill_mask is not None and bool(distill_mask.any())):
            log_p = F.log_softmax(logits, dim=1)
            q = torch.softmax(teacher_logits.detach(), dim=1)
            kl = (q * (torch.log(q.clamp_min(1e-8)) - log_p)).sum(dim=1)
            m = distill_mask.to(kl.dtype)
            distill = (kl * m).sum() / m.sum().clamp(min=1.0)
            total = total + self.adapt_distill_weight * distill
            parts['distill'] = float(distill.detach())
        parts['loss'] = float(total.detach())
        return total, parts


def build_loss(cfg):
    l = cfg.get('loss', {})
    if l.get('boundary_sigma'):
        raise SystemExit('loss.boundary_sigma is set but the Gaussian soft-target formulation '
                         'has been removed from losses.py. Delete the key to train with hard '
                         'targets and the boundary band, or check out the pre-removal revision '
                         'to reproduce a soft-target run.')
    return DelineationLoss(
        cross_entropy_weight=l.get('cross_entropy_weight', 1.0),
        dice_weight=l.get('dice_weight', 0.5),
        class_weights=l.get('class_weights'),
        boundary_weight=l.get('boundary_weight', 1.0),
        boundary_band_samples=l.get('boundary_band_samples', 0),
        label_smoothing=l.get('label_smoothing', 0.0),
        tail_background_weight=l.get('tail_background_weight', 1.0),
        mask_t_supervision=l.get('mask_t_supervision', False),
        adapt_qrs_local=l.get('adapt_qrs_local', False),
        adapt_qrs_local_margin_samples=l.get('adapt_qrs_local_margin_samples', 10),
        adapt_distill_weight=l.get('adapt_distill_weight', 0.0),
        n_classes=int(cfg.get('model', {}).get('n_classes', N_CLASSES)),
    )
