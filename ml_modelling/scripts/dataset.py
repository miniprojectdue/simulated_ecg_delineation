#!/usr/bin/env python3
"""
dataset.py  -  the beat-window dataset that both training stages read.

One item is one delineation unit, which is one beat of one lead of one record. The input is
all twelve leads over that beat's padded window, the target is a four-class per-sample
segmentation of the beat on its own lead, and the lead identity is carried alongside so the
network knows which trace it is being asked to delineate.

Three conventions are load bearing, namely,

    context vs supervision   the crop is longer than the labelled window so the network sees
                             real neighbouring signal, but the loss applies only inside that
                             window. everything outside carries IGNORE_INDEX
    inclusive coordinates    win_start_sample and win_end_sample both name samples that
                             belong to the window, so its length is end - start + 1
    data.window_source       table, landmarks or auto. auto is the default and the effective
                             window is recorded per unit in the item metadata, so the choice
                             is auditable rather than silent

"""

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CLASS_INDEX, IGNORE_INDEX, LANDMARKS, LEAD_INDEX, N_LEADS, ROOT, log,
)

try:
    from augmentations import apply_structural
except ImportError:                                    # the module is optional
    apply_structural = None

WAVE_SPANS = [
    ('p', 'p_onset_sample', 'p_offset_sample'),
    ('qrs', 'qrs_onset_sample', 'qrs_offset_sample'),
    ('t', 't_onset_sample', 't_offset_sample'),
]
LANDMARK_COLUMNS = [name + '_sample' for name in LANDMARKS]

REQUIRED_COLUMNS = [
    'record_id', 'lead', 'beat_id', 'split', 'path_raw', 'n_samples',
    'win_start_sample', 'win_end_sample',
] + LANDMARK_COLUMNS


def load_units(units_csv, split=None, keep_disease_classes=None, keep_leads=None, max_units=0,
               drop_record_ids=None):
    """Read a units table and apply the filters the config asks for."""
    import pandas as pd
    path = units_csv if os.path.isabs(units_csv) else os.path.join(ROOT, units_csv)
    if not os.path.isfile(path):
        raise SystemExit('units table not found at %s' % path)
    frame = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SystemExit('%s is missing the columns %s' % (path, missing))
    if split:
        frame = frame[frame['split'] == split]
    if keep_disease_classes and 'disease_class' in frame.columns:
        frame = frame[frame['disease_class'].isin(list(keep_disease_classes))]
    if keep_leads:
        frame = frame[frame['lead'].isin(list(keep_leads))]
    if drop_record_ids:
        frame = frame[~frame['record_id'].isin(set(drop_record_ids))]
    frame = frame.reset_index(drop=True)
    if max_units and len(frame) > max_units:
        frame = frame.iloc[:max_units].reset_index(drop=True)
    return frame


def landmark_span(row):
    """The inclusive span of every finite landmark on a unit, or None when the unit has none."""
    values = []
    for col in LANDMARK_COLUMNS:
        value = row.get(col)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return int(np.floor(min(values))), int(np.ceil(max(values)))


def resolve_window(row, source='auto', pad_samples=20, crop_length=0, n_samples=0):
    """Return the inclusive (lo, hi) window this unit is supervised over.

    See the module docstring for why the table window is not always trusted. The landmark
    definition is only reachable when the unit actually carries landmarks, so a unit with none
    always falls back to the table window regardless of what was asked for.
    """
    lo = int(row['win_start_sample'])
    hi = int(row['win_end_sample'])
    if source == 'table':
        return lo, hi

    span = landmark_span(row)
    if span is None:
        return lo, hi
    a = span[0] - int(pad_samples)
    b = span[1] + int(pad_samples)
    limit = int(n_samples or row.get('n_samples') or 0)
    a = max(0, a)
    if limit:
        b = min(limit - 1, b)
    if b < a:
        return lo, hi

    if source == 'landmarks':
        return a, b
    if source == 'auto':
        # Keep the curated window wherever it fits, narrow only where it cannot.
        if crop_length and (hi - lo + 1) > int(crop_length):
            return a, b
        return lo, hi
    raise ValueError('unknown data.window_source %r, expected table, landmarks or auto' % source)


def build_target(row, crop_start, crop_length, window_span, supervise_window_only=True):
    """Return the per-sample class vector for one unit, in crop coordinates.

    Everything inside the labelled window starts as background, each wave that is present
    overwrites its own span, and everything outside the window is IGNORE_INDEX.
    """
    target = np.full(crop_length, IGNORE_INDEX, dtype=np.int64)
    win_lo, win_hi = window_span
    lo = max(0, win_lo - crop_start)
    hi = min(crop_length - 1, win_hi - crop_start)
    if supervise_window_only:
        if hi >= lo:
            target[lo:hi + 1] = CLASS_INDEX['background']
    else:
        target[:] = CLASS_INDEX['background']

    for wave, on_col, off_col in WAVE_SPANS:
        on, off = row.get(on_col), row.get(off_col)
        if on is None or off is None:
            continue
        if not np.isfinite(on) or not np.isfinite(off):
            continue
        a = int(round(float(on))) - crop_start
        b = int(round(float(off))) - crop_start
        if b < a:
            a, b = b, a
        a = max(a, 0)
        b = min(b, crop_length - 1)
        if b < a:
            continue
        target[a:b + 1] = CLASS_INDEX[wave]
    return target


def landmark_vector(row, crop_start, crop_length):
    """The eleven truth landmarks in crop coordinates, NaN where the landmark is absent."""
    out = np.full(len(LANDMARK_COLUMNS), np.nan, dtype=np.float32)
    for i, col in enumerate(LANDMARK_COLUMNS):
        value = row.get(col)
        if value is None or not np.isfinite(value):
            continue
        pos = float(value) - crop_start
        if -0.5 <= pos <= crop_length - 0.5:
            out[i] = pos
    return out


class SignalStore(object):
    """Serve a record's (12, n) trace from the .npy cache, falling back to the raw CSV."""

    def __init__(self, cache_dir, allow_csv_fallback=True, mmap=True):
        self.cache_dir = cache_dir if os.path.isabs(cache_dir) else os.path.join(ROOT, cache_dir)
        self.allow_csv_fallback = allow_csv_fallback
        self.mmap = mmap
        self._warned = set()

    def get(self, record_id, rel_path):
        dest = os.path.join(self.cache_dir, record_id + '.npy')
        if os.path.isfile(dest):
            return np.load(dest, mmap_mode='r' if self.mmap else None)
        if not self.allow_csv_fallback:
            raise FileNotFoundError('no cached signal for %s at %s, run cache_signals.py' % (record_id, dest))
        if record_id not in self._warned and len(self._warned) < 5:
            self._warned.add(record_id)
            log('warning, reading %s from raw CSV, the .npy cache is missing' % record_id)
        src = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT, rel_path)
        arr = np.loadtxt(src, delimiter=',', dtype=np.float32)
        if arr.shape[0] != N_LEADS and arr.shape[1] == N_LEADS:
            arr = arr.T
        return arr


# The eight independent leads. The augmented limb leads are linear combinations of I and II,
# so including them would weight the frontal plane three times over. This is the lead set of
# Equation 3.4, the curve the human reviewers actually looked at when they placed the landmarks.
MAG_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def spatial_magnitude(window, valid=None):
    """Equation 3.4 over the eight independent leads, baselined on the window median.

    Supplied to the network as a thirteenth input channel. It carries no information the twelve
    leads do not already carry, but it is a Euclidean norm, which a stack of rectified linear
    convolutions cannot form cheaply, and it is identical for all twelve units of a recording, so
    it gives the network a lead-agnostic view of where the beat is to set against the lead-specific
    query of Equation 3.13. It is also, by construction, the view the annotator had.
    """
    x = window[MAG_LEADS]
    if valid is not None and bool(valid.any()):
        base = np.median(x[:, valid], axis=1, keepdims=True)
    else:
        base = np.median(x, axis=1, keepdims=True)
    return np.sqrt(((x - base) ** 2).sum(axis=0)).astype(np.float32)


def normalise(window, mode='robust', eps=1e-6, clip_sigma=20.0, valid=None,
              percentile=99.0):
    """Scale each lead of the crop independently. Returns a float32 array of the same shape.

    The statistics are taken over the valid samples alone. A crop that runs off the end of
    a record is zero padded, and a record shorter than the crop is mostly padding, in which
    case both quartiles fall inside the zero block, the interquartile range collapses to
    zero and every real sample saturates against clip_sigma. Restricting the statistics to
    the valid samples keeps the scaling a property of the signal rather than of the padding.
    """
    if mode in (None, 'none'):
        return window.astype(np.float32, copy=True)
    x = window.astype(np.float32, copy=True)
    ref = x
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.any() and not mask.all():
            ref = x[:, mask]
    if mode == 'zscore':
        centre = ref.mean(axis=1, keepdims=True)
        scale = ref.std(axis=1, keepdims=True)
    elif mode == 'robust':
        centre = np.median(ref, axis=1, keepdims=True)
        q75 = np.percentile(ref, 75, axis=1, keepdims=True)
        q25 = np.percentile(ref, 25, axis=1, keepdims=True)
        scale = (q75 - q25) / 1.349          # IQR to a standard deviation for a normal variate
    elif mode in ('percentile', 'abs_percentile'):
        # Unlike an IQR, a high absolute-deviation percentile does not collapse when a
        # single-beat record contains more than 50% quiet tail. This mode is intended as a
        # separately trained ablation; changing a trained checkpoint's scale at inference is
        # not valid.
        centre = np.median(ref, axis=1, keepdims=True)
        q = float(percentile)
        if not 50.0 <= q <= 100.0:
            raise ValueError('normalisation.percentile must lie in [50, 100], got %r' % q)
        scale = np.percentile(np.abs(ref - centre), q, axis=1, keepdims=True)
    else:
        raise ValueError('unknown normalisation mode %r' % mode)
    scale = np.maximum(scale, eps)
    x = (x - centre) / scale
    if clip_sigma:
        np.clip(x, -clip_sigma, clip_sigma, out=x)
    return x


P_CLASS_INDEX = 1          # BG 0, P 1, QRS 2, T 3


class BeatWindowDataset(Dataset):
    """One item per delineation unit."""

    def __init__(self, frame, cache_dir, crop_length=1280, normalisation=None, augment=None,
                 training=False, allow_csv_fallback=True, context_mode='real',
                 supervise_window_only=True, pad_value=0.0, pad_mode='zero',
                 structural=None, eval_structural=None, seed=0,
                 window_source='auto', window_pad_ms=40.0, fs_hz=500.0,
                 magnitude_channel=False, validity_channel=False,
                 record_start_at_sample=None):
        self.frame = frame.reset_index(drop=True)
        self.records = self.frame.to_dict('records')
        self.store = SignalStore(cache_dir, allow_csv_fallback=allow_csv_fallback)
        self.crop_length = int(crop_length)
        self.norm = dict(normalisation or {'mode': 'robust'})
        self._qm = self._load_quantile_match(self.norm.get('quantile_match'))
        self.augment = dict(augment or {})
        self.training = bool(training)
        self.context_mode = context_mode
        self.supervise_window_only = bool(supervise_window_only)
        self.pad_value = float(pad_value)
        self.pad_mode = str(pad_mode)
        self.structural = dict(structural or {})
        # Structural transforms applied at scoring time rather than training time. This exists
        # for one purpose, which is to put an in-distribution unit into the observation geometry
        # of the external corpus and ask whether the model still delineates it. Training and
        # scoring otherwise stay separate, and this stays empty unless a diagnostic sets it.
        self.eval_structural = dict(eval_structural or {})
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_resample = bool(self.augment.get('epoch_resample', False))
        self.window_source = str(window_source or 'auto')
        self.magnitude_channel = bool(magnitude_channel)
        self.validity_channel = bool(validity_channel)
        self.record_start_at_sample = (None if record_start_at_sample is None
                                       else int(record_start_at_sample))
        self.fs_hz = float(fs_hz or 500.0)
        self.window_pad_samples = int(round(float(window_pad_ms or 0.0) * self.fs_hz / 1000.0))
        self.windows = self._resolve_windows()
        self._check_windows()

    @staticmethod
    def _load_quantile_match(path):
        """Load a test-time quantile-matching table (CSV), or return None when unset.

        Produced by build_quantile_match.py. Columns are ``level, external, training`` for a
        single global mapping, or ``lead, level, external, training`` for a per-lead mapping
        (one block per lead, in the canonical order the leads occupy in the signal). At each
        quantile ``level`` an ``external`` normalised amplitude maps onto the ``training`` value.
        Applied ONLY to the twelve lead channels, after normalisation, and never to an
        in-distribution run (where it is simply not configured). It aligns the amplitude
        distribution of a shifted domain onto the one the network was trained under, without
        changing a single weight. Returns (external, training) as (Q,) global or (12, Q) per-lead.
        """
        if not path:
            return None
        import csv as _csv
        with open(path, newline='') as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            raise ValueError('quantile_match table %s is empty' % path)
        cols = set(rows[0].keys())
        if not {'external', 'training'} <= cols:
            raise ValueError("quantile_match CSV %s needs 'external' and 'training' columns" % path)
        if 'lead' in cols:
            order, buf = [], {}
            for r in rows:                       # preserve lead block order as written
                lead = r['lead']
                if lead not in buf:
                    buf[lead] = ([], []); order.append(lead)
                buf[lead][0].append(float(r['external']))
                buf[lead][1].append(float(r['training']))
            src = np.array([buf[l][0] for l in order], dtype=np.float64)
            ref = np.array([buf[l][1] for l in order], dtype=np.float64)
        else:
            src = np.array([float(r['external']) for r in rows], dtype=np.float64)
            ref = np.array([float(r['training']) for r in rows], dtype=np.float64)
        return (src, ref)

    def _apply_quantile_match(self, window):
        """Map the twelve lead channels through the loaded quantile table. Magnitude untouched."""
        src, ref = self._qm
        x = window.astype(np.float32, copy=True)
        n = min(12, x.shape[0])
        if src.ndim == 1:
            x[:n] = np.interp(x[:n], src, ref).astype(np.float32)
        else:
            for i in range(n):
                x[i] = np.interp(x[i], src[i], ref[i]).astype(np.float32)
        return x

    def _resolve_windows(self):
        return [resolve_window(row, source=self.window_source,
                               pad_samples=self.window_pad_samples,
                               crop_length=self.crop_length)
                for row in self.records]

    def _check_windows(self):
        table = (self.frame['win_end_sample'] - self.frame['win_start_sample'] + 1).astype(int)
        widths = np.array([hi - lo + 1 for lo, hi in self.windows], dtype=np.int64) \
            if self.windows else np.zeros(0, dtype=np.int64)
        widest = int(widths.max()) if widths.size else 0
        narrowed = int((widths < table.to_numpy()).sum()) if widths.size else 0
        if narrowed:
            log('window_source %s narrowed %d of %d units whose curated beat span disagreed with '
                'their own fiducials, widest effective window now %d samples'
                % (self.window_source, narrowed, len(widths), widest))
        if widest > self.crop_length:
            raise SystemExit(
                'crop_length %d is shorter than the widest effective window %d under '
                'window_source %s. Either raise crop_length to the next multiple of 32 above %d '
                'or set data.window_source to landmarks.'
                % (self.crop_length, widest, self.window_source, widest))

    def __len__(self):
        return len(self.records)

    def _rng(self, index):
        # A per-item, per-epoch generator keeps augmentation reproducible under any worker count
        # without freezing every unit to the same transform for the whole training run.
        epoch = self.epoch if (self.training and self.epoch_resample) else 0
        return np.random.default_rng(
            (self.seed * 1000003 + int(index) + int(epoch) * 9176) % (2 ** 32))

    def set_epoch(self, epoch):
        """Select the deterministic augmentation draw used for the next training epoch."""
        self.epoch = int(epoch)

    def _crop_start(self, row, rng, window):
        if not self.training and self.record_start_at_sample is not None:
            # A single-beat record can be placed without looking at any reference landmark:
            # record sample zero always lands at the configured crop coordinate.
            return -self.record_start_at_sample
        win_lo, win_hi = int(window[0]), int(window[1])
        width = win_hi - win_lo + 1
        free = max(0, self.crop_length - width)
        left = free // 2
        jitter = int(self.augment.get('time_jitter_samples', 0) or 0) if self.training else 0
        if jitter > 0 and free > 0:
            left = int(np.clip(left + rng.integers(-jitter, jitter + 1), 0, free))
        start = win_lo - left
        n_samples = int(row.get('n_samples') or 0)
        if n_samples >= self.crop_length:
            start = int(np.clip(start, 0, n_samples - self.crop_length))
        else:
            start = min(start, 0) if start < 0 else start
        return int(start)

    def _slice(self, signal, crop_start):
        """Take the crop, zero padding wherever it runs off the end of the record."""
        n_samples = signal.shape[1]
        out = np.full((N_LEADS, self.crop_length), self.pad_value, dtype=np.float32)
        valid = np.zeros(self.crop_length, dtype=bool)
        src_lo = max(0, crop_start)
        src_hi = min(n_samples, crop_start + self.crop_length)
        if src_hi > src_lo:
            dst_lo = src_lo - crop_start
            dst_hi = dst_lo + (src_hi - src_lo)
            out[:, dst_lo:dst_hi] = np.asarray(signal[:, src_lo:src_hi], dtype=np.float32)
            valid[dst_lo:dst_hi] = True
            # Optional edge padding. A record that begins at ventricular activation has no
            # signal to the left of its QRS onset, and a zero block there is a step change the
            # convolutions read as a deflection. Replicating the first and last real sample
            # removes the step without inventing morphology. Default is unchanged.
            if getattr(self, 'pad_mode', 'zero') == 'edge':
                if dst_lo > 0:
                    out[:, :dst_lo] = out[:, dst_lo:dst_lo + 1]
                if dst_hi < self.crop_length:
                    out[:, dst_hi:] = out[:, dst_hi - 1:dst_hi]
        return out, valid

    def _apply_augment(self, window, lead_idx, rng):
        cfg = self.augment
        if not self.training or not cfg.get('enabled', False):
            return window
        amp = float(cfg.get('amplitude_scale', 0.0) or 0.0)
        if amp > 0:
            window = window * (1.0 + rng.uniform(-amp, amp))
        wander = float(cfg.get('baseline_wander_mv', 0.0) or 0.0)
        if wander > 0:
            hz = float(cfg.get('baseline_wander_hz', 0.5) or 0.5)
            n = window.shape[1]
            t = np.arange(n, dtype=np.float32) / 500.0
            phase = rng.uniform(0, 2 * np.pi, size=(N_LEADS, 1)).astype(np.float32)
            window = window + wander * np.sin(2 * np.pi * hz * t[None, :] + phase).astype(np.float32)
        noise = float(cfg.get('gaussian_noise_mv', 0.0) or 0.0)
        if noise > 0:
            window = window + rng.normal(0.0, noise, size=window.shape).astype(np.float32)
        drop = float(cfg.get('lead_dropout_p', 0.0) or 0.0)
        if drop > 0:
            mask = rng.random(N_LEADS) < drop
            mask[lead_idx] = False      # never drop the lead being delineated
            window[mask, :] = 0.0
        return window.astype(np.float32, copy=False)

    def __getitem__(self, index):
        row = self.records[index]
        rng = self._rng(index)
        lead = str(row['lead'])
        if lead not in LEAD_INDEX:
            raise KeyError('unit %d names lead %r which is not one of the twelve' % (index, lead))
        lead_idx = LEAD_INDEX[lead]

        signal = self.store.get(str(row['record_id']), str(row['path_raw']))
        span = self.windows[index]
        crop_start = self._crop_start(row, rng, span)
        window, valid = self._slice(signal, crop_start)

        win_lo = int(span[0]) - crop_start
        win_hi = int(span[1]) - crop_start

        if self.context_mode == 'pad':
            keep = np.zeros(self.crop_length, dtype=bool)
            keep[max(0, win_lo):min(self.crop_length, win_hi + 1)] = True
            window[:, ~keep] = self.pad_value
            valid = valid & keep

        # The target and the landmark vector are built before the structural transforms, since
        # those transforms have to move the signal, the validity mask, the target and the
        # landmarks together. Building the target afterwards would silently undo them.
        target = build_target(row, crop_start, self.crop_length,
                              (int(span[0]), int(span[1])),
                              supervise_window_only=self.supervise_window_only)
        marks = landmark_vector(row, crop_start, self.crop_length)

        applied = {}
        active = self.structural if self.training else self.eval_structural
        if active and apply_structural is not None:
            window, valid, target, marks, applied = apply_structural(
                window, valid, target, marks, win_lo, win_hi, active, rng,
                fs_hz=float(self.fs_hz))

        raw_trace = window[lead_idx].copy()      # voltages before scaling, used for peak reading
        window = self._apply_augment(window, lead_idx, rng)
        if self.magnitude_channel:
            # Computed after augmentation so that a lead dropped by the augmentation is dropped
            # from the curve too. Computing it first would let the curve leak the dropped lead
            # straight back in and quietly defeat the augmentation.
            window = np.vstack([window, spatial_magnitude(window, valid)[None, :]])
        # normalisation.scope names the samples the per-lead statistics are taken over.
        #
        #   crop     every valid sample of the crop, the historical behaviour and the default
        #   window   the effective beat window plus scope_margin_ms on each side, widened to
        #            cover any wave sample the structural transforms moved past the table window
        #
        # Why window exists: with crop statistics the same beat's normalised T amplitude is a
        # median 3.3 times larger when the beat is isolated than when its neighbours are in the
        # crop, because the quiet content shrinks the interquartile range. The model's T-offset
        # errors concentrate four to one in the units whose T comes out small, so the crop's
        # incidental composition sets the beat's salience. Window statistics give the same beat
        # the same gain whatever surrounds it. Note for the label-free external protocol: the
        # window derives from the units table, so a strictly label-free run should keep
        # scope: crop in its eval config and accept the (bounded, single-beat) mismatch.
        stats_valid = valid
        scope = str(self.norm.get('scope', 'crop')).lower()
        if scope not in ('crop', 'window'):
            raise ValueError("normalisation.scope %r is not one of crop or window" % scope)
        if scope == 'window':
            # Margin 0 is the default and gives exact gain invariance: the window's content is
            # identical whether the beat is isolated or in natural context, so the statistics
            # are too (measured: gain median 1.00, IQR 1.00-1.00 at 0 ms; 1.48, 1.13-1.86 at
            # 120 ms, where the margin reads flat tail in one case and live neighbour in the
            # other). The effective window already carries data.window_pad_ms of context, and
            # the wave-span widening below covers anything a structural transform moved.
            margin = int(round(float(self.norm.get('scope_margin_ms', 0.0) or 0.0)
                               * self.fs_hz / 1000.0))
            span_lo, span_hi = win_lo, win_hi
            wave = np.flatnonzero((target != IGNORE_INDEX) & (target > 0))
            if wave.size:
                span_lo = min(span_lo, int(wave[0]))
                span_hi = max(span_hi, int(wave[-1]))
            mask = np.zeros_like(valid)
            mask[max(0, span_lo - margin):min(self.crop_length, span_hi + 1 + margin)] = True
            mask &= valid
            if int(mask.sum()) >= 32:
                stats_valid = mask
        window = normalise(window, mode=self.norm.get('mode', 'robust'),
                           eps=float(self.norm.get('eps', 1e-6)),
                           clip_sigma=float(self.norm.get('clip_sigma', 0.0) or 0.0),
                           valid=stats_valid,
                           percentile=float(self.norm.get('percentile', 99.0)))
        if self._qm is not None:
            # Test-time domain alignment: reshape the normalised lead amplitudes onto the
            # training distribution. Applied to the 12 leads only, before edge padding so the
            # replicated boundary samples carry the matched values. Off for in-distribution.
            window = self._apply_quantile_match(window)
        if getattr(self, 'pad_mode', 'zero') == 'edge' and valid.any() and not valid.all():
            # Preserve the edge padding that _slice replicated into the off-record margins so
            # pad_mode 'edge' is not silently undone here. Normalisation used valid-only stats,
            # so replicate the *normalised* boundary sample outward into the two off-record
            # margins. Any interior invalid run (e.g. context_mode 'pad', which zeros the
            # neighbours on purpose) is still zeroed, matching _slice, which only edge-fills the
            # outer margins. Without this the replicated padding was overwritten with zeros here,
            # reintroducing the very step change at the record edge that pad_mode 'edge' removes.
            idx = np.flatnonzero(valid)
            lo, hi = int(idx[0]), int(idx[-1])
            interior = ~valid
            interior[:lo] = False
            interior[hi + 1:] = False
            window[:, interior] = 0.0
            if lo > 0:
                window[:, :lo] = window[:, lo:lo + 1]
            if hi < self.crop_length - 1:
                window[:, hi + 1:] = window[:, hi:hi + 1]
        else:
            window[:, ~valid] = 0.0
        # The validity channel is appended last and is never normalised. Its valid samples are
        # all one, so a robust normaliser would find a zero scale and either collapse it or
        # divide it by eps. Appending after normalisation and after the zeroing above also
        # means the channel says exactly what it should, which is that a flat run inside the
        # record and a run that was never recorded are different things even though both hold
        # zero volts in every lead.
        if self.validity_channel:
            window = np.vstack([window, valid.astype(np.float32)[None, :]])
        target[~valid] = IGNORE_INDEX

        supervised = np.zeros(self.crop_length, dtype=bool)
        supervised[max(0, win_lo):min(self.crop_length, win_hi + 1)] = True
        supervised &= valid

        # The supervised isolated tail, for loss.tail_background_weight. Read off the finished
        # target rather than tracked through the transforms, so every later shift (stretch,
        # jitter, warp) is already accounted for: the trailing run of supervised background
        # after the final wave sample, present only when the beat was isolated. All zeros
        # otherwise, and the loss ignores it unless the config sets the weight above one.
        tail = np.zeros(self.crop_length, dtype=bool)
        if applied.get('isolate_beat'):
            wave_idx = np.flatnonzero(target > 0)
            if wave_idx.size:
                tail = ((np.arange(self.crop_length) > int(wave_idx[-1]))
                        & valid & (target == CLASS_INDEX['background']))

        # The auxiliary target is read off the finished arrays rather than kept as a separate
        # flag, so it cannot drift out of step with what the transforms actually did.
        p_observable = bool(np.any(target == P_CLASS_INDEX))
        truncated = bool(applied.get('truncated', False)) or bool((~valid).any())
        cell = ('B' if p_observable else 'D') if truncated else ('A' if p_observable else 'C')

        item = {
            'p_observable': torch.tensor(float(p_observable), dtype=torch.float32),
            'cell_idx': torch.tensor('ABCD'.index(cell), dtype=torch.long),
            'signal': torch.from_numpy(np.ascontiguousarray(window)),
            'trace': torch.from_numpy(np.ascontiguousarray(raw_trace)),
            'target': torch.from_numpy(target),
            'valid': torch.from_numpy(valid),
            'supervised': torch.from_numpy(supervised),
            'tail': torch.from_numpy(tail),
            'lead_idx': torch.tensor(lead_idx, dtype=torch.long),
            'landmarks': torch.from_numpy(marks),
            'crop_start': torch.tensor(crop_start, dtype=torch.long),
            'index': torch.tensor(index, dtype=torch.long),
        }
        item['meta'] = {
            'record_id': str(row['record_id']),
            'lead': lead,
            'beat_id': int(row['beat_id']) if np.isfinite(row.get('beat_id', np.nan)) else -1,
            'disease_class': str(row.get('disease_class', '')),
            'label_source': str(row.get('label_source', '')),
            'structural': applied,
            'cell': cell,
            'p_observable': p_observable,
            'crop_start': crop_start,
            'win_start_sample': int(row['win_start_sample']),
            'win_end_sample': int(row['win_end_sample']),
            'eff_start_sample': int(span[0]),
            'eff_end_sample': int(span[1]),
            'window_source': self.window_source,
            'fs_hz': float(row.get('fs_hz', 500) or 500),
        }
        return item


def collate(batch):
    """Stack the tensors and keep the metadata as a plain list of dicts."""
    out = {}
    for key in batch[0]:
        if key == 'meta':
            out['meta'] = [item['meta'] for item in batch]
        else:
            out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def _eval_structural(cfg):
    """The diagnostic transform applied at scoring time, or None when none is asked for.

    eval.isolate_beat_ms puts a unit into the external corpus's observation geometry, meaning one
    beat with its neighbours flattened away and a quiet tail of the stated length behind it. It
    answers a question no amount of external scoring can, which is whether the model loses its
    repolarisation timing because of that geometry or because of the morphology it has never
    seen. Scoring an in-distribution unit both ways separates the two.
    """
    ms = cfg.get('eval', {}).get('isolate_beat_ms')
    if ms is None or float(ms) < 0:
        return None
    return {'isolate_beat_p': 1.0, 'isolate_tail_ms': float(ms), 'isolate_supervise_tail': False}


def dataset_kwargs(cfg, training=False, **overrides):
    """Every BeatWindowDataset argument that comes from a config, in one place.

    build_loaders, evaluate.py and the diagnostic scripts all read from here. Keeping two copies
    of this list is what let data.magnitude_channel reach training and silently miss scoring,
    which cost a run, so there is now one copy and adding a key to it reaches every caller.
    """
    data = cfg['data']
    kw = dict(
        cache_dir=data['signal_cache'],
        crop_length=int(data['crop_length']),
        normalisation=cfg.get('normalisation'),
        allow_csv_fallback=bool(data.get('allow_csv_fallback', True)),
        context_mode=data.get('context_mode', 'real'),
        supervise_window_only=bool(data.get('supervise_window_only', True)),
        pad_value=float(data.get('pad_value', 0.0)),
        pad_mode=data.get('pad_mode', 'zero'),
        structural=cfg.get('structural_augment') if training else None,
        eval_structural=(None if training else _eval_structural(cfg)),
        seed=int(cfg.get('run', {}).get('seed', 0)),
        window_source=data.get('window_source', 'auto'),
        window_pad_ms=float(data.get('window_pad_ms', 40.0)),
        fs_hz=float(data.get('fs_hz', 500.0)),
        magnitude_channel=bool(data.get('magnitude_channel', False)),
        validity_channel=bool(data.get('validity_channel', False)),
        record_start_at_sample=(None if training else
                                cfg.get('eval', {}).get('record_start_at_sample')),
    )
    kw.update(overrides)
    return kw


def build_loaders(cfg, stage_frames=None):
    """Return the train and validation loaders described by a merged config."""
    from torch.utils.data import DataLoader

    data = cfg['data']
    frames = stage_frames or {}
    train_frame = frames.get('train')
    val_frame = frames.get('val')
    if train_frame is None:
        train_frame = load_units(data['units_csv'], split=data.get('train_split', 'train'),
                                 keep_disease_classes=data.get('keep_disease_classes'),
                                 keep_leads=data.get('keep_leads'),
                                 max_units=int(data.get('max_units', 0) or 0))
    if val_frame is None:
        val_frame = load_units(data['units_csv'], split=data.get('val_split', 'val'),
                               keep_disease_classes=data.get('keep_disease_classes'),
                               keep_leads=data.get('keep_leads'),
                               max_units=int(data.get('max_units', 0) or 0))

    overlap = set(train_frame['record_id']) & set(val_frame['record_id'])
    if overlap:
        raise SystemExit('%d record ids appear in both splits, which leaks near duplicate beats. '
                         'The split is meant to be by record.' % len(overlap))

    common_kwargs = dataset_kwargs(cfg, training=True)
    train_set = BeatWindowDataset(train_frame, augment=cfg.get('augment'), training=True, **common_kwargs)
    val_kwargs = dict(common_kwargs, structural=None)
    val_set = BeatWindowDataset(val_frame, augment=None, training=False, **val_kwargs)

    workers = int(data.get('num_workers', 0) or 0)
    loader_kwargs = dict(num_workers=workers, collate_fn=collate,
                         pin_memory=bool(data.get('pin_memory', False)))
    if workers > 0:
        # With persistent workers each process keeps the epoch value it was spawned with. Turn
        # persistence off only for epoch-resampled augmentation so the fresh deterministic draw
        # reaches every worker; legacy/static configs keep the faster previous behaviour.
        loader_kwargs['persistent_workers'] = not train_set.epoch_resample
        loader_kwargs['prefetch_factor'] = int(data.get('prefetch_factor', 2) or 2)

    train_loader = DataLoader(train_set, batch_size=int(cfg['train']['batch_size']), shuffle=True,
                              drop_last=bool(data.get('drop_last', True)), **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=int(cfg['train'].get('eval_batch_size',
                                                                    cfg['train']['batch_size'])),
                            shuffle=False, drop_last=False, **loader_kwargs)
    log('train units %d over %d records, val units %d over %d records'
        % (len(train_set), train_frame['record_id'].nunique(),
           len(val_set), val_frame['record_id'].nunique()))
    return train_loader, val_loader


def build_geometry_validation_loaders(cfg):
    """Small deterministic source-domain panels with prescribed quiet-tail geometry.

    Ordinary validation masks everything beyond the annotated beat window, so it cannot see a
    T region that runs into a long quiet tail. When train.geometry_validation is configured,
    this returns one loader per requested tail length. By default it keeps one unit per record,
    making the panel record-diverse without repeating near-identical beats and leads.
    """
    from torch.utils.data import DataLoader

    spec = dict(cfg.get('train', {}).get('geometry_validation') or {})
    tails = spec.get('tail_ms') or []
    if not spec.get('enabled', False) or not tails:
        return {}
    if not isinstance(tails, (list, tuple)):
        tails = [tails]

    data = cfg['data']
    frame = load_units(data['units_csv'], split=data.get('val_split', 'val'),
                       keep_disease_classes=data.get('keep_disease_classes'),
                       keep_leads=data.get('keep_leads'))
    if bool(spec.get('one_per_record', True)):
        frame = frame.sort_values(['record_id', 'lead', 'beat_id']).drop_duplicates('record_id')
        frame = frame.reset_index(drop=True)
    limit = int(spec.get('max_units', 0) or 0)
    if limit and len(frame) > limit:
        frame = frame.iloc[:limit].reset_index(drop=True)

    workers = int(spec.get('num_workers', 0) or 0)
    batch_size = int(spec.get('batch_size', cfg['train'].get('eval_batch_size', 64)))
    loaders = {}
    for tail in tails:
        kwargs = dataset_kwargs(cfg, training=False)
        kwargs['record_start_at_sample'] = None
        kwargs['eval_structural'] = {
            'isolate_beat_p': 1.0,
            'isolate_tail_ms': float(tail),
            'isolate_supervise_tail': False,
        }
        panel = BeatWindowDataset(frame, augment=None, training=False, **kwargs)
        loaders[float(tail)] = DataLoader(panel, batch_size=batch_size, shuffle=False,
                                          num_workers=workers, collate_fn=collate)
    log('geometry validation uses %d units over %d records at tails %s ms'
        % (len(frame), frame['record_id'].nunique(), ', '.join('%g' % float(x) for x in tails)))
    return loaders
