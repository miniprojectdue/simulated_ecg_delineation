"""
augmentations.py  -  structural transforms that manufacture alternative observation
conditions from MedalCare-XL data.

Five transforms: record-edge truncation, isolated beat, P masking, repolarisation stretch
and T-offset jitter. Every parameter is derived from MedalCare-XL's own measured
morphology; no external fiducial annotation sets any value here, and none is used in the
objective, in checkpoint selection or in any parameter fit.

Each transform acts on the crop before normalisation and keeps four things consistent with
one another: the twelve-lead window, the validity mask, the per-sample target and the
eleven-element landmark vector. Moving one without the others silently corrupts the
supervision.

Landmark order is the canonical one from common.LANDMARKS, so indices 0 to 2 are the P wave
and 8 to 10 are the T onset, peak and offset.

"""
import numpy as np

P_LANDMARK_IDX = (0, 1, 2)
T_OFFSET_IDX = 10
T_PEAK_IDX = 9
BACKGROUND, P_CLASS, QRS_CLASS, T_CLASS = 0, 1, 2, 3
T_ONSET_IDX = 8


def _isoelectric(window, valid, lo, hi):
    """A per-lead isoelectric level taken from the valid samples outside a span."""
    mask = valid.copy()
    mask[max(0, lo):min(mask.size, hi + 1)] = False
    if not mask.any():
        return np.zeros((window.shape[0], 1), dtype=window.dtype)
    return np.median(window[:, mask], axis=1, keepdims=True).astype(window.dtype)


def isolate_beat(window, valid, win_lo, win_hi, pad_value=0.0, keep_tail_ms=0.0, fs_hz=500.0,
                 target=None, supervise_tail=True):
    """Replace the neighbouring beats with the isoelectric level, leaving one beat standing.

    The external records hold a single beat followed by a flat post-T tail and nothing before
    it, so the signal after the window is flattened rather than removed while the signal before
    it is removed outright.

    The tail this builds is background by construction. It was written from the isoelectric
    level and holds no wave at all, so labelling it background states a fact rather than an
    assumption. Doing that is the difference between showing the network the external corpus's
    geometry and telling it what the geometry means. Without it the tail lies outside the
    supervised window, every sample of it carries the ignore index, and the loss never scores
    one of them. The network was shown a quiet post-T run on thirty per cent of pretraining
    crops and was never once asked to call it background, which is why its T region runs on
    without stopping when the record ends after its own T and no next beat arrives to end it.

    Returns the window, the validity mask and the target. The target comes back unchanged when
    none is passed or when supervise_tail is false, which reproduces the earlier behaviour.
    """
    window = window.copy()
    valid = valid.copy()
    if target is not None:
        target = target.copy()
    n = window.shape[1]
    level = _isoelectric(window, valid, win_lo, win_hi)

    lo = max(0, int(win_lo))
    hi = min(n - 1, int(win_hi))
    if hi < lo:
        # Nothing to isolate. This returns three values like every other exit, since the caller
        # unpacks three and a two value return here would be an unpacking error on whichever
        # unit first has a window the crop cannot hold.
        return window, valid, target

    # ahead of the beat there is nothing at all
    if lo > 0:
        window[:, :lo] = pad_value
        valid[:lo] = False

    # behind the beat a flat tail of real signal, then nothing
    tail = int(round(float(keep_tail_ms) * float(fs_hz) / 1000.0))
    tail_end = min(n, hi + 1 + max(0, tail))
    if tail_end > hi + 1:
        window[:, hi + 1:tail_end] = level
        if target is not None and supervise_tail:
            target[hi + 1:tail_end] = BACKGROUND
    if tail_end < n:
        window[:, tail_end:] = pad_value
        valid[tail_end:] = False
    return window, valid, target


def left_edge(window, valid, win_lo, lead_in_ms=4.0, fs_hz=500.0, pad_value=0.0):
    """Truncate the signal so the record appears to begin just before the supervised window.

    lead_in_ms is how much real signal is left standing ahead of the window. The external
    corpus offers a median of two milliseconds, so a small value here reproduces its hardest
    property. Everything earlier becomes padding and leaves the validity mask.
    """
    window = window.copy()
    valid = valid.copy()
    lead_in = int(round(float(lead_in_ms) * float(fs_hz) / 1000.0))
    cut = max(0, int(win_lo) - lead_in)
    if cut > 0:
        window[:, :cut] = pad_value
        valid[:cut] = False
    return window, valid


def remove_p_wave(window, valid, target, landmarks, fs_hz=500.0, blend_ms=6.0):
    """Blank the P wave in every lead and relabel its span as background.

    The span is taken from the target rather than from the landmark vector, so a unit whose P
    is absent or already outside the crop is left untouched. The blanked segment is filled by a
    straight line between the two endpoints rather than by a constant, which avoids introducing
    a step the convolutions would read as a deflection. The three P landmarks are cleared, so
    the unit no longer asserts a P wave anywhere.
    """
    idx = np.flatnonzero(target == P_CLASS)
    if idx.size == 0:
        return window, valid, target, landmarks, False

    window = window.copy()
    target = target.copy()
    landmarks = landmarks.copy()
    lo, hi = int(idx[0]), int(idx[-1])
    n = window.shape[1]
    blend = max(1, int(round(float(blend_ms) * float(fs_hz) / 1000.0)))
    a = max(0, lo - blend)
    b = min(n - 1, hi + blend)
    if b <= a:
        return window, valid, target, landmarks, False

    left = window[:, a:a + 1]
    right = window[:, b:b + 1]
    ramp = np.linspace(0.0, 1.0, b - a + 1, dtype=window.dtype)[None, :]
    window[:, a:b + 1] = left * (1.0 - ramp) + right * ramp

    target[lo:hi + 1] = BACKGROUND
    for i in P_LANDMARK_IDX:
        landmarks[i] = np.nan
    return window, valid, target, landmarks, True


def jitter_t_offset(target, landmarks, shift_samples):
    """Move the T offset label by a whole number of samples, signal untouched.

    Two published conventions for the same waveform differ by 17.57 ms on this corpus, so a
    network trained on one is systematically wrong against the other. Perturbing the label
    within that range teaches the network that the T offset carries definitional uncertainty
    rather than a single correct answer. The shift is refused where it would cross the T peak
    or run outside the supervised region, so no unit is made incoherent.
    """
    shift = int(shift_samples)
    if shift == 0:
        return target, landmarks
    idx = np.flatnonzero(target == 3)
    if idx.size == 0:
        return target, landmarks
    target = target.copy()
    landmarks = landmarks.copy()
    lo, hi = int(idx[0]), int(idx[-1])
    new_hi = hi + shift
    peak = landmarks[T_PEAK_IDX]
    floor = lo if not np.isfinite(peak) else max(lo, int(peak) + 1)
    supervised = np.flatnonzero(target != -100)
    if supervised.size == 0:
        return target, landmarks
    ceiling = int(supervised[-1])
    new_hi = int(np.clip(new_hi, floor, ceiling))
    if new_hi == hi:
        return target, landmarks

    if new_hi > hi:
        target[hi + 1:new_hi + 1] = 3
    else:
        target[new_hi + 1:hi + 1] = BACKGROUND
    if np.isfinite(landmarks[T_OFFSET_IDX]):
        landmarks[T_OFFSET_IDX] = float(new_hi)
    return target, landmarks


def time_warp(window, target, landmarks, valid, factor):
    """Resample the crop in time by a small factor, carrying the labels with it.

    Lower confidence than the other four and off by default. MonoAlg3D beats have different QRS
    and QT proportions from MedalCare-XL, and this widens the range of proportions the network
    has seen. Run it as its own arm rather than folding it in, so its effect is separable.
    """
    factor = float(factor)
    if abs(factor - 1.0) < 1e-6:
        return window, target, landmarks, valid
    n = window.shape[1]
    src = np.clip(np.arange(n, dtype=np.float64) / factor, 0, n - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (src - lo).astype(window.dtype)
    window = window[:, lo] * (1.0 - frac)[None, :] + window[:, hi] * frac[None, :]
    target = target[np.clip(np.round(src).astype(int), 0, n - 1)]
    valid = valid[np.clip(np.round(src).astype(int), 0, n - 1)]
    landmarks = np.where(np.isfinite(landmarks), landmarks * factor, landmarks)
    landmarks = np.where(np.isfinite(landmarks), np.clip(landmarks, 0, n - 1), landmarks)
    return window, target, landmarks, valid



# ======================================================================================
# Context and morphology transforms added for the context-aware retrain.
#
# The three below replace left_edge and extend remove_p_wave. Their purpose is to make two
# properties of a training crop independent of one another, which they were not before.
#
#     whether the record was truncated        reported to the network by the validity channel
#     whether a P wave is observable          the auxiliary head's target
#
# In the previous setup a P was observable on 100 per cent of the 185,022 units and became
# unobservable only when truncation fired, so the two were the same event and any classifier
# could read the second off the first. Drawing them separately is what forces the auxiliary
# head to look at the waveform.
# ======================================================================================


def _first_last(target, cls):
    idx = np.flatnonzero(target == cls)
    return (int(idx[0]), int(idx[-1])) if idx.size else (None, None)


def _isoelectric_donor(window, valid, target, length, exclude, rng):
    """A background run of the requested length, taken from the same crop and detrended.

    Returns None when the crop holds no background run long enough, which leaves the caller to
    fall back on the plain interpolation.
    """
    ok = (target == BACKGROUND) & valid
    lo_x, hi_x = exclude
    ok[max(0, lo_x - 2):hi_x + 3] = False
    if ok.sum() < length:
        return None
    # longest contiguous run, then a random offset inside it
    runs, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(ok)))
    runs = [r for r in runs if r[1] - r[0] >= length]
    if not runs:
        return None
    a, b = runs[int(rng.integers(len(runs)))]
    off = a + int(rng.integers(0, b - a - length + 1))
    seg = np.asarray(window[:, off:off + length], dtype=np.float64)
    # remove the local linear trend so only the residual variation is carried across
    t = np.linspace(0.0, 1.0, length)
    for i in range(seg.shape[0]):
        m = (seg[i, -1] - seg[i, 0])
        seg[i] = seg[i] - (seg[i, 0] + m * t)
    return seg


def mask_p_wave(window, valid, target, landmarks, rng, fs_hz=500.0, blend_ms=6.0,
                matched_residual=True):
    """Remove the P wave from an otherwise intact record and relabel its span as background.

    This differs from remove_p_wave in one respect that matters. A straight line between the
    two endpoints carries exactly zero high frequency content, and no genuine isoelectric
    segment does, so a classifier asked whether a P is observable can answer by measuring the
    second difference rather than by looking for atrial activity. The replacement here is the
    same straight line plus the detrended residual of a real background run taken from the same
    crop and the same leads, which leaves the segment continuous at both ends and statistically
    ordinary in between.
    """
    lo, hi = _first_last(target, P_CLASS)
    if lo is None:
        return window, valid, target, landmarks, False

    window = window.copy()
    target = target.copy()
    landmarks = landmarks.copy()
    n = window.shape[1]
    blend = max(1, int(round(float(blend_ms) * float(fs_hz) / 1000.0)))
    a = max(0, lo - blend)
    b = min(n - 1, hi + blend)
    if b <= a:
        return window, valid, target, landmarks, False

    span = b - a + 1
    ramp = np.linspace(0.0, 1.0, span, dtype=window.dtype)[None, :]
    fill = window[:, a:a + 1] * (1.0 - ramp) + window[:, b:b + 1] * ramp
    if matched_residual:
        donor = _isoelectric_donor(window, valid, target, span, (a, b), rng)
        if donor is not None:
            fill = fill + donor.astype(window.dtype)
    window[:, a:b + 1] = fill

    target[lo:hi + 1] = BACKGROUND
    for i in P_LANDMARK_IDX:
        landmarks[i] = np.nan
    return window, valid, target, landmarks, True


def truncate_context(window, valid, target, landmarks, mode, lead_in_ms,
                     fs_hz=500.0, pad_value=0.0):
    """Cut the record so it appears to begin partway through the cardiac cycle.

    mode 'pre_p' cuts a stated distance ahead of the P onset, so the record is truncated and the
    P survives. mode 'qrs_edge' cuts a stated distance ahead of the QRS onset, so the record is
    truncated and the P is gone. The external corpus is the second case at a median of four
    milliseconds of lead in, and the first case exists only to stop truncation and P
    observability being the same event.
    """
    anchor_cls = P_CLASS if mode == 'pre_p' else QRS_CLASS
    lo, _ = _first_last(target, anchor_cls)
    if lo is None and mode == 'pre_p':
        lo, _ = _first_last(target, QRS_CLASS)
    if lo is None:
        return window, valid, target, landmarks, None

    window = window.copy()
    valid = valid.copy()
    target = target.copy()
    landmarks = landmarks.copy()
    lead_in = int(round(float(lead_in_ms) * float(fs_hz) / 1000.0))
    cut = max(0, int(lo) - lead_in)
    if cut <= 0:
        return window, valid, target, landmarks, 0.0

    window[:, :cut] = pad_value
    valid[:cut] = False
    target[:cut] = -100
    # a landmark that now sits outside the observed record is no longer asserted
    for i in range(landmarks.size):
        v = landmarks[i]
        if np.isfinite(v) and int(round(float(v))) < cut:
            landmarks[i] = np.nan
    return window, valid, target, landmarks, 1000.0 * cut / float(fs_hz)


def scale_t_amplitude(window, landmarks, alpha, fs_hz=500.0, blend_ms=8.0):
    """Scale the T wave's voltage toward its local baseline by alpha. Labels do not move.

    The diagnosis this serves: natural-context T-offset failures concentrate four to one in
    the units whose T is small in normalised units, and per-crop normalisation makes the same
    beat's T about three times larger when the crop is isolated than when its neighbours are
    present. The corpus therefore under-represents low-salience T waves relative to what the
    model meets, and this transform manufactures them deliberately.

    The segment from T onset to T offset is compressed toward a per-lead baseline drawn as the
    straight line between the segment's endpoint voltages, so both endpoints are continuous by
    construction, and the attenuation ramps in over blend_ms at each edge so no derivative step
    is introduced. Every lead is scaled by the same factor, which is what a globally smaller
    repolarisation deflection looks like. Timing labels are untouched on purpose: the transform
    changes how salient the T is, never where it is, so the supervision still states the true
    boundary.
    """
    on, off = landmarks[T_ONSET_IDX], landmarks[T_OFFSET_IDX]
    if not (np.isfinite(on) and np.isfinite(off)):
        return window, False
    n = window.shape[1]
    a, b = int(round(float(on))), int(round(float(off)))
    if not (0 <= a < b < n) or (b - a) < 6:
        return window, False
    w = window.copy()
    length = b - a + 1
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)[None, :]
    base = w[:, a:a + 1] * (1.0 - t) + w[:, b:b + 1] * t
    ramp = np.ones(length, dtype=np.float32)
    m = min(max(1, int(round(float(blend_ms) * float(fs_hz) / 1000.0))), length // 2)
    ramp[:m] = np.linspace(0.0, 1.0, m, dtype=np.float32)
    ramp[length - m:] = np.linspace(1.0, 0.0, m, dtype=np.float32)
    effective = 1.0 - (1.0 - float(alpha)) * ramp
    w[:, a:b + 1] = base + (w[:, a:b + 1] - base) * effective[None, :]
    return w, True


def stretch_repolarisation(window, valid, target, landmarks, alpha):
    """Resample the T peak-to-offset segment, holding the T onset and T peak fixed.

    Eighty three per cent of the excess T duration on the external records that fail sits after
    the T peak, and the R peak to T peak timing is the same to within four milliseconds across
    every stratum, so stretching the whole repolarisation segment would manufacture a
    discrepancy the data does not show. Only the descending limb moves. Material after the
    original T offset shifts right by the added duration and whatever leaves the crop is
    discarded. The factor is capped at what the crop can hold rather than the unit being
    skipped, which keeps the transform firing broadly instead of only on units with a long
    diastolic gap.
    """
    n = window.shape[1]
    p = landmarks[T_PEAK_IDX]
    o = landmarks[T_OFFSET_IDX]
    if not (np.isfinite(p) and np.isfinite(o)):
        _, o_t = _first_last(target, T_CLASS)
        if o_t is None:
            return window, valid, target, landmarks, None
        o = float(o_t)
        p = float(np.floor(0.5 * o)) if not np.isfinite(p) else p
    p, o = int(round(float(p))), int(round(float(o)))
    if not (0 <= p < o < n - 2):
        return window, valid, target, landmarks, None

    length = o - p
    if float(alpha) <= 0:
        return window, valid, target, landmarks, None
    new_len = int(round(float(alpha) * length))
    new_len = max(2, min(new_len, n - 2 - p))       # cap, do not skip
    delta = new_len - length
    if delta == 0:
        return window, valid, target, landmarks, 1.0

    # Index map for the resampled segment. Both endpoints are explicit, so the last T target
    # sample and the mapped T-offset landmark agree exactly at p + new_len.
    src = np.linspace(p + 1, o, new_len)
    lo_i = np.clip(np.floor(src).astype(int), 0, n - 1)
    hi_i = np.clip(lo_i + 1, 0, n - 1)
    frac = (src - lo_i).astype(window.dtype)

    w = window.copy()
    v = valid.copy()
    t = target.copy()

    w[:, p + 1:p + 1 + new_len] = (window[:, lo_i] * (1.0 - frac[None, :])
                                   + window[:, hi_i] * frac[None, :])
    near = np.clip(np.round(src).astype(int), 0, n - 1)
    v[p + 1:p + 1 + new_len] = valid[near]
    t[p + 1:p + 1 + new_len] = target[near]

    tail_from, tail_to = o + 1, p + 1 + new_len
    if delta > 0:
        # Make room for a longer T by shifting the subsequent context to the right.
        room = n - tail_to
        if room > 0 and tail_from < n:
            take = min(room, n - tail_from)
            w[:, tail_to:tail_to + take] = window[:, tail_from:tail_from + take]
            v[tail_to:tail_to + take] = valid[tail_from:tail_from + take]
            t[tail_to:tail_to + take] = target[tail_from:tail_from + take]
            if take < room:
                w[:, tail_to + take:] = window[:, -1:]
                v[tail_to + take:] = False
                t[tail_to + take:] = -100
    else:
        # A shorter descending limb leaves a real isoelectric interval before the unchanged
        # suffix. Fill the samples formerly occupied by T with the first post-T baseline and
        # supervise that constructed interval as background.
        gap_lo, gap_hi = tail_to, o + 1
        baseline_at = min(o + 1, n - 1)
        w[:, gap_lo:gap_hi] = window[:, baseline_at:baseline_at + 1]
        t[gap_lo:gap_hi] = np.where(v[gap_lo:gap_hi], BACKGROUND, -100)

    marks = landmarks.copy()
    for i in range(marks.size):
        x = marks[i]
        if not np.isfinite(x):
            continue
        x = float(x)
        if x <= p:
            continue
        if x <= o:
            marks[i] = p + (x - p) * (new_len / float(length))
        elif delta > 0:
            marks[i] = x + delta
        if marks[i] > n - 1:
            marks[i] = np.nan
    return w, v, t, marks, new_len / float(length)



def _draw_context(cfg, rng):
    """Pick one of the three observation geometries. Returns the mode and the lead in."""
    c = cfg.get('context') or {}
    if not c:
        return None, 0.0
    modes = ['none', 'pre_p', 'qrs_edge']
    w = np.array([float(c.get('none_p', 0.0) or 0.0),
                  float(c.get('pre_p_p', 0.0) or 0.0),
                  float(c.get('qrs_edge_p', 0.0) or 0.0)], dtype=float)
    if w.sum() <= 0:
        return None, 0.0
    mode = modes[int(rng.choice(3, p=w / w.sum()))]
    if mode == 'none':
        return 'none', 0.0
    key = 'pre_p_lead_in_ms' if mode == 'pre_p' else 'qrs_edge_lead_in_ms'
    span = c.get(key, [40.0, 150.0] if mode == 'pre_p' else [0.0, 8.0])
    if isinstance(span, (list, tuple)) and len(span) == 2:
        return mode, float(rng.uniform(float(span[0]), float(span[1])))
    return mode, float(span)


def apply_structural(window, valid, target, landmarks, win_lo, win_hi, cfg, rng, fs_hz=500.0):
    """Draw and apply the structural transforms for one item.

    cfg keys, all defaulting to off:

        isolate_beat_p           probability of removing the neighbouring beats
        isolate_tail_ms          flat post-T tail left standing when the beat is isolated
        repolarisation_stretch_p probability of resampling the descending limb of the T wave
        repolarisation_alpha     [lo, hi] factor applied to the T peak to T offset segment
        repolarisation_target_ms optional [lo, hi] uniform or [lo, mode, hi] triangular target
                                 duration, supporting both shortening and lengthening
        repolarisation_alpha_clip optional [lo, hi] guard for degenerate landmark spans
        repolarisation_min_source_ms optional minimum usable peak-to-offset source span
        t_amplitude_p            probability of scaling the T wave toward its baseline
        t_amplitude_range        [lo, hi] factor drawn for that scaling, labels untouched
        context                  a three way draw over the observation geometry, holding
                                 none_p, pre_p_p, qrs_edge_p and the two lead in ranges
        p_mask_p                 probability of removing the P from an intact record
        t_offset_jitter_ms       half-width of a uniform jitter on the T offset label
        time_warp_pct            half-width of a uniform time warp, off by default

        left_edge_p, p_absent_p  the earlier single-purpose transforms, still honoured when
                                 the newer keys are absent so old configs reproduce exactly

    The order is morphology, then observation geometry, then the P. That order matters. A crop
    truncated at the QRS onset has already lost its P, so the P mask finds nothing and does not
    fire, while a crop truncated ahead of the P still has one and the mask may fire on it. That
    is what gives all four combinations of truncation and P observability rather than two.

    Returns the four arrays and a dict recording what was applied, which train.py writes into
    the run log so a reproduction can be audited rather than inferred.
    """
    applied = {}
    if not cfg:
        return window, valid, target, landmarks, applied

    if rng.random() < float(cfg.get('isolate_beat_p', 0.0) or 0.0):
        # The tail length is drawn rather than fixed. A single value teaches one record duration
        # and the network can learn the position instead of the fact, so the draw spans from a
        # record that ends soon after the T to one that fills the crop. isolate_beat clips a draw
        # the crop cannot hold, which is the case of a record continuing past the crop edge.
        span = cfg.get('isolate_tail_ms', 300.0)
        tail_ms = (float(rng.uniform(float(span[0]), float(span[1])))
                   if isinstance(span, (list, tuple)) and len(span) == 2 else float(span or 0.0))
        window, valid, target = isolate_beat(
            window, valid, win_lo, win_hi, keep_tail_ms=tail_ms, fs_hz=fs_hz, target=target,
            supervise_tail=bool(cfg.get('isolate_supervise_tail', True)))
        applied['isolate_beat'] = True
        applied['isolate_tail_ms'] = round(tail_ms, 1)

    if rng.random() < float(cfg.get('repolarisation_stretch_p', 0.0) or 0.0):
        target_span = cfg.get('repolarisation_target_ms')
        p_before = landmarks[T_PEAK_IDX]
        o_before = landmarks[T_OFFSET_IDX]
        current = (float(o_before) - float(p_before)
                   if np.isfinite(p_before) and np.isfinite(o_before) else np.nan)
        if isinstance(target_span, (list, tuple)) and len(target_span) in (2, 3):
            if len(target_span) == 3:
                wanted = float(rng.triangular(float(target_span[0]), float(target_span[1]),
                                              float(target_span[2])))
            else:
                wanted = float(rng.uniform(float(target_span[0]), float(target_span[1])))
            alpha = ((wanted * float(fs_hz) / 1000.0) / current
                     if np.isfinite(current) and current > 0 else 1.0)
            alpha_clip = cfg.get('repolarisation_alpha_clip')
            if isinstance(alpha_clip, (list, tuple)) and len(alpha_clip) == 2:
                alpha = float(np.clip(alpha, float(alpha_clip[0]), float(alpha_clip[1])))
            minimum_source_ms = float(cfg.get('repolarisation_min_source_ms', 0.0) or 0.0)
            current_ms = (current * 1000.0 / float(fs_hz)
                          if np.isfinite(current) else float('nan'))
            if minimum_source_ms and (not np.isfinite(current_ms)
                                      or current_ms < minimum_source_ms):
                alpha = None
        else:
            span = cfg.get('repolarisation_alpha', [1.0, 2.3])
            lo, hi = ((float(span[0]), float(span[1]))
                      if isinstance(span, (list, tuple)) else (1.0, float(span)))
            alpha = float(rng.uniform(lo, hi))
        if alpha is None:
            got = None
        else:
            window, valid, target, landmarks, got = stretch_repolarisation(
                window, valid, target, landmarks, alpha)
        if got is not None:
            applied['repolarisation_alpha'] = round(float(got), 3)
            if np.isfinite(current) and current > 0:
                applied['repolarisation_target_ms'] = round(
                    float(got) * current * 1000.0 / float(fs_hz), 1)

    # Morphology, like the stretch above, so it acts on the post-stretch landmarks and before
    # the observation-geometry draws. The draw is consumed even when the transform cannot fire,
    # keeping the per-epoch random stream aligned across units with and without a usable T.
    if rng.random() < float(cfg.get('t_amplitude_p', 0.0) or 0.0):
        span = cfg.get('t_amplitude_range', [0.4, 1.0])
        lo, hi = ((float(span[0]), float(span[1]))
                  if isinstance(span, (list, tuple)) and len(span) == 2 else (float(span), 1.0))
        alpha = float(rng.uniform(lo, hi))
        window, did = scale_t_amplitude(window, landmarks, alpha, fs_hz=fs_hz,
                                        blend_ms=float(cfg.get('t_amplitude_blend_ms', 8.0)))
        if did:
            applied['t_amplitude'] = round(alpha, 3)

    mode, lead_in = _draw_context(cfg, rng)
    if mode is None:
        # the earlier transform, kept so configs written before this change reproduce
        if rng.random() < float(cfg.get('left_edge_p', 0.0) or 0.0):
            span = cfg.get('left_edge_lead_in_ms', 4.0)
            lead_in = (float(rng.uniform(float(span[0]), float(span[1])))
                       if isinstance(span, (list, tuple)) and len(span) == 2 else float(span))
            window, valid = left_edge(window, valid, win_lo, lead_in_ms=lead_in, fs_hz=fs_hz)
            applied['left_edge_ms'] = round(lead_in, 2)
    elif mode != 'none':
        window, valid, target, landmarks, cut_ms = truncate_context(
            window, valid, target, landmarks, mode, lead_in, fs_hz=fs_hz)
        applied['context'] = mode
        applied['context_lead_in_ms'] = round(float(lead_in), 2)
    else:
        applied['context'] = 'none'

    if 'p_mask_p' in cfg:
        if rng.random() < float(cfg.get('p_mask_p', 0.0) or 0.0):
            window, valid, target, landmarks, did = mask_p_wave(
                window, valid, target, landmarks, rng, fs_hz=fs_hz,
                matched_residual=bool(cfg.get('p_mask_matched_residual', True)))
            if did:
                applied['p_mask'] = True
    elif rng.random() < float(cfg.get('p_absent_p', 0.0) or 0.0):
        window, valid, target, landmarks, did = remove_p_wave(window, valid, target, landmarks,
                                                              fs_hz=fs_hz)
        if did:
            applied['p_absent'] = True

    jitter = float(cfg.get('t_offset_jitter_ms', 0.0) or 0.0)
    if jitter > 0:
        shift_ms = float(rng.uniform(-jitter, jitter))
        shift = int(round(shift_ms * fs_hz / 1000.0))
        target, landmarks = jitter_t_offset(target, landmarks, shift)
        applied['t_offset_jitter_ms'] = round(shift_ms, 2)

    warp = float(cfg.get('time_warp_pct', 0.0) or 0.0)
    if warp > 0:
        factor = 1.0 + float(rng.uniform(-warp, warp)) / 100.0
        window, target, landmarks, valid = time_warp(window, target, landmarks, valid, factor)
        applied['time_warp'] = round(factor, 4)

    applied['truncated'] = bool(applied.get('context') not in (None, 'none')
                                or 'left_edge_ms' in applied)
    return window, valid, target, landmarks, applied
