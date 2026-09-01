#!/usr/bin/env python3
"""
postprocess.py  -  turn per-sample class probabilities into the eleven fiducials.

The network produces a label sequence. The dissertation's Prediction Post-processing section
requires fiducials, and the two are related by a fixed set of rules.

Boundaries come from the label sequence. The onset of a wave is the first sample of its region
and the offset is the last, after a median filter removes single sample speckle and after
regions shorter than a minimum duration are discarded.

Peaks come from the voltage trace, never from the probabilities.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_INDEX, LANDMARKS  # noqa: E402

WAVE_CLASSES = {'p': CLASS_INDEX['p'], 'qrs': CLASS_INDEX['qrs'], 't': CLASS_INDEX['t']}


def median_filter_labels(labels, kernel=9):
    """A mode filter over a short window. Removes speckle without moving a real boundary far."""
    if kernel is None or kernel <= 1:
        return labels
    if kernel % 2 == 0:
        kernel += 1
    half = kernel // 2
    padded = np.pad(labels, (half, half), mode='edge')
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    out = np.empty_like(labels)
    for i in range(labels.shape[0]):
        counts = np.bincount(windows[i])
        out[i] = int(np.argmax(counts))
    return out


def contiguous_regions(mask):
    """Return the inclusive start and end index of every run of True in a boolean vector."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, len(edges), 2)]


def pick_region(labels, class_id, min_samples=8, keep_largest=True, anchor=None, anchor_mask=None,
                strict_anchor=False):
    """Choose the one region of a class in this crop, since a crop holds a single beat.

    anchor_mask, when given, is the indexed beat's window. A region that touches it belongs to
    the beat being delineated and a region that does not belongs to a neighbour, so the mask
    decides identity while the region itself decides extent. That separation is what lets the
    search run wider than the window without the widening changing which beat is answered.

    strict_anchor changes what happens when no region touches the anchor_mask. The default
    (False) preserves the historical behaviour of falling back to the largest region anywhere in
    the crop. With strict_anchor True the wave is instead reported absent, so a beat whose class
    does not appear inside its own window cannot silently be answered with a neighbour's region.
    This matters under a label-free window, where borrowing a neighbour's wave would leak an
    off-beat reading into the score.
    """
    regions = [r for r in contiguous_regions(labels == class_id) if (r[1] - r[0] + 1) >= min_samples]
    if not regions:
        return None
    if anchor_mask is not None:
        touching = [r for r in regions if anchor_mask[r[0]:r[1] + 1].any()]
        if touching:
            regions = touching
        elif strict_anchor:
            return None
    if not keep_largest:
        return regions[0]
    if anchor is not None:
        inside = [r for r in regions if r[0] <= anchor <= r[1]]
        if inside:
            return max(inside, key=lambda r: r[1] - r[0])
    return max(regions, key=lambda r: r[1] - r[0])


def local_extrema(x, kind='max'):
    """Indices of strict local extrema, with plateaux reported at their centre."""
    if x.size < 3:
        return np.array([], dtype=int)
    if kind == 'min':
        x = -x
    out = []
    i = 1
    n = x.size
    while i < n - 1:
        if x[i] > x[i - 1]:
            j = i
            while j + 1 < n and x[j + 1] == x[i]:
                j += 1
            if j + 1 < n and x[j + 1] < x[i]:
                out.append((i + j) // 2)
            i = j + 1
        else:
            i += 1
    return np.array(out, dtype=int)


def extremum_peak(trace, lo, hi, baseline):
    """The sample of largest absolute deviation from the baseline inside an inclusive span."""
    if hi < lo:
        return None
    segment = trace[lo:hi + 1] - baseline
    return int(lo + int(np.argmax(np.abs(segment))))


def name_qrs_deflections(trace, onset, offset, amplitude_frac=0.05, amplitude_floor=0.02):
    """Apply the polarity naming rule inside the QRS region.

    Returns a dict with q_peak, r_peak and s_peak, any of which may be None, plus the pattern
    string in the usual case notation where a capital letter marks the dominant deflection.
    """
    result = {'q_peak': None, 'r_peak': None, 's_peak': None, 'pattern': ''}
    if offset <= onset:
        return result
    baseline = float(trace[onset])
    dev = trace[onset:offset + 1].astype(np.float64) - baseline
    if dev.size < 3:
        return result
    threshold = max(float(amplitude_floor), float(amplitude_frac) * float(np.max(np.abs(dev))))

    maxima = [i for i in local_extrema(dev, 'max') if dev[i] > threshold]
    # The global positive extremum counts as a deflection even when it sits at the edge of the
    # region, where a strict local maximum test cannot see it.
    global_max = int(np.argmax(dev))
    if dev[global_max] > threshold and global_max not in maxima:
        maxima.append(global_max)
    maxima = sorted(set(int(i) for i in maxima))

    if not maxima:
        # No positive deflection anywhere, so this is a QS complex. It carries a Q and nothing else.
        nadir = int(np.argmin(dev))
        if dev[nadir] < -threshold:
            result['q_peak'] = onset + nadir
            result['pattern'] = 'QS'
        return result

    r_rel = maxima[0]
    result['r_peak'] = onset + r_rel

    before = dev[:r_rel]
    if before.size:
        q_rel = int(np.argmin(before))
        if before[q_rel] < -threshold:
            result['q_peak'] = onset + q_rel

    after = dev[r_rel + 1:]
    if after.size:
        s_rel = int(np.argmin(after)) + r_rel + 1
        if dev[s_rel] < -threshold:
            result['s_peak'] = onset + s_rel

    result['pattern'] = _pattern_string(dev, result, onset)
    return result


def _pattern_string(dev, picked, onset):
    """Case notation, upper case for the deflection carrying at least half the largest amplitude."""
    entries = []
    for key, letter in (('q_peak', 'q'), ('r_peak', 'r'), ('s_peak', 's')):
        idx = picked.get(key)
        if idx is None:
            continue
        entries.append((idx, letter, abs(float(dev[idx - onset]))))
    if not entries:
        return ''
    entries.sort()
    biggest = max(e[2] for e in entries)
    return ''.join(letter.upper() if amp >= 0.5 * biggest else letter for _, letter, amp in entries)


def fiducials_from_labels(labels, trace, cfg=None, supervised=None, readable=None):
    """The full label sequence to fiducials mapping for one crop.

    labels     integer class per sample, already argmaxed
    trace      the voltage of the lead being delineated, same length
    supervised optional boolean mask naming the indexed beat's window
    readable   optional boolean mask naming how far a region of that beat may be followed.
               Defaults to supervised, which is the behaviour every earlier result was produced
               under.

    The two masks answer different questions and used to be one object. The window says which
    beat is being asked about. It was never a statement about how long that beat's waves are
    allowed to be, and using it as one silently truncates a wave that ends past the window edge
    and then reports the truncation as the model's error. Passing a wider readable mask lets a
    wave be followed to its own end while the window still decides which wave is this beat's.
    """
    cfg = cfg or {}
    min_samples = int(cfg.get('min_region_samples', 8))
    # A per-class floor. Defaults to the shared value, so behaviour is unchanged unless set.
    min_p = int(cfg.get('min_region_samples_p', min_samples))
    min_qrs = int(cfg.get('min_region_samples_qrs', min_samples))
    min_t = int(cfg.get('min_region_samples_t', min_samples))
    kernel = int(cfg.get('smoothing_kernel', 9))
    keep_largest = bool(cfg.get('keep_largest_region', True))
    # When set, a wave whose class does not touch the beat window is reported absent rather than
    # answered with the largest region anywhere in the crop. Off by default so existing results
    # reproduce; intended for the label-free external evaluation.
    strict = bool(cfg.get('region_anchor_strict', False))
    frac = float(cfg.get('qrs_amplitude_frac', 0.05))
    floor = float(cfg.get('qrs_amplitude_floor_mv', 0.02))

    labels = np.asarray(labels).astype(np.int64).copy()
    window = None if supervised is None else np.asarray(supervised).astype(bool)
    scope = window if readable is None else np.asarray(readable).astype(bool)
    if scope is not None:
        labels[~scope] = CLASS_INDEX['background']
    labels = median_filter_labels(labels, kernel)

    out = {name: None for name in LANDMARKS}
    out['qrs_pattern'] = ''
    present = {'p_present': 0, 'qrs_present': 0, 't_present': 0,
               'q_present': 0, 'r_present': 0, 's_present': 0}

    qrs = pick_region(labels, WAVE_CLASSES['qrs'], min_qrs, keep_largest, anchor_mask=window,
                      strict_anchor=strict)
    if qrs is not None:
        out['qrs_onset'], out['qrs_offset'] = qrs
        present['qrs_present'] = 1
        named = name_qrs_deflections(trace, qrs[0], qrs[1], frac, floor)
        out['q_peak'] = named['q_peak']
        out['r_peak'] = named['r_peak']
        out['s_peak'] = named['s_peak']
        out['qrs_pattern'] = named['pattern']
        present['q_present'] = int(named['q_peak'] is not None)
        present['r_present'] = int(named['r_peak'] is not None)
        present['s_present'] = int(named['s_peak'] is not None)

    p = pick_region(labels, WAVE_CLASSES['p'], min_p, keep_largest, anchor_mask=window,
                    strict_anchor=strict)
    if p is not None:
        # The P wave belongs to the QRS that follows it, so a region after the QRS is not this beat's P.
        if qrs is None or p[0] < qrs[0]:
            out['p_onset'], out['p_offset'] = p
            out['p_peak'] = extremum_peak(trace, p[0], p[1], float(trace[p[0]]))
            present['p_present'] = 1

    t = pick_region(labels, WAVE_CLASSES['t'], min_t, keep_largest, anchor_mask=window,
                    strict_anchor=strict)
    if t is not None:
        if qrs is None or t[1] > qrs[1]:
            out['t_onset'], out['t_offset'] = t
            out['t_peak'] = extremum_peak(trace, t[0], t[1], float(trace[t[0]]))
            present['t_present'] = 1

    out.update(present)
    return out


def _suppressed_class_ids(cfg):
    """Class ids the caller has asked to remove from the decision entirely."""
    names = (cfg or {}).get('suppress_classes', '')
    if not names:
        return []
    if isinstance(names, str):
        names = [n.strip() for n in names.split(',') if n.strip()]
    return [CLASS_INDEX[n] for n in names if n in CLASS_INDEX]


def batch_fiducials(logits, traces, cfg=None, supervised=None, readable=None):
    """Run the mapping over a batch of logits. Accepts torch tensors or numpy arrays.

    When cfg carries suppress_classes, those classes are driven to minus infinity before the
    argmax, so their samples fall to whichever class the network ranked next. This is how a
    corpus documented to lack a wave entirely is scored without retraining.
    """
    drop = _suppressed_class_ids(cfg)
    try:
        import torch
        # The dual-head model returns (logits, p_logit). Unwrapping here rather than at each
        # call site means a caller that forgets cannot reach the numpy branch with a tuple.
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, torch.Tensor):
            if drop:
                logits = logits.clone()
                for c in drop:
                    logits[:, c, :] = float('-inf')
            labels = logits.argmax(dim=1).detach().cpu().numpy()
        else:
            arr = np.asarray(logits, dtype=np.float64).copy()
            for c in drop:
                arr[:, c, :] = -np.inf
            labels = arr.argmax(axis=1)
        if isinstance(traces, torch.Tensor):
            traces = traces.detach().cpu().numpy()
        if supervised is not None and isinstance(supervised, torch.Tensor):
            supervised = supervised.detach().cpu().numpy()
        if readable is not None and isinstance(readable, torch.Tensor):
            readable = readable.detach().cpu().numpy()
    except ImportError:
        arr = np.asarray(logits, dtype=np.float64).copy()
        for c in drop:
            arr[:, c, :] = -np.inf
        labels = arr.argmax(axis=1)
        traces = np.asarray(traces)

    results = []
    for i in range(labels.shape[0]):
        mask = None if supervised is None else supervised[i]
        wide = None if readable is None else readable[i]
        results.append(fiducials_from_labels(labels[i], np.asarray(traces[i]), cfg, mask, wide))
    return results


def enforce_order(fiducials):
    """Report whether the landmarks that were found respect the canonical time order.

    The order applies only to the landmarks the beat actually carries, so an absent wave never
    counts as a violation. This mirrors the ordering rule stated in the Methods chapter.
    """
    order = [f for f in LANDMARKS if fiducials.get(f) is not None]
    values = [fiducials[f] for f in order]
    violations = [(order[i], order[i + 1]) for i in range(len(values) - 1) if values[i] > values[i + 1]]
    return {'order_ok': int(not violations), 'violations': violations}
