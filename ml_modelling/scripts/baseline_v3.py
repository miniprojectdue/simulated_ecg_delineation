"""
baseline_v3.py  -  a faithful Python port of delineate_ecg_v3.m.

The MATLAB original is written for a 1000 Hz beat, where one sample is one millisecond, and
every constant in it is therefore simultaneously a sample count and a duration. This port keeps
the constants in milliseconds and converts them to samples at the actual sampling rate, so at
1000 Hz it reduces exactly to the original and at 500 Hz it preserves the algorithm's behaviour
in time rather than in samples.

Nothing else is changed. The thresholds, the search windows, the refinement, the fallbacks and
the ordering of the six stages are as written in the source.

The delineator produces five landmarks, the QRS onset and offset and the T onset, peak and
offset. It has no P-wave stage and does not name the Q, R and S deflections, which is the
documented scope of the tool rather than a limitation of this port.
"""
import numpy as np

LANDMARKS_OUT = ['qrs_onset', 'qrs_offset', 't_onset', 't_peak', 't_offset']


def _movmean(x, k):
    """MATLAB movmean, centred, with shrinking windows at the two edges."""
    n = x.size
    half_lo = (k - 1) // 2
    half_hi = k // 2
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    lo = np.maximum(np.arange(n) - half_lo, 0)
    hi = np.minimum(np.arange(n) + half_hi + 1, n)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _mad1(x):
    """MATLAB mad(x, 1), the median absolute deviation about the median."""
    return float(np.median(np.abs(x - np.median(x))))


def _gw(data, i1, width):
    """MATLAB get_window(data, idx, width) with a 1-based idx, the mean over idx +- width."""
    lo = max(i1 - width - 1, 0)
    hi = min(i1 + width, data.size)
    if hi <= lo:
        return 0.0
    return float(np.mean(data[lo:hi]))


def delineate_v3(v, fs_hz=500.0, activation_time_ms=67.0, width_ms=3.0, duration_ms=None):
    """Return the five landmarks as 0-based sample indices, plus the diagnostics the source keeps.

    v is one lead of one beat, already normalised the way buildECGTable normalises it.
    """
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    per_ms = float(fs_hz) / 1000.0

    def S(ms, minimum=1):
        """A duration in milliseconds as a whole number of samples, at least `minimum`."""
        return max(int(minimum), int(round(float(ms) * per_ms)))

    width = S(width_ms)
    duration = n if duration_ms is None else min(n, S(duration_ms))
    if duration < S(200):
        duration = n

    dv = np.gradient(v)
    ddv = np.gradient(dv)

    v_ex = np.concatenate((np.repeat(v[0], width), v))
    dv_ex = np.gradient(v_ex)
    ddv_ex = np.gradient(dv_ex)

    # ---- baseline noise, early plus late ------------------------------------------------
    base_a = np.arange(0, min(S(80), n))
    lo_b = max(n - S(150), 1) - 1
    hi_b = max(n - S(20), 1)
    base_b = np.arange(lo_b, max(hi_b, lo_b + 1))
    base_idx = np.unique(np.concatenate((base_a, base_b)))
    base_idx = base_idx[(base_idx >= 0) & (base_idx < n)]

    eps = np.finfo(np.float64).eps
    sigma_dv = 1.4826 * _mad1(dv[base_idx])
    sigma_ddv = 1.4826 * _mad1(ddv[base_idx])
    sigma_dv = max(sigma_dv, eps, 0.01 * float(np.max(np.abs(dv))) + eps)
    sigma_ddv = max(sigma_ddv, eps, 0.01 * float(np.max(np.abs(ddv))) + eps)

    k_qrs_start, k_qrs_end, k_qrs_end_dv, k_t_start, k_t_end = 0.5, 1.0, 1.0, 1.0, 0.6
    qrs_start_tol = k_qrs_start * sigma_ddv
    qrs_end_tol_ddv = k_qrs_end * sigma_ddv
    t_start_tol = k_t_start * sigma_dv
    t_end_tol = k_t_end * sigma_dv
    _qrs_end_tol_dv = k_qrs_end_dv * sigma_dv  # computed by the source, unused downstream

    diag = {'sigma_dV': sigma_dv, 'sigma_ddV': sigma_ddv, 'fallbacks': []}

    # ---- 1) QRS onset -------------------------------------------------------------------
    qrs_start = None
    search_start = max(width + 1, S(15))
    search_end = min(n, S(200))
    qrs_start_tol_dv = 0.2 * sigma_dv

    for i1 in range(search_start, search_end + 1):
        if _gw(np.abs(ddv_ex), i1, width) > qrs_start_tol and _gw(np.abs(dv_ex), i1, width) > qrs_start_tol_dv:
            qrs_start = i1 - width                      # 1-based index into v
            ref_back, ref_sustain, ref_frac = S(30), S(6), 0.15
            j0 = max(1, qrs_start - ref_back)
            j1 = min(n, qrs_start + S(20))
            local_peak = float(np.max(np.abs(dv[j0 - 1:j1]))) if j1 >= j0 else 0.0
            if local_peak > 0:
                thr = ref_frac * local_peak
                for jj in range(j0, qrs_start - ref_sustain + 1):
                    if np.all(np.abs(dv[jj - 1:jj + ref_sustain]) > thr):
                        qrs_start = jj
                        break
            break

    if qrs_start is None:
        diag['fallbacks'].append('qrs_onset')
        win = np.arange(search_start, search_end + 1)
        imax = int(win[int(np.argmax(np.abs(ddv[win - 1])))])
        qrs_start = max(1, imax - S(20))

    # ---- 2) QRS offset ------------------------------------------------------------------
    qrs_end = None
    end_start = max(1, qrs_start + S(20))
    end_stop = min(n, qrs_start + S(160))
    hold = S(8)
    for idx in range(end_start, end_stop - hold + 1):
        if float(np.max(np.abs(ddv[idx - 1:idx + hold]))) < qrs_end_tol_ddv:
            qrs_end = idx
            break
    if qrs_end is None:
        diag['fallbacks'].append('qrs_offset')
        win = np.arange(end_start, end_stop + 1)
        qrs_end = int(win[int(np.argmin(np.abs(ddv[win - 1])))])
    qrs_end = max(1, min(n, int(round(qrs_end))))

    # ---- 4) T peak ----------------------------------------------------------------------
    tp_start = qrs_end + S(40)
    tp_end = duration
    if tp_end <= tp_start + 5:
        tp_start = min(n, qrs_end + S(20))
        tp_end = n
    tp_start = max(1, min(n, tp_start))
    tp_end = max(tp_start, min(n, tp_end))
    segment = v[tp_start - 1:tp_end]
    rel = int(np.argmax(np.abs(segment)))
    t_magnitude = float(np.abs(segment[rel]))
    t_peak = tp_start + rel
    t_sign = np.sign(v[t_peak - 1]) or 1.0
    t_magnitude_true = float(t_sign * t_magnitude)

    # ---- 5) T offset --------------------------------------------------------------------
    t_end = None
    sustain, max_delay, min_delay = S(35), S(260), S(30)
    tail_start = min(n, t_peak + S(120))
    tail_end = min(n, t_peak + max_delay)
    if tail_end > tail_start + 10:
        base_level = float(np.median(v[tail_start - 1:tail_end]))
    else:
        b1, b2 = min(n, qrs_end + S(20)), min(n, qrs_end + S(60))
        if b2 <= b1:
            b1, b2 = min(n, qrs_end + S(5)), min(n, qrs_end + S(30))
        base_level = float(np.median(v[b1 - 1:max(b2, b1)]))

    a_peak = max(abs(float(v[t_peak - 1]) - base_level), 1e-12)
    amp_tol_end = max(0.15 * a_peak, 0.01 * float(np.max(np.abs(v))))

    i_start = min(n, t_peak + min_delay)
    i_stop = min(n - sustain - 1, t_peak + max_delay)
    v_s = _movmean(v, 7)
    dv_s = np.gradient(v_s)

    if i_stop > i_start:
        for i1 in range(i_start, i_stop + 1):
            flat = float(np.max(np.abs(dv_s[i1 - 1:i1 + sustain]))) < t_end_tol
            small = float(np.max(np.abs(v[i1 - 1:i1 + sustain] - base_level))) < amp_tol_end
            if flat and small:
                t_end = i1
                break
        if t_end is None:
            for i1 in range(i_start, i_stop + 1):
                if float(np.max(np.abs(dv_s[i1 - 1:i1 + sustain]))) < t_end_tol:
                    t_end = i1
                    diag['fallbacks'].append('t_offset_slope_only')
                    break
    if t_end is None:
        diag['fallbacks'].append('t_offset_fixed')
        t_end = min(n, t_peak + S(140))
    t_end = max(1, min(n, int(round(t_end))))

    # pinned to the end of the beat, retry with a longer smoother
    if t_end >= n - 2:
        dv_s2 = np.gradient(_movmean(v, 11))
        sustain2 = S(25)
        i_start2 = min(n, t_peak + S(30))
        i_stop2 = min(n - sustain2 - 1, t_peak + S(260))
        for i1 in range(i_start2, i_stop2 + 1):
            if float(np.max(np.abs(dv_s2[i1 - 1:i1 + sustain2]))) < 1.5 * t_end_tol:
                t_end = max(1, min(n, i1))
                diag['fallbacks'].append('t_offset_unpinned')
                break

    # secondary bump after the current offset
    post_start = min(n, t_end + 5)
    post_end = min(n, t_peak + max_delay)
    if post_end > post_start + 10:
        post = v[post_start - 1:post_end]
        med = float(np.median(post))
        if float(np.max(np.abs(post - med))) > 0.12 * a_peak:
            bump = post_start + int(np.argmax(np.abs(post - med)))
            sustain3 = S(20)
            s_from = min(n, bump + 10)
            s_to = min(n - sustain3 - 1, bump + S(180))
            for i1 in range(s_from, s_to + 1):
                if float(np.max(np.abs(dv_s[i1 - 1:i1 + sustain3]))) < 1.5 * t_end_tol:
                    t_end = i1
                    diag['fallbacks'].append('t_offset_bump')
                    break

    # ---- 6) T onset ---------------------------------------------------------------------
    t_start = None
    st1, st2 = min(n, qrs_end + S(15)), min(n, qrs_end + S(60))
    if st2 <= st1:
        st1, st2 = min(n, qrs_end + S(5)), min(n, qrs_end + S(30))
    st_level = float(np.median(v[st1 - 1:max(st2, st1)]))
    a_peak_st = max(abs(float(v[t_peak - 1]) - st_level), 1e-12)
    amp_tol_start = 0.16 * a_peak_st
    sustain_st = S(12)
    t_sign_st = np.sign(float(v[t_peak - 1]) - st_level) or 1.0

    i_start = min(n, qrs_end + S(10))
    i_stop = min(n - sustain_st - 1, t_peak - S(10))
    if i_stop > i_start:
        for i1 in range(i_start, i_stop + 1):
            seg = v[i1 - 1:i1 + sustain_st]
            if np.all(np.abs(seg - st_level) > amp_tol_start) \
                    and float(np.mean(t_sign_st * dv[i1 - 1:i1 + sustain_st])) > 0:
                t_start = i1
                break
    if t_start is None:
        diag['fallbacks'].append('t_onset')
        t_start = max(1, t_peak - S(80))
    t_start = max(1, min(n, int(round(t_start))))

    qrs_end_flag = abs((qrs_end - 1) / per_ms - activation_time_ms) >= 25

    # 1-based internally, 0-based out, to match the loader and the reference table
    return {
        'qrs_onset': qrs_start - 1,
        'qrs_offset': qrs_end - 1,
        't_onset': t_start - 1,
        't_peak': t_peak - 1,
        't_offset': t_end - 1,
        'qrs_end_flag': bool(qrs_end_flag),
        't_magnitude_true': t_magnitude_true,
        'diag': diag,
    }


def normalise_record(x12):
    """buildECGTable's scaling, every lead divided by the largest absolute value over all leads."""
    x12 = np.asarray(x12, dtype=np.float64)
    m = float(np.max(np.abs(x12)))
    if m == 0.0 or not np.isfinite(m):
        m = 1.0
    return x12 / m
