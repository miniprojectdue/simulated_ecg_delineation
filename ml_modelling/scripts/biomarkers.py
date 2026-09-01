import numpy as np

BIOMARKERS = {
    'qrs_duration': ('qrs_onset', 'qrs_offset'),
    'qt_interval': ('qrs_onset', 't_offset'),
    't_duration': ('t_onset', 't_offset'),
    't_peak_to_end': ('t_peak', 't_offset'),
}
BIOMARKER_NAMES = ['qrs_duration', 'qt_interval', 't_duration', 't_peak_to_end']


def compute_biomarkers(fiducials, fs_hz=500.0, names=None):
    """Milliseconds per biomarker, NaN where a required fiducial is missing."""
    step_ms = 1000.0 / float(fs_hz)
    out = {}
    for name in (names or BIOMARKER_NAMES):
        start_key, end_key = BIOMARKERS[name]
        start, end = fiducials.get(start_key), fiducials.get(end_key)
        if start is None or end is None or (isinstance(start, float) and np.isnan(start)) \
                or (isinstance(end, float) and np.isnan(end)):
            out[name] = float('nan')
        else:
            out[name] = (float(end) - float(start)) * step_ms
    return out


def biomarker_errors(predicted, truth, fs_hz=500.0, names=None):
    """Signed error per biomarker in milliseconds, prediction minus truth."""
    p = compute_biomarkers(predicted, fs_hz, names)
    t = compute_biomarkers(truth, fs_hz, names)
    return {name: p[name] - t[name] for name in p}


def summarise_biomarker_errors(rows, names=None):
    """Bias, MAE, standard deviation and coverage over a list of per-unit error dicts."""
    names = names or BIOMARKER_NAMES
    summary = {}
    for name in names:
        values = np.array([r.get(name, float('nan')) for r in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name] = {
            'n_comparable': int(finite.size),
            'n_total': int(values.size),
            'coverage': float(finite.size) / float(values.size) if values.size else 0.0,
            'bias_ms': float(np.mean(finite)) if finite.size else float('nan'),
            'mae_ms': float(np.mean(np.abs(finite))) if finite.size else float('nan'),
            'sd_ms': float(np.std(finite, ddof=1)) if finite.size > 1 else float('nan'),
            'p95_abs_ms': float(np.percentile(np.abs(finite), 95)) if finite.size else float('nan'),
        }
    return summary
