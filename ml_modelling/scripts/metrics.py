"""
metrics.py  -  every number the evaluation protocol in the dissertation asks for.

Segmentation quality. Per-class F1 and intersection over union, computed from a confusion
matrix accumulated over supervised samples only, plus the mean intersection over union across
the three wave classes.

Fiducial quality. For each of the eleven landmarks, a prediction is a true positive when the
truth carries that landmark and the prediction falls within the tolerance, a false positive
when the prediction exists and the truth does not or the prediction misses by more than the
tolerance, and a false negative when the truth carries the landmark and the prediction does
not. Sensitivity and precision follow. Timing error is reported over the matched pairs as a
signed mean, a mean absolute error, a standard deviation and the percentage inside each
tolerance.

Every timing number is in milliseconds
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CLASS_NAMES, IGNORE_INDEX, LANDMARKS, N_CLASSES  # noqa: E402


class ConfusionAccumulator(object):
    """Accumulate a class by class confusion matrix over batches without holding the data."""

    def __init__(self, n_classes=N_CLASSES, ignore_index=IGNORE_INDEX):
        self.n_classes = int(n_classes)
        self.ignore_index = int(ignore_index)
        self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, predicted, target):
        predicted = np.asarray(predicted).reshape(-1)
        target = np.asarray(target).reshape(-1)
        keep = target != self.ignore_index
        predicted, target = predicted[keep], target[keep]
        if predicted.size == 0:
            return
        index = target.astype(np.int64) * self.n_classes + predicted.astype(np.int64)
        counts = np.bincount(index, minlength=self.n_classes ** 2)
        self.matrix += counts.reshape(self.n_classes, self.n_classes)

    def summary(self):
        m = self.matrix.astype(np.float64)
        true_positive = np.diag(m)
        predicted_total = m.sum(axis=0)
        actual_total = m.sum(axis=1)
        union = predicted_total + actual_total - true_positive

        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where(predicted_total > 0, true_positive / predicted_total, np.nan)
            recall = np.where(actual_total > 0, true_positive / actual_total, np.nan)
            f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
            iou = np.where(union > 0, true_positive / union, np.nan)

        per_class = {}
        for i, name in enumerate(CLASS_NAMES[:self.n_classes]):
            per_class[name] = {
                'precision': float(precision[i]), 'recall': float(recall[i]),
                'f1': float(f1[i]), 'iou': float(iou[i]),
                'support': int(actual_total[i]),
            }
        wave_slice = slice(1, self.n_classes)
        return {
            'per_class': per_class,
            'mean_iou_waves': float(np.nanmean(iou[wave_slice])),
            'mean_f1_waves': float(np.nanmean(f1[wave_slice])),
            'mean_iou_all': float(np.nanmean(iou)),
            'pixel_accuracy': float(true_positive.sum() / m.sum()) if m.sum() else float('nan'),
            'confusion': self.matrix.tolist(),
        }


class FiducialAccumulator(object):
    """Collect the signed errors and the detection counts landmark by landmark."""

    def __init__(self, landmarks=None, tolerances_ms=(10, 25, 50, 75, 150), primary_ms=25, fs_hz=500.0):
        self.landmarks = list(landmarks or LANDMARKS)
        self.tolerances = [float(t) for t in tolerances_ms]
        self.primary = float(primary_ms)
        self.step_ms = 1000.0 / float(fs_hz)
        self.errors = {name: [] for name in self.landmarks}
        self.pairs = {name: {'predicted': [], 'truth': []} for name in self.landmarks}
        self.presence = {name: {'truth': 0, 'predicted': 0, 'paired': 0}
                         for name in self.landmarks}
        self.counts = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for name in self.landmarks}

    def update_one(self, predicted, truth):
        for name in self.landmarks:
            p = predicted.get(name)
            t = truth.get(name)
            p_ok = p is not None and np.isfinite(float(p))
            t_ok = t is not None and np.isfinite(float(t))
            self.presence[name]['truth'] += int(t_ok)
            self.presence[name]['predicted'] += int(p_ok)
            self.presence[name]['paired'] += int(t_ok and p_ok)
            if t_ok and p_ok:
                error_ms = (float(p) - float(t)) * self.step_ms
                if abs(error_ms) <= self.primary:
                    self.counts[name]['tp'] += 1
                else:
                    # Beyond tolerance the landmark was not detected and something else was.
                    self.counts[name]['fp'] += 1
                    self.counts[name]['fn'] += 1
                self.errors[name].append(error_ms)
                self.pairs[name]['predicted'].append(float(p) * self.step_ms)
                self.pairs[name]['truth'].append(float(t) * self.step_ms)
            elif t_ok and not p_ok:
                self.counts[name]['fn'] += 1
            elif p_ok and not t_ok:
                self.counts[name]['fp'] += 1
            else:
                self.counts[name]['tn'] += 1

    def update(self, predicted_list, truth_list):
        for predicted, truth in zip(predicted_list, truth_list):
            self.update_one(predicted, truth)

    def summary(self):
        out = {}
        for name in self.landmarks:
            errors = np.array(self.errors[name], dtype=np.float64)
            counts = self.counts[name]
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            entry = {
                'n_compared': int(errors.size),
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': counts['tn'],
                'sensitivity': float(tp) / (tp + fn) if (tp + fn) else float('nan'),
                'precision': float(tp) / (tp + fp) if (tp + fp) else float('nan'),
                'detection_coverage': (float(self.presence[name]['paired'])
                                       / self.presence[name]['truth']
                                       if self.presence[name]['truth'] else float('nan')),
                'n_truth_present': int(self.presence[name]['truth']),
                'n_prediction_present': int(self.presence[name]['predicted']),
            }
            denominator = entry['sensitivity'] + entry['precision']
            entry['f1'] = (2 * entry['sensitivity'] * entry['precision'] / denominator) \
                if denominator and np.isfinite(denominator) and denominator > 0 else float('nan')
            if errors.size:
                entry['bias_ms'] = float(np.mean(errors))
                entry['mae_ms'] = float(np.mean(np.abs(errors)))
                entry['sd_ms'] = float(np.std(errors, ddof=1)) if errors.size > 1 else float('nan')
                entry['median_abs_ms'] = float(np.median(np.abs(errors)))
                entry['p95_abs_ms'] = float(np.percentile(np.abs(errors), 95))
                for tol in self.tolerances:
                    entry['within_%dms' % int(tol)] = float(np.mean(np.abs(errors) <= tol))
                predicted = np.asarray(self.pairs[name]['predicted'], dtype=np.float64)
                truth = np.asarray(self.pairs[name]['truth'], dtype=np.float64)
                entry['predicted_sd_ms'] = (float(np.std(predicted, ddof=1))
                                            if predicted.size > 1 else float('nan'))
                entry['reference_sd_ms'] = (float(np.std(truth, ddof=1))
                                            if truth.size > 1 else float('nan'))
                if truth.size > 1 and float(np.var(truth)) > 0 and float(np.var(predicted)) > 0:
                    entry['response_slope'] = float(np.cov(truth, predicted, ddof=0)[0, 1]
                                                    / np.var(truth))
                    entry['response_r'] = float(np.corrcoef(truth, predicted)[0, 1])
                else:
                    entry['response_slope'] = float('nan')
                    entry['response_r'] = float('nan')
            else:
                entry.update({'bias_ms': float('nan'), 'mae_ms': float('nan'), 'sd_ms': float('nan'),
                              'median_abs_ms': float('nan'), 'p95_abs_ms': float('nan'),
                              'predicted_sd_ms': float('nan'), 'reference_sd_ms': float('nan'),
                              'response_slope': float('nan'), 'response_r': float('nan')})
                for tol in self.tolerances:
                    entry['within_%dms' % int(tol)] = float('nan')
            out[name] = entry

        boundary = [n for n in self.landmarks if n.endswith('_onset') or n.endswith('_offset')]
        peaks = [n for n in self.landmarks if n.endswith('_peak')]
        out['_aggregate'] = {
            'boundary_mae_ms': _pooled(out, boundary, 'mae_ms'),
            'peak_mae_ms': _pooled(out, peaks, 'mae_ms'),
            'all_mae_ms': _pooled(out, self.landmarks, 'mae_ms'),
            'boundary_sensitivity': _pooled(out, boundary, 'sensitivity'),
            'boundary_precision': _pooled(out, boundary, 'precision'),
        }
        return out


def _pooled(summary, names, key):
    values = [summary[n][key] for n in names if n in summary and np.isfinite(summary[n].get(key, np.nan))]
    return float(np.mean(values)) if values else float('nan')


def landmarks_from_vector(vector):
    """Turn the eleven element truth vector produced by the dataset into a fiducial dict."""
    out = {}
    for i, name in enumerate(LANDMARKS):
        value = float(vector[i])
        out[name] = None if not np.isfinite(value) else value
    return out


def format_report(seg_summary, fid_summary, bio_summary, title='evaluation'):
    """A plain text table, which is what goes into the results chapter."""
    lines = ['', '=' * 78, title, '=' * 78, '', 'Segmentation']
    lines.append('  %-12s %8s %8s %8s %8s %12s' % ('class', 'prec', 'rec', 'f1', 'iou', 'support'))
    for name, entry in seg_summary['per_class'].items():
        lines.append('  %-12s %8.4f %8.4f %8.4f %8.4f %12d'
                     % (name, entry['precision'], entry['recall'], entry['f1'], entry['iou'],
                        entry['support']))
    lines.append('  mean IoU over the three wave classes  %.4f' % seg_summary['mean_iou_waves'])
    lines.append('  mean F1 over the three wave classes   %.4f' % seg_summary['mean_f1_waves'])
    lines.append('  per sample accuracy                   %.4f' % seg_summary['pixel_accuracy'])

    lines += ['', 'Fiducials, all timings in milliseconds']
    header = '  %-12s %7s %7s %8s %8s %8s %8s %8s' % ('landmark', 'sens', 'prec', 'bias', 'mae', 'sd', 'p95', 'w25ms')
    lines.append(header)
    for name in LANDMARKS:
        if name not in fid_summary:
            continue
        e = fid_summary[name]
        lines.append('  %-12s %7.3f %7.3f %8.2f %8.2f %8.2f %8.2f %8.3f'
                     % (name, e['sensitivity'], e['precision'], e['bias_ms'], e['mae_ms'],
                        e['sd_ms'], e['p95_abs_ms'], e.get('within_25ms', float('nan'))))
    agg = fid_summary.get('_aggregate', {})
    lines.append('  boundary MAE %.2f ms, peak MAE %.2f ms'
                 % (agg.get('boundary_mae_ms', float('nan')), agg.get('peak_mae_ms', float('nan'))))

    lines += ['', 'Timing response, prediction regressed on reference']
    lines.append('  %-12s %9s %9s %12s %12s' %
                 ('landmark', 'slope', 'r', 'pred SD ms', 'ref SD ms'))
    for name in LANDMARKS:
        if not (name.endswith('_onset') or name.endswith('_offset')) or name not in fid_summary:
            continue
        e = fid_summary[name]
        lines.append('  %-12s %9.3f %9.3f %12.2f %12.2f' %
                     (name, e.get('response_slope', float('nan')),
                      e.get('response_r', float('nan')),
                      e.get('predicted_sd_ms', float('nan')),
                      e.get('reference_sd_ms', float('nan'))))

    lines += ['', 'Biomarkers, all values in milliseconds']
    lines.append('  %-16s %10s %10s %10s %10s' % ('biomarker', 'bias', 'mae', 'sd', 'coverage'))
    for name, entry in bio_summary.items():
        lines.append('  %-16s %10.2f %10.2f %10.2f %10.3f'
                     % (name, entry['bias_ms'], entry['mae_ms'], entry['sd_ms'], entry['coverage']))
    lines.append('')
    return '\n'.join(lines)
