#!/usr/bin/env python3
"""
make_labelfree_external_units.py  -  rebuild the external test table with LABEL-FREE crop placement.

Motivation
----------
The shipped external table (smith2026_test_units_alllead.csv) sets, for every row,

    win_start_sample = 0
    win_end_sample   = t_offset_sample + 20

so the crop window - which the loader centres in the crop and which names the beat for region
selection - is a function of the reference T offset being scored. Empirically win_end == t_offset+20
on 1200/1200 rows and 96/100 records receive a different per-lead shift. That is label leakage: the
placement and the region-identity mask both read the answer. No score computed that way is
deployment-valid.

This script rewrites win_start_sample / win_end_sample from the SIGNAL ALONE, never from the
reference fiducials, and writes two tables:

  QRS-anchored (primary corrected evaluation)
      R-anchor = argmax of the spatial-magnitude curve over the eight independent leads
      (I, II, V1..V6 - the MAG_LEADS the model receives as channel 13), computed once per record
      from the raw 12-lead signal and shared by all twelve leads. The window is placed at
          [anchor - LEAD_IN, anchor + LEAD_OUT]
      with LEAD_IN / LEAD_OUT the MEDIAN R-peak->window-start / R-peak->window-end offsets measured
      on the pretraining corpus (211 / 220 samples at 500 Hz, width 432 ~= training median 433,
      R-peak fractional position 0.487). This reproduces the position the network was trained to
      look at, using only the signal at test time. The window may extend left of sample 0 (the
      external records begin at the QRS, so the pre-beat interval is genuinely absent and becomes
      padding); the crop centres the anchor regardless.

  record (secondary stress test)
      win_start = 0, win_end = n_samples-1, identical for all twelve leads. The simplest possible
      label-free placement. The beat sits off-centre, which the evaluator's geometry_report will
      flag; it is included only to show the external result is not an artifact of one placement rule.

The reference fiducials are copied through UNCHANGED as the scoring truth. They are read here only
(a) to be written back verbatim and (b) for an offline validation of the detector (does the
signal anchor fall inside the reference QRS?) that is reported but never used for placement.

    python3 make_labelfree_external_units.py \
        --in-table  test_export/smith2026_test_units_alllead.csv \
        --signals   test_export/signals \
        --out-qrs   test_export/smith2026_test_units_qrsanchor.csv \
        --out-record test_export/smith2026_test_units_record.csv \
        --write

Dry run by default; nothing is written without --write.
"""
import argparse
import collections
import csv
import os

import numpy as np

# The eight independent leads, in the 12-lead order I, II, III, aVR, aVL, aVF, V1..V6.
# Identical to dataset.MAG_LEADS. The augmented limb leads are excluded so the frontal plane
# is not triple counted; this is the curve the model receives as its thirteenth channel and the
# curve evaluate.qrs_anchor reads.
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
MAG_LEADS = [0, 1, 6, 7, 8, 9, 10, 11]


def load_signal(path):
    arr = np.loadtxt(path, delimiter=',', dtype=np.float64)
    if arr.shape[0] != 12 and arr.shape[1] == 12:
        arr = arr.T
    return arr


def spatial_magnitude(sig):
    """Euclidean norm over the eight independent leads, baselined on the per-lead median.

    Matches dataset.spatial_magnitude. The whole external record is observed, so the baseline is
    the median over all samples rather than over a validity mask.
    """
    x = sig[MAG_LEADS]
    base = np.median(x, axis=1, keepdims=True)
    return np.sqrt(((x - base) ** 2).sum(axis=0))


def r_anchor(sig):
    """The label-free ventricular anchor: the sample of largest spatial magnitude (the R peak).

    Reads the signal only. argmax of the magnitude curve is the dominant deflection of the beat;
    on this corpus it falls inside the reference QRS on every record and sits a median of one
    sample from the reference R peak, so it is a faithful stand-in that never touches a fiducial.
    """
    return int(np.argmax(spatial_magnitude(sig)))


def rewrite(rows, signals_dir, lead_in, lead_out, mode):
    """Return rows with win_start_sample / win_end_sample rewritten label-free for one mode."""
    by_rec = collections.OrderedDict()
    for r in rows:
        by_rec.setdefault(r['record_id'], []).append(r)

    out = []
    validation = []  # (rec, anchor, qrs_on, qrs_off, r_peak, ws, we, n)
    for rec, rs in by_rec.items():
        n = int(rs[0]['n_samples'])
        path = rs[0]['path_raw']
        if not os.path.exists(path):
            path = os.path.join(signals_dir, rec + '_raw.csv')
        sig = load_signal(path)
        n = sig.shape[1] if sig.shape[1] else n

        if mode == 'record':
            ws, we, anchor = 0, n - 1, None
        elif mode == 'qrs':
            anchor = r_anchor(sig)
            # No left clamp: the pre-beat interval is absent in these QRS-aligned records, so the
            # window extends into padding and the crop still centres the anchor. Clamp only the
            # right edge to the last real sample so the window never names padding as beat.
            ws = anchor - lead_in
            we = min(n - 1, anchor + lead_out)
        else:
            raise ValueError('unknown mode %r' % mode)

        # validation vs reference (NOT used for placement)
        q_on = rs[0].get('qrs_onset_sample', '')
        q_off = rs[0].get('qrs_offset_sample', '')
        r_pk = rs[0].get('r_peak_sample', '')
        validation.append((rec, anchor, q_on, q_off, r_pk, ws, we, n))

        for r in rs:
            new = dict(r)
            new['win_start_sample'] = str(ws)
            new['win_end_sample'] = str(we)
            out.append(new)
    return out, validation


def report(mode, validation, lead_in, lead_out):
    n = len(validation)
    in_qrs = 0
    near_r = []
    widths = []
    centre_off = []
    for rec, anchor, q_on, q_off, r_pk, ws, we, nS in validation:
        widths.append(we - ws + 1)
        if anchor is not None:
            centre_off.append(anchor - (ws + we) / 2.0)
            if q_on != '' and q_off != '' and int(q_on) <= anchor <= int(q_off):
                in_qrs += 1
            if r_pk != '':
                near_r.append(anchor - int(r_pk))
    w = np.array(widths, dtype=float)
    lines = ['[%s] records %d' % (mode, n),
             '  window width samples: median %.0f  min %.0f  max %.0f' % (np.median(w), w.min(), w.max())]
    if mode == 'qrs':
        nr = np.array(near_r, dtype=float)
        co = np.array(centre_off, dtype=float)
        lines += [
            '  detector validation (NOT used for placement):',
            '    signal anchor inside reference QRS: %d/%d (%.0f%%)' % (in_qrs, n, 100 * in_qrs / max(n, 1)),
            '    anchor - reference R peak (samples): median %.1f  p10 %.1f  p90 %.1f'
            % (np.median(nr), np.percentile(nr, 10), np.percentile(nr, 90)),
            '  anchor offset from window centre: median %.1f  p10 %.1f  p90 %.1f (0 = centred)'
            % (np.median(co), np.percentile(co, 10), np.percentile(co, 90)),
            '  placement offsets used: LEAD_IN=%d LEAD_OUT=%d (pretrain-corpus medians)' % (lead_in, lead_out)]
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument('--in-table', default=os.path.join(here, 'test_export', 'smith2026_test_units_alllead.csv'))
    p.add_argument('--signals', default=os.path.join(here, 'test_export', 'signals'))
    p.add_argument('--out-qrs', default=os.path.join(here, 'test_export', 'smith2026_test_units_qrsanchor.csv'))
    p.add_argument('--out-record', default=os.path.join(here, 'test_export', 'smith2026_test_units_record.csv'))
    p.add_argument('--lead-in', type=int, default=211)
    p.add_argument('--lead-out', type=int, default=220)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()

    with open(a.in_table, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    print('read %d rows over %d records from %s'
          % (len(rows), len({r['record_id'] for r in rows}), a.in_table))

    qrs_rows, qrs_val = rewrite(rows, a.signals, a.lead_in, a.lead_out, 'qrs')
    rec_rows, rec_val = rewrite(rows, a.signals, a.lead_in, a.lead_out, 'record')
    print(report('qrs', qrs_val, a.lead_in, a.lead_out))
    print(report('record', rec_val, a.lead_in, a.lead_out))

    if not a.write:
        print('\ndry run, nothing written. Re-run with --write.')
        return
    for path, data in ((a.out_qrs, qrs_rows), (a.out_record, rec_rows)):
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(data)
        print('wrote %s (%d rows)' % (path, len(data)))


if __name__ == '__main__':
    main()
