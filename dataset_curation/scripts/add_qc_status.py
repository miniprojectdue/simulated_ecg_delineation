#!/usr/bin/env python3
"""
add_qc_status.py  -  Append per-beat QC status to master_labels.csv.

Reproduces the exact rules of qc_review_list.py and adds two columns
  qc_status  in {critical, minor, clean}
  qc_flags   ','-joined reasons for critical beats ("" otherwise)

Rules (per beat)
  CRITICAL  - any of QRS-internal order violated (>TOL), a boundary inversion > GROSS samples,
              QRSdur outside 40-200 ms, QT outside 250-700 ms, PR outside 80-400 ms, or R missing.
  MINOR     - a moderate boundary inversion (TOL..GROSS samples), no critical reason.
  CLEAN     - nothing beyond a trivial (<= TOL) wobble.

Run python3 add_qc_status.py     (rewrites master_labels.csv in place via a temp file)
"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
SRC  = os.path.join(ROOT, "dataset_curation","data","assembled","master_labels.csv"); TMP = SRC + ".tmp"
TOL, GROSS = 3, 20
QRS_MIN, QRS_MAX = 40, 200
QT_MIN,  QT_MAX  = 250, 700
PR_MIN,  PR_MAX  = 80, 400
ORDER = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
         "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
QRS_POS = set(range(3, 8))   # 0-based positions of QRSon..QRSoff in ORDER

def gi(v):
    return int(v) if v not in ("", "None") else None

with open(SRC, newline="") as fin, open(TMP, "w", newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=r.fieldnames + ["qc_status", "qc_flags"]); w.writeheader()
    ncrit = nmin = nclean = 0
    for row in r:
        fs = float(row["fs_hz"] or 500); ms = 1000.0 / fs
        seq = [(i, gi(row[k])) for i, k in enumerate(ORDER)]
        seq = [(i, v) for i, v in seq if v is not None]
        struct = False; gross = 0; mild = 0
        for (ia, va), (ib, vb) in zip(seq, seq[1:]):
            if va > vb:
                m = va - vb
                if ia in QRS_POS and ib in QRS_POS and m > TOL: struct = True
                elif m > GROSS: gross = max(gross, m)
                elif m > TOL:   mild = max(mild, m)
        reasons = []
        if struct: reasons.append("QRS_structural")
        if gross:  reasons.append(f"gross_boundary:{gross}smp")
        qon, qoff = gi(row["qrs_onset_sample"]), gi(row["qrs_offset_sample"])
        pon, toff = gi(row["p_onset_sample"]),  gi(row["t_offset_sample"])
        if qon is not None and qoff is not None:
            d = (qoff - qon) * ms
            if d < QRS_MIN or d > QRS_MAX: reasons.append(f"QRSdur={d:.0f}ms")
        if qon is not None and toff is not None:
            d = (toff - qon) * ms
            if d < QT_MIN or d > QT_MAX: reasons.append(f"QT={d:.0f}ms")
        if pon is not None and qon is not None:
            d = (qon - pon) * ms
            if d < PR_MIN or d > PR_MAX: reasons.append(f"PR={d:.0f}ms")
        if row["qrs_present"] != "1" or gi(row["r_peak_sample"]) is None:
            reasons.append("R_missing")
        if reasons:   status, ncrit = "critical", ncrit + 1
        elif mild:    status, nmin  = "minor",    nmin + 1
        else:         status, nclean = "clean",   nclean + 1
        row["qc_status"] = status; row["qc_flags"] = ";".join(reasons)
        w.writerow(row)
os.replace(TMP, SRC)
print(f"critical {ncrit} | minor {nmin} | clean {nclean}")
