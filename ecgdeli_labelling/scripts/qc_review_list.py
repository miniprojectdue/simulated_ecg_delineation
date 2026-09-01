#!/usr/bin/env python3
"""
qc_review_list.py  -  Flag low-quality ECGdeli beats for manual review( if time allows)

Streams the ECGdeli fiducials CSV (works on the full 2.6M-row file) and writes
  * medalcare_qc_review_list.csv     one row per CRITICAL (record, lead, beat) + reasons
  * medalcare_qc_record_summary.csv  one row per record (worst first)

Severity tiers (a small tolerance ignores trivial boundary wobble)
  CRITICAL  -> review/fix or exclude. Any of
      - structural QRS-internal order wrong  (QRSon<Q<R<S<QRSoff violated > TOL)
      - QRSdur / QT / PR outside physiological range
      - gross boundary inversion > GROSS samples
      - R peak missing
  MINOR     -> moderate P/T boundary inversion (TOL..GROSS samples), usually fine
  CLEAN     -> nothing, or inversion <= TOL samples (negligible)

Only CRITICAL beats go in the review list, so manual effort is targeted.

Usage python3 qc_review_list.py [fiducials.csv]
"""

import csv, sys, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT,"ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
OUTDIR = os.path.join(ROOT,"ecgdeli_labelling","data","qc")
REVIEW = os.path.join(OUTDIR, "medalcare_qc_review_list.csv")
SUMMARY = os.path.join(OUTDIR, "medalcare_qc_record_summary.csv")

# ---- thresholds ----
TOL   = 3      # samples ignore boundary inversions this small (<=6 ms @500Hz)
GROSS = 20     # samples boundary inversion above this (>40 ms) is critical
QRS_MIN, QRS_MAX = 40, 200      # ms
QT_MIN,  QT_MAX  = 250, 700     # ms
PR_MIN,  PR_MAX  = 80, 400      # ms

ORDER = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
         "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
LBL   = ["P_on","P_pk","P_off","QRSon","Q","R","S","QRSoff","T_on","T_pk","T_off"]
QRSset = {"QRSon","Q","R","S","QRSoff"}

def gi(r, k):
    v = r.get(k, ""); return int(v) if v not in ("", "None") else None

review = []
per_record = defaultdict(lambda: {"total": 0, "critical": 0, "minor": 0, "reasons": Counter(),
                                  "disease_class": "", "mi_subclass": "", "split": ""})
# per (record, lead) a MedalCare-XL ECG repeats a common P/QRST template with beat-level RR
# and repolarisation-timing variation, so the beats of a (record,lead) share one source morphology.
# A lone flagged beat among clean siblings is a transient delineation glitch, a unit is only a
# persistently flagged unit when a majority of its beats are flagged. Tracking this separates
# persistent issues from transient per-beat glitches.
per_rl = defaultdict(lambda: {"total": 0, "critical": 0, "disease_class": ""})
crit_tally = Counter()
total = nminor = 0

for r in csv.DictReader(open(SRC)):
    total += 1
    rec = r["record_id"]; pr_ = per_record[rec]
    rl_ = per_rl[(rec, r["lead"])]; rl_["total"] += 1; rl_["disease_class"] = r["disease_class"]
    pr_["total"] += 1
    pr_["disease_class"] = r["disease_class"]; pr_["mi_subclass"] = r.get("mi_subclass", ""); pr_["split"] = r["split"]
    fs = float(r["fs_hz"]); ms = 1000.0 / fs
    reasons = []

    seq = [(LBL[i], gi(r, k)) for i, k in enumerate(ORDER)]
    seq = [(l, v) for l, v in seq if v is not None]
    struct = False; gross = 0; mild = 0
    for (la, va), (lb, vb) in zip(seq, seq[1:]):
        if va > vb:
            m = va - vb
            if la in QRSset and lb in QRSset and m > TOL:
                struct = True
            elif m > GROSS:
                gross = max(gross, m)
            elif m > TOL:
                mild = max(mild, m)
    if struct: reasons.append("QRS_structural")
    if gross:  reasons.append(f"gross_boundary:{gross}smp")

    # biomarkers
    qon, qoff = gi(r, "qrs_onset_sample"), gi(r, "qrs_offset_sample")
    pon, toff = gi(r, "p_onset_sample"), gi(r, "t_offset_sample")
    if qon is not None and qoff is not None:
        d = (qoff - qon) * ms
        if d < QRS_MIN or d > QRS_MAX: reasons.append(f"QRSdur={d:.0f}ms")
    if qon is not None and toff is not None:
        d = (toff - qon) * ms
        if d < QT_MIN or d > QT_MAX: reasons.append(f"QT={d:.0f}ms")
    if pon is not None and qon is not None:
        d = (qon - pon) * ms
        if d < PR_MIN or d > PR_MAX: reasons.append(f"PR={d:.0f}ms")
    if r["qrs_present"] != "1" or gi(r, "r_peak_sample") is None:
        reasons.append("R_missing")

    if reasons:                       # CRITICAL
        pr_["critical"] += 1; rl_["critical"] += 1
        for f in reasons:
            k = f.split(":")[0].split("=")[0]; pr_["reasons"][k] += 1; crit_tally[k] += 1
        review.append({"record_id": rec, "disease_class": r["disease_class"], "mi_subclass": r.get("mi_subclass", ""),
            "split": r["split"], "lead": r["lead"], "beat_id": r["beat_id"], "flags": ";".join(reasons),
            "p_onset_sample": r["p_onset_sample"], "p_offset_sample": r["p_offset_sample"],
            "qrs_onset_sample": r["qrs_onset_sample"], "r_peak_sample": r["r_peak_sample"],
            "qrs_offset_sample": r["qrs_offset_sample"], "t_peak_sample": r["t_peak_sample"], "t_offset_sample": r["t_offset_sample"]})
    elif mild:                        # MINOR
        pr_["minor"] += 1; nminor += 1

# write review list
if review:
    with open(REVIEW, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys())); w.writeheader(); w.writerows(review)

# per-record summary, worst (by critical rate) first
summ = []
for rec, d in per_record.items():
    frac = d["critical"] / d["total"] if d["total"] else 0
    dom = d["reasons"].most_common(1)[0][0] if d["reasons"] else ""
    summ.append({"record_id": rec, "disease_class": d["disease_class"], "mi_subclass": d["mi_subclass"], "split": d["split"],
                 "beats_total": d["total"], "beats_critical": d["critical"], "beats_minor": d["minor"],
                 "frac_critical": round(frac, 4), "dominant_reason": dom})
summ.sort(key=lambda x: -x["frac_critical"])
with open(SUMMARY, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

ncrit = len(review)
print(f"beats total            : {total}")
print(f"CRITICAL (review these): {ncrit}  ({100*ncrit/total:.2f}%)  -> medalcare_qc_review_list.csv")
print(f"MINOR (small P/T wobble): {nminor}  ({100*nminor/total:.2f}%)  -> usually fine, left out of review list")
print(f"CLEAN                   : {total-ncrit-nminor}  ({100*(total-ncrit-nminor)/total:.2f}%)")
print(f"records                 : {len(per_record)}   fully-clean records: {sum(1 for d in per_record.values() if d['critical']==0)}")
print("\ncritical-flag reasons:")
for k, v in crit_tally.most_common():
    print(f"  {k:16s}: {v}")
print("\ncritical rate by disease class:")
agg = defaultdict(lambda: [0, 0])
for d in per_record.values():
    agg[d["disease_class"]][0] += d["critical"]; agg[d["disease_class"]][1] += d["total"]
for c in sorted(agg, key=lambda k: -agg[k][0]/agg[k][1]):
    cr, to = agg[c]; print(f"  {c:10s}: {100*cr/to:5.1f}%  ({cr}/{to})")
print("\nworst 5 records to review first:")
for s in summ[:5]:
    print(f"  {s['record_id']:36s} {s['disease_class']:8s} {s['beats_critical']}/{s['beats_total']} ({100*s['frac_critical']:.0f}%)  {s['dominant_reason']}")

# ---- per-(record,lead) view separate persistently flagged units from transient per-beat glitches ----
# Because each ECG repeats one source template, the meaningful unit is (record, lead). A unit is a
# persistently flagged unit only if a majority (>=50%) of its beats are flagged, otherwise the flag is
# a transient glitch on an isolated beat of an otherwise-clean repeated morphology.
PERSIST_FRAC = 0.5
n_units = len(per_rl)
units_any  = sum(1 for d in per_rl.values() if d["critical"] > 0)
units_gen  = sum(1 for d in per_rl.values() if d["total"] and d["critical"]/d["total"] >= PERSIST_FRAC)
crit_in_gen = sum(d["critical"] for d in per_rl.values() if d["total"] and d["critical"]/d["total"] >= PERSIST_FRAC)
transient_share = 100*(1 - crit_in_gen/ncrit) if ncrit else 0
print("\nper-(record,lead) view  [each ECG repeats one source template, so this is the meaningful unit]:")
print(f"  (record,lead) units              : {n_units}")
print(f"  units with >=1 critical beat     : {units_any}  ({100*units_any/n_units:.2f}%)")
print(f"  persistently flagged units (>=50%): {units_gen}  ({100*units_gen/n_units:.2f}%)")
print(f"  transient share of critical beats: {transient_share:.0f}%  (isolated glitches a per-record consensus label would absorb)")
print("\npersistent-flag rate by disease class (per record-lead):")
gagg = defaultdict(lambda: [0, 0])
for d in per_rl.values():
    gen = 1 if (d["total"] and d["critical"]/d["total"] >= PERSIST_FRAC) else 0
    gagg[d["disease_class"]][0] += gen; gagg[d["disease_class"]][1] += 1
for c in sorted(gagg, key=lambda k: -gagg[k][0]/gagg[k][1]):
    g, to = gagg[c]; print(f"  {c:10s}: {100*g/to:5.2f}%  ({g}/{to})")
