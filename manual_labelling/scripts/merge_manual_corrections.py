#!/usr/bin/env python3
"""
merge_manual_corrections.py  -  Fold the manual fiducial corrections (from the corrector tool) back
into master_labels.csv, propagating each corrected representative beat to all beats of its (record,
lead) unit.

Propagation.  Each MedalCare beat is the same P/QRST template with RR variation and a per-beat
stretch of the QRS-offset -> T-offset segment.  The beat-to-beat displacement is measured entirely
inside ECGdeli's own frame, so the reviewer's corrected geometry is transplanted unchanged onto the
representative beat and merely translated onto the others.
  * anchor A(beat) = the beat's ECGdeli qrs_onset_sample (falls back to r_peak_sample)
  * d = A(beat) - A(representative beat), taken from the pre-merge ECGdeli values
  * P and QRS fiducials     new = corr_k + d
  * T fiducials are anchored at the new QRS-offset and scaled by the beat's RR
        new = new_qrs_off + (corr_k - corr_qrs_off) * (beat_RR / corr_RR)
  * presence flags (p/qrs/q/r/s/t_present) are taken from the corrected beat.
Corrected rows get label_source=manual_corrected and refreshed *_ms + QT.  label_quality is
"corrected" on the reviewed beat itself and "propagated" on the other beats of the same unit.

The review relabels which QRS extremum is R, so the corrected r_peak is not the same landmark as ECGdeli's r_peak, and
49 of the reviewed units have no R at all.

If master_labels.csv predates the Q/R/S presence review it will not carry q_present, r_present and
s_present columns.  They are appended here so the review is not silently discarded.  Rows that were
never reviewed are left empty in those columns, meaning unknown, rather than being back-filled from
the estimator, whose q_peak placeholder is unreliable.

Input one or more corrections CSVs exported by the tool (reviewed rows only are applied).
Usage python3 merge_manual_corrections.py corrections1.csv [corrections2.csv ...]
        (writes master_labels.csv in place, after a .bak copy)
"""
import os, sys, csv, shutil

csv.field_size_limit(10 ** 9)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)
MASTER = os.path.join(ROOT, "dataset_curation", "data", "assembled", "master_labels.csv")

PQRS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample",
        "q_peak_sample","r_peak_sample","s_peak_sample","qrs_offset_sample"]
TFID = ["t_onset_sample","t_peak_sample","t_offset_sample"]
FKEYS = PQRS + TFID
PRESENCE = ("p_present", "qrs_present", "q_present", "r_present", "s_present", "t_present")
NEW_COLS = ("q_present", "r_present", "s_present")
MS = 2.0  # 500 Hz


def gi(v):
    return int(float(v)) if v not in ("", "None", None) else None


def load_corrections(paths):
    """key (record,lead) -> corrected fiducial samples + presence + reviewed beat_id."""
    corr = {}
    for p in paths:
        for r in csv.DictReader(open(p, newline="")):
            if r.get("reviewed", "0") != "1":
                continue
            unit = (r["record_id"], r["lead"])
            c = {k: gi(r.get(k, "")) for k in FKEYS}
            c.update({f: r.get(f, "") for f in PRESENCE})
            c["_beat_id"] = gi(r.get("beat_id", ""))
            corr[unit] = c
    return corr


def anchor_of(row, idx):
    """Beat anchor in the ECGdeli frame: QRS onset, else R peak.  A boundary is preferred because
    relabelling which extremum is R does not move it."""
    a = gi(row[idx["qrs_onset_sample"]])
    return a if a is not None else gi(row[idx["r_peak_sample"]])


def propagate(unit_rows, c, header):
    """Apply corrected fiducials c to every beat row of one unit (list of row-lists), in place."""
    idx = {name: i for i, name in enumerate(header)}
    bid_i = idx.get("beat_id")
    # anchors must be read before any row is overwritten
    A = [anchor_of(row, idx) for row in unit_rows]
    # per-beat RR (next anchor - this anchor), last beat reuses previous, fall back to median
    rr = []
    for i in range(len(A)):
        rr.append(A[i + 1] - A[i] if (i + 1 < len(A) and A[i] is not None and A[i + 1] is not None) else None)
    known = sorted(x for x in rr if x)
    med = known[len(known) // 2] if known else None
    rr = [x if x else med for x in rr]

    corr_qon = c["qrs_onset_sample"]
    corr_qoff = c["qrs_offset_sample"]

    # locate the reviewed beat: by beat_id, else the beat whose ECGdeli anchor is nearest corr_qon
    rep = None
    if bid_i is not None and c.get("_beat_id") is not None:
        for i, row in enumerate(unit_rows):
            if gi(row[bid_i]) == c["_beat_id"]:
                rep = i
                break
    if rep is None and corr_qon is not None:
        cand = [(abs(a - corr_qon), i) for i, a in enumerate(A) if a is not None]
        if cand:
            rep = min(cand)[1]
    if rep is None or A[rep] is None:
        return 0, 0
    A0 = A[rep]
    corr_RR = rr[rep] or med

    ns_i = idx.get("n_samples")
    oob = 0
    for bi, row in enumerate(unit_rows):
        if A[bi] is None:
            continue
        nmax = (gi(row[ns_i]) or 5000) - 1 if ns_i is not None else 4999
        d = A[bi] - A0
        new_qoff = corr_qoff + d if corr_qoff is not None else None
        for k in FKEYS:
            v = c[k]
            if v is None:
                row[idx[k]] = ""                        # reviewed as absent
                continue
            if k in PQRS:
                nv = v + d
            elif new_qoff is None or corr_qoff is None:
                row[idx[k]] = ""
                continue
            else:                                       # T fiducial anchored at new QRS offset, RR scaled
                sc = (rr[bi] / corr_RR) if (corr_RR and rr[bi]) else 1.0
                sc = min(2.0, max(0.5, sc))
                nv = int(round(new_qoff + (v - corr_qoff) * sc))
            # an edge beat can push a fiducial off the end of the record; a sample index outside
            # the record is not usable, so it is left empty rather than written negative or past n
            if 0 <= nv <= nmax:
                row[idx[k]] = str(nv)
            else:
                row[idx[k]] = ""
                oob += 1
        # refresh *_ms for the fiducials + QT, presence, provenance
        for k in FKEYS:
            ms_col = k.replace("_sample", "_ms")
            if ms_col in idx:
                s = row[idx[k]]
                row[idx[ms_col]] = "" if s == "" else f"{int(s) * MS:.3f}"
        for f in PRESENCE:
            if f in idx and c.get(f) not in (None, ""):
                row[idx[f]] = c[f]
        if "qt_interval_ms" in idx:
            qon, tof = row[idx["qrs_onset_sample"]], row[idx["t_offset_sample"]]
            row[idx["qt_interval_ms"]] = f"{(int(tof)-int(qon))*MS:.3f}" if (qon and tof) else ""
        if "qt_peak_ms" in idx:
            qon, tpk = row[idx["qrs_onset_sample"]], row[idx["t_peak_sample"]]
            row[idx["qt_peak_ms"]] = f"{(int(tpk)-int(qon))*MS:.3f}" if (qon and tpk) else ""
        if "label_source" in idx: row[idx["label_source"]] = "manual_corrected"
        if "label_quality" in idx: row[idx["label_quality"]] = "corrected" if bi == rep else "propagated"
    return 1, oob


def main(paths):
    corr = load_corrections(paths)
    if not corr:
        sys.exit("no reviewed corrections found in the given CSV(s)")
    print(f"corrected units to apply: {len(corr):,}", flush=True)
    shutil.copy(MASTER, MASTER + ".bak")
    tmp = MASTER + ".tmp"
    applied = 0; rows_out = 0; split_units = 0; oob_total = 0
    seen_runs = {}
    with open(MASTER, newline="") as fin, open(tmp, "w", newline="") as fout:
        r = csv.reader(fin); w = csv.writer(fout)
        header = next(r)
        added = [cn for cn in NEW_COLS if cn not in header]
        pad = [""] * len(added)
        header = header + added
        if added:
            print(f"appended presence columns: {', '.join(added)}", flush=True)
        w.writerow(header)
        ri, li = header.index("record_id"), header.index("lead")
        buf = []; key = None
        def flush():
            nonlocal applied, rows_out, split_units, oob_total
            if not buf: return
            if key in corr:
                n = seen_runs.get(key, 0)
                if n:
                    split_units += 1
                seen_runs[key] = n + 1
                a, ob = propagate(buf, corr[key], header)
                applied += a; oob_total += ob
            w.writerows(buf)
            rows_out += len(buf)
            buf.clear()
        for row in r:
            if added:
                row = row + pad
            k = (row[ri], row[li])
            if k != key:
                flush(); key = k
            buf.append(row)
        flush()
    os.replace(tmp, MASTER)
    matched = len(seen_runs)
    print(f"applied corrections to {applied:,} units ({matched:,}/{len(corr):,} corrected units found in master)", flush=True)
    if oob_total:
        print(f"note: {oob_total:,} fiducial(s) on edge beats fell outside the record and were left empty", flush=True)
    if split_units:
        print(f"WARNING: {split_units:,} unit(s) were not contiguous in the file", flush=True)
    missing = [k for k in corr if k not in seen_runs]
    if missing:
        print(f"WARNING: {len(missing):,} corrected unit(s) had no matching rows, e.g. {missing[:5]}", flush=True)
    print(f"wrote {MASTER} ({rows_out:,} data rows; backup at {MASTER}.bak)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python3 merge_manual_corrections.py corrections1.csv [corrections2.csv ...]")
    main(sys.argv[1:])
