#!/usr/bin/env python3
"""
extract_signal.py  -  Pull ONE signal together with its fiducial labels from the Master files.

Given a record_id, it (1) finds the waveform via signals_index.csv, (2) copies the raw 12-lead
signal out, (3) extracts that record's rows from master_labels.csv (all leads/beats, with the
fiducial positions), and (4) optionally plots lead II with the fiducials overlaid.

Usage
    python3 extract_signal.py                       # first record in the index
    python3 extract_signal.py <record_id>           # a specific signal
    python3 extract_signal.py <record_id> --plot    # also draw lead II with fiducials

Outputs (in ./extracted/)
    <record_id>_signal.csv   the 12 x N raw waveform (mV)
    <record_id>_labels.csv   per-lead, per-beat fiducial labels for that signal
    <record_id>_leadII.png   (with --plot)
"""
import csv, os, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
IDX  = os.path.join(ROOT, "dataset_curation","data","review","signals_index.csv")
LAB  = os.path.join(ROOT, "dataset_curation","data","assembled","master_labels.csv")
OUT  = os.path.join(ROOT, "dataset_curation","data","audit_samples"); os.makedirs(OUT, exist_ok=True)

args = [a for a in sys.argv[1:]]
do_plot = "--plot" in args
rid = next((a for a in args if not a.startswith("--")), None)

# 1) locate the signal in the index
idx = {r["record_id"]: r for r in csv.DictReader(open(IDX))}
if rid is None: rid = next(iter(idx))
assert rid in idx, f"record_id '{rid}' not found in signals_index.csv"
info = idx[rid]
sig_path = os.path.join(ROOT, info["path_raw"])
assert os.path.isfile(sig_path), f"waveform missing: {sig_path}"

# 2) copy the raw waveform
dst = os.path.join(OUT, f"{rid}_signal.csv"); shutil.copy(sig_path, dst)

# 3) extract this record's label rows (stream-filter the big labels file)
labout = os.path.join(OUT, f"{rid}_labels.csv"); n = 0
with open(LAB, newline="") as fin, open(labout, "w", newline="") as fo:
    r = csv.reader(fin); w = csv.writer(fo)
    header = next(r); w.writerow(header); ri = header.index("record_id")
    for row in r:
        if row[ri] == rid:
            w.writerow(row); n += 1
        elif n:                       # rows for one record are contiguous -> stop after the block
            break
print(f"record: {rid}  ({info['disease_class']}, split={info['split']})")
print(f"  signal -> {dst}  (12 x {info['n_samples']} @ {info['fs_hz']} Hz)")
print(f"  labels -> {labout}  ({n} rows = leads x beats)")

# 4) optional lead-II plot with fiducials
if do_plot:
    import pandas as pd, numpy as np
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    M = pd.read_csv(sig_path, header=None).to_numpy()
    if M.shape[0] != 12: M = M.T
    sig = M[1] - np.median(M[1]); t = np.arange(len(sig)) * 2
    lab = pd.read_csv(labout); l2 = lab[lab.lead == "II"].sort_values("beat_id")
    end = int(l2.iloc[min(2, len(l2)-1)].t_offset_sample) + 80
    cmap = {"qrs_onset_sample":"#1E8449","r_peak_sample":"#000000","qrs_offset_sample":"#145A32",
            "t_onset_sample":"#2C7FB8","t_peak_sample":"#E67E22","t_offset_sample":"#7B3294"}
    fig, ax = plt.subplots(figsize=(12,4)); ax.plot(t[:end], sig[:end], color="#333", lw=0.9)
    ax.axhline(0, color="#bbb", ls=":", lw=0.6)
    for bi in range(min(3, len(l2))):
        for c, col in cmap.items():
            v = l2.iloc[bi][c]
            if pd.notna(v) and int(v) < end: ax.axvline(int(v)*2, color=col, lw=1, alpha=0.8)
    ax.set_title(f"{rid} — lead II with ECGdeli fiducials"); ax.set_xlabel("ms"); ax.set_ylabel("mV")
    p = os.path.join(OUT, f"{rid}_leadII.png"); plt.savefig(p, dpi=130, bbox_inches="tight")
    print(f"  plot   -> {p}")
