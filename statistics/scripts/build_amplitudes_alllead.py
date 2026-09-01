"""
build_amplitudes_alllead.py  -  All-lead healthy-sinus amplitude comparison vs the published
Table 6 'sim' means (the source of the report's Table 8 per-feature mean absolute difference and
lead-sign agreement). Amplitude = signal at the fiducial peak minus the pre-QRS isoelectric baseline
(median of the 12 samples ending at QRS onset), per-signal median over beats, then per-lead cohort
mean, then mean absolute difference across the 12 leads vs Table 6.

Sampling a REPRODUCIBLE RANDOM sample of N_AMP sinus recordings, drawn without replacement from the
sorted list of eligible records with a fixed seed, so the result is deterministic and unbiased.

Reads dataset_curation/data/assembled/master_labels.csv (peak positions), raw signals.
Writes statistics/data/amplitudes_alllead_sinus.csv , statistics/data/amplitude_sample_sinus.csv
"""
import os, csv
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA   = os.path.join(HERE, "..", "data")
MASTER = os.path.join(ROOT, "dataset_curation", "data", "assembled", "master_labels.csv")
SIGIDX = os.path.join(ROOT, "dataset_curation", "data", "review", "signals_index.csv")
N_AMP  = 200
SEED   = 2026

LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
LROW  = {l:i for i,l in enumerate(LEADS)}
AMP   = ["Pamp","Qamp","Ramp","Samp","Tamp"]
PK    = {"Pamp":"p_peak_sample","Qamp":"q_peak_sample","Ramp":"r_peak_sample",
         "Samp":"s_peak_sample","Tamp":"t_peak_sample"}
# Table 6 'sim' lead-wise amplitude means [P,Q,R,S,T]
T6A = {"I":[0.09,0.02,0.37,-0.05,-0.03],"II":[0.09,0.06,0.59,-0.20,0.49],"III":[0.03,0.08,0.11,-0.10,0.51],
 "aVR":[-0.09,-0.03,-0.49,0.13,-0.24],"aVL":[0.03,-0.03,0.14,0.02,-0.27],"aVF":[0.05,0.08,0.34,-0.15,0.49],
 "V1":[-0.05,0.13,-0.48,-0.05,0.65],"V2":[-0.02,0.21,-1.55,-0.02,1.47],"V3":[0.06,0.21,-0.95,-0.05,1.09],
 "V4":[0.06,0.21,-0.36,-0.11,0.79],"V5":[0.09,0.08,0.54,-0.23,0.46],"V6":[0.11,0.03,0.72,-0.17,0.30]}

# --- reproducible random sample of sinus records (fixed seed, sorted eligible list) ---
idx = {r["record_id"]: r for r in csv.DictReader(open(SIGIDX, newline=""))}
elig = sorted(r for r, v in idx.items() if v["disease_class"] == "sinus")
want = np.random.default_rng(SEED).choice(elig, size=min(N_AMP, len(elig)), replace=False).tolist()
wantset = set(want)
pd.DataFrame({"record_id": want}).to_csv(os.path.join(DATA, "amplitude_sample_sinus.csv"), index=False)
print(f"sampled {len(want)} sinus records (seed {SEED})")

# --- collect all-lead peak positions for the sampled records ---
beats = {}
cols = ["record_id", "disease_class", "lead", "qrs_onset_sample"] + list(PK.values())
for ch in pd.read_csv(MASTER, usecols=cols, dtype=str, na_filter=False, chunksize=400000):
    ch = ch[(ch.disease_class == "sinus") & (ch.record_id.isin(wantset))]
    for (rid, ld), g in ch.groupby(["record_id", "lead"]):
        beats.setdefault((rid, ld), []).extend(g.to_dict("records"))

# --- amplitude per (lead, feature) per-record median, then per-lead cohort mean ---
our = {ld: {f: [] for f in AMP} for ld in LEADS}
for rid in want:
    try:
        M = pd.read_csv(idx[rid]["path_raw"] if os.path.isabs(idx[rid]["path_raw"])
                        else os.path.join(ROOT, idx[rid]["path_raw"]), header=None).to_numpy()
    except Exception:
        continue
    if M.shape[0] != 12:
        M = M.T
    for ld in LEADS:
        sig = M[LROW[ld]]; tmp = {f: [] for f in AMP}
        for row in beats.get((rid, ld), []):
            qon = row["qrs_onset_sample"]
            if qon in ("", "None"):
                continue
            qon = int(float(qon)); base = np.median(sig[max(0, qon-11):qon+1])
            for f in AMP:
                v = row[PK[f]]
                if v not in ("", "None") and 0 <= int(float(v)) < len(sig):
                    tmp[f].append(sig[int(float(v))] - base)
        for f in AMP:
            if tmp[f]:
                our[ld][f].append(np.median(tmp[f]))

fi = {"Pamp":0,"Qamp":1,"Ramp":2,"Samp":3,"Tamp":4}
out = os.path.join(DATA, "amplitudes_alllead_sinus.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["feature", "mean_abs_diff_mV", "lead_sign_agreement", "n_records"])
    for f_ in AMP:
        diffs = []; agree = 0
        for ld in LEADS:
            om = np.mean(our[ld][f_]) if our[ld][f_] else np.nan
            t6 = T6A[ld][fi[f_]]
            if np.isfinite(om):
                diffs.append(abs(om - t6))
                if (om >= 0) == (t6 >= 0):
                    agree += 1
        w.writerow([f_, round(np.nanmean(diffs), 3), f"{agree}/12", len(want)])
        print(f"  {f_}: mean|Δ|={np.nanmean(diffs):.3f} mV  sign {agree}/12")
print("wrote", out)
