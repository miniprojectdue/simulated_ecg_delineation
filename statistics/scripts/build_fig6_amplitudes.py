"""
build_fig6_amplitudes.py  -  Per-class amplitude contrasts for Figure 6 (healthy sinus vs disease,
in the disease-relevant lead). amp = signal at the fiducial peak minus the pre-QRS baseline
(median of the 12 samples ending at QRS onset), per-signal median over beats, then cohort mean.

Sampling a REPRODUCIBLE RANDOM sample of N per disease class, drawn without replacement from the
sorted list of eligible records with a fixed seed (one sample per class, applied independently).

Reads dataset_curation/data/assembled/master_labels.csv (peak positions), raw signals.
Writes statistics/data/fig6_amplitudes_by_class.csv
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
N = 100
SEED = 2026

LROW = {"I":0,"II":1,"III":2,"aVR":3,"aVL":4,"aVF":5,"V1":6,"V2":7,"V3":8,"V4":9,"V5":10,"V6":11}
PKCOL = {"Pamp":"p_peak_sample","Qamp":"q_peak_sample","Ramp":"r_peak_sample"}
# panels Figure 6 shows (feature, lead, class)  (healthy = sinus in the same lead)
PANELS = [("Qamp","II","mi"), ("Ramp","V2","mi"), ("Ramp","II","lae"),
          ("Pamp","aVL","fam"), ("Pamp","V6","fam")]
CLASSES = sorted({"sinus"} | {c for _,_,c in PANELS})
LEADS   = sorted({ld for _,ld,_ in PANELS})
FEATS   = sorted({f for f,_,_ in PANELS})

# --- one reproducible random sample of N records per class ---
idx = {r["record_id"]: r for r in csv.DictReader(open(SIGIDX, newline=""))}
sample = {}
for cls in CLASSES:
    elig = sorted(r for r, v in idx.items() if v["disease_class"] == cls)
    sample[cls] = set(np.random.default_rng(SEED).choice(elig, size=min(N, len(elig)), replace=False).tolist())
want = set().union(*sample.values())

# --- collect peak positions for the sampled records in the needed leads ---
usecols = ["record_id","disease_class","lead","qrs_onset_sample"] + [PKCOL[f] for f in FEATS]
beats = {}   # (rid, lead) -> rows
for ch in pd.read_csv(MASTER, usecols=usecols, dtype=str, na_filter=False, chunksize=400000):
    ch = ch[ch.record_id.isin(want) & ch.lead.isin(LEADS)]
    for (rid, ld), g in ch.groupby(["record_id", "lead"]):
        beats.setdefault((rid, ld), []).extend(g.to_dict("records"))

# --- per (record, lead, feature) median amplitude ---
CACHE = {}
def sig_of(rid):
    if rid in CACHE: return CACHE[rid]
    try:
        M = pd.read_csv(idx[rid]["path_raw"] if os.path.isabs(idx[rid]["path_raw"])
                        else os.path.join(ROOT, idx[rid]["path_raw"]), header=None).to_numpy()
        if M.shape[0] != 12: M = M.T
    except Exception:
        M = None
    CACHE[rid] = M; return M

def cohort_mean(cls, lead, feat):
    col = PKCOL[feat]; per_rec = []
    for rid in sample[cls]:
        M = sig_of(rid)
        if M is None: continue
        sig = M[LROW[lead]]; vals = []
        for row in beats.get((rid, lead), []):
            qon, pk = row["qrs_onset_sample"], row[col]
            if qon in ("","None") or pk in ("","None"): continue
            qon = int(float(qon)); base = np.median(sig[max(0, qon-11):qon+1]); pk = int(float(pk))
            if 0 <= pk < len(sig): vals.append(sig[pk] - base)
        if vals: per_rec.append(np.median(vals))
    return (float(np.mean(per_rec)) if per_rec else np.nan), len(per_rec)

rows = []
print(f"{'feat':5s} {'lead':4s} {'class':5s} {'healthy':>8s} {'disease':>8s} {'shift':>7s}")
for feat, lead, cls in PANELS:
    h, _ = cohort_mean("sinus", lead, feat)
    dz, nd = cohort_mean(cls, lead, feat)
    rows.append({"feature": feat, "lead": lead, "class": cls,
                 "healthy_mV": round(h, 3), "disease_mV": round(dz, 3),
                 "shift_mV": round(dz - h, 3), "n": nd})
    print(f"{feat:5s} {lead:4s} {cls:5s} {h:8.3f} {dz:8.3f} {dz-h:+7.3f}")
pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig6_amplitudes_by_class.csv"), index=False)
print("\nsaved fig6_amplitudes_by_class.csv")
