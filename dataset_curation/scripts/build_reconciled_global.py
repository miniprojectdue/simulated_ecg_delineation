#!/usr/bin/env python3
"""
build_reconciled_global.py  -  Collapse the per-lead ECGdeli labels into a single
GLOBAL delineation per signal, and attach a quality-control status so signals that
need manual fixing are easy to track.

WHY GLOBAL
    We train a 12-lead-in / one-delineation-out model (see the methodology). ECGdeli
    labels each lead separately, so this script reconciles the 12 leads of each beat
    into one global fiducial set using the standard clinical convention
        R-peak = MASTER, from a single reference channel (lead II) [definitive beat anchor]
        onset  = earliest across leads (min)     [wave begins when it first appears]
        offset = latest across leads   (max)     [wave ends when it last disappears]
        P/Q/S/T peak = median across leads       [robust central estimate]
    Beat counts are identical across all 12 leads for every record (verified 16,848/16,848),
    so beat_id already aligns the same cardiac cycle, the R-peak fiducial itself legitimately
    lands at slightly different samples per lead (median ~32 ms spread) because the QRS
    projects with a different shape onto each lead, so we anchor the beat to the master
    reference-channel R-peak rather than to twelve independent detections.

    Since each MedalCare-XL record is one simulated beat repeated ~13x, we output
    ONE ROW PER SIGNAL using a representative (median) beat, beat_start/end give its
    window, and path_raw points at the 12x5000 waveform.

QC STATUS  (carried in the file so signals with issues are trackable)
    critical -> GENUINE problem at least one lead has >=50% of its beats flagged
                critical. These are the units to correct manually (~4,251 units).
    minor    -> has some critical beats but no genuine lead (transient glitches on
                otherwise-clean repeated beats, a per-record consensus absorbs them).
    clean    -> no critical beats.

Run python3 build_reconciled_global.py
Out reconciled_global_fiducials.csv   (one row per signal, global fiducials + QC)
"""
import pandas as pd, numpy as np, csv, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
FID  = os.path.join(ROOT, "ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
QC   = os.path.join(ROOT, "ecgdeli_labelling","data","qc","medalcare_qc_review_list.csv")
MAN  = os.path.join(ROOT, "ecgdeli_labelling","data","input","medalcare_manifest.csv")
OUT  = os.path.join(ROOT, "dataset_curation","data","global","reconciled_global_fiducials.csv")

ONS  = ["p_onset_sample", "qrs_onset_sample", "t_onset_sample"]
OFFS = ["p_offset_sample", "qrs_offset_sample", "t_offset_sample"]
PKS  = ["p_peak_sample", "q_peak_sample", "r_peak_sample", "s_peak_sample", "t_peak_sample"]
PRES = ["p_present", "qrs_present", "t_present"]
GLOBAL_FIDS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
               "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]

# ---- 1. load per-lead labels + flag critical beats ----
use = ["record_id","lead","beat_id","disease_class","mi_subclass","split","fs_hz","n_samples",
       "beat_start_sample","beat_end_sample"] + ONS+OFFS+PKS + PRES
f = pd.read_csv(FID, usecols=use, na_values=["","None"])
qc = pd.read_csv(QC, usecols=["record_id","lead","beat_id"]); qc["is_crit"] = 1
f = f.merge(qc, on=["record_id","lead","beat_id"], how="left")
f["is_crit"] = f["is_crit"].fillna(0).astype(int)

# ---- 2. genuine-problem units (record,lead) with >=50% of beats critical ----
u = f.groupby(["record_id","lead"]).agg(nb=("beat_id","size"), nc=("is_crit","sum")).reset_index()
u["genuine"] = u["nc"]/u["nb"] >= 0.5
gen = u[u["genuine"]]
problem_leads = gen.groupby("record_id")["lead"].apply(lambda s: ";".join(sorted(s))).rename("problem_leads")
rec_genuine  = set(gen["record_id"])
rec_anycrit  = set(f.loc[f["is_crit"] > 0, "record_id"])

# ---- 3. reconcile every beat across the 12 leads -> global fiducials ----
# Beat identity is anchored to a MASTER R-peak taken from a single reference channel
# (lead II, the rhythm lead), rather than reconciling twelve independently-detected
# R-peaks. Beat counts are identical across all 12 leads for every record (verified
# 16,848/16,848), so beat_id already aligns the same cardiac cycle, taking the R-peak
# from the master channel makes the anchor definitive and independent of per-lead
# detection. Onsets are the earliest across leads, offsets the latest (the global
# convention that recovers a wave from whichever lead shows it above the isoelectric
# line), the secondary P/Q/S/T peaks are the robust median across leads.
REF_LEAD = "II"
agg = {c:"min" for c in ONS}
agg.update({c:"max" for c in OFFS})
agg.update({c:"median" for c in PKS})          # p/q/(r)/s/t peaks, r_peak overridden below
agg.update({c:"max" for c in PRES})
agg["beat_start_sample"] = "min"; agg["beat_end_sample"] = "max"
recon = f.groupby(["record_id","beat_id"]).agg(agg).reset_index()
# override the global R-peak with the master reference-channel R-peak
masterR = (f.loc[f["lead"]==REF_LEAD, ["record_id","beat_id","r_peak_sample"]]
             .rename(columns={"r_peak_sample":"r_master"}))
recon = recon.merge(masterR, on=["record_id","beat_id"], how="left")
recon["r_peak_sample"] = recon["r_master"].fillna(recon["r_peak_sample"])
recon = recon.drop(columns=["r_master"])
for c in GLOBAL_FIDS+["beat_start_sample","beat_end_sample"]:
    recon[c] = recon[c].round()

# ---- 4. one row per signal the representative (median beat_id) beat ----
med = f.groupby("record_id")["beat_id"].median().rename("medb").reset_index()
recon = recon.merge(med, on="record_id")
recon["d"] = (recon["beat_id"] - recon["medb"]).abs()
rep = recon.sort_values("d").groupby("record_id", as_index=False).first().drop(columns=["d","medb"])
rep = rep.rename(columns={"beat_id":"rep_beat_id"})

# ---- 5. attach metadata, QC status, priority, path ----
meta = f.drop_duplicates("record_id").set_index("record_id")[["disease_class","mi_subclass","split","fs_hz","n_samples"]]
rep = rep.merge(meta, on="record_id").merge(problem_leads, on="record_id", how="left")
rep["problem_leads"] = rep["problem_leads"].fillna("")
rep["n_problem_leads"] = rep["problem_leads"].apply(lambda s: 0 if s=="" else s.count(";")+1)
rep["genuine"] = rep["record_id"].isin(rec_genuine).astype(int)
def status(r):
    if r.record_id in rec_genuine: return "critical"
    if r.record_id in rec_anycrit: return "minor"
    return "clean"
rep["qc_status"] = rep.apply(status, axis=1)
man = {r["record_id"]: r["path_raw"] for r in csv.DictReader(open(MAN))}
rep["path_raw"] = rep["record_id"].map(man)
cw = {"lbbb":3,"mi":3,"rbbb":3}
rep["review_priority"] = rep.apply(lambda r: (cw.get(r.disease_class,1) if r.qc_status=="critical" else 0)
                                             + r.n_problem_leads, axis=1)

COLS = (["record_id","disease_class","mi_subclass","split","fs_hz","n_samples","path_raw",
         "rep_beat_id","beat_start_sample","beat_end_sample"] + GLOBAL_FIDS + PRES +
        ["qc_status","genuine","n_problem_leads","problem_leads","review_priority"])
rep = rep[COLS].sort_values(["qc_status","review_priority"], ascending=[True,False])
rep.to_csv(OUT, index=False)

# ---- report ----
n = len(rep); vc = rep["qc_status"].value_counts()
print(f"wrote {OUT}")
print(f"signals (rows): {n}")
for s in ["critical","minor","clean"]:
    print(f"  {s:8s}: {vc.get(s,0)}  ({100*vc.get(s,0)/n:.1f}%)")
print("critical (genuine) by class:")
print(rep[rep.qc_status=='critical'].disease_class.value_counts().to_string())

def load_signal(record_id, dataset_root=ROOT):
    """Load the 12x5000 waveform for a row's record_id (leads x samples)."""
    p = pd.read_csv(reconciled_path(record_id), header=None).to_numpy()
    return p if p.shape[0]==12 else p.T
def reconciled_path(record_id):
    return os.path.join(ROOT, man[record_id])
