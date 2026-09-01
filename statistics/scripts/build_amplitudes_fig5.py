"""
build_amplitudes_fig5.py  -  Sinus lead-II amplitude features (P/Q/R/S/T) for the preprint
Figure 5 amplitude panels. Amplitude = signal at the fiducial peak minus the pre-QRS baseline.
Per-signal median over beats, then cohort mean, on a sample of recordings.

Reads ../ecgdeli_labelling/medalcare_fiducials_ecgdeli.csv , ../ecgdeli_labelling/medalcare_manifest.csv
Writes amplitudes_sinus_leadII.csv
"""
import pandas as pd, numpy as np, os, csv
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
FID  = os.path.join(ROOT,"ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
MAN  = os.path.join(ROOT,"ecgdeli_labelling","data","input","medalcare_manifest.csv")
LROW = {"II":1}; N = 200
AMP  = ["Pamp","Qamp","Ramp","Samp","Tamp"]
PK   = {"Pamp":"p_peak_sample","Qamp":"q_peak_sample","Ramp":"r_peak_sample",
        "Samp":"s_peak_sample","Tamp":"t_peak_sample"}
# Table 6 (sim) lead-II amplitude means [Pamp,Qamp,Ramp,Samp,Tamp] and preprint Fig 5 estimates
T6 = {"Pamp":0.09,"Qamp":0.06,"Ramp":0.59,"Samp":-0.20,"Tamp":0.49}
PRE= {"Pamp":0.10,"Qamp":0.05,"Ramp":0.70,"Samp":-0.20,"Tamp":0.45}

cols = ["disease_class","lead","record_id","qrs_onset_sample"]+list(PK.values())
df = pd.read_csv(FID, usecols=cols, na_values=["","None"])
d2 = df[(df.disease_class=="sinus")&(df.lead=="II")]
man = {r["record_id"]: r["path_raw"] for r in csv.DictReader(open(MAN)) if r["disease_class"]=="sinus"}
eligible = sorted(r for r in d2.record_id.unique() if r in man)
recs = np.random.default_rng(2026).choice(eligible, size=min(N, len(eligible)), replace=False).tolist()

amp_rec = {f:[] for f in AMP}
for rid, g in d2[d2.record_id.isin(recs)].groupby("record_id"):
    try: M = pd.read_csv(os.path.join(ROOT, man[rid]), header=None).to_numpy()
    except Exception: continue
    if M.shape[0] != 12: M = M.T
    sig = M[LROW["II"]]; tmp = {f:[] for f in AMP}
    for _,row in g.iterrows():
        qon = row["qrs_onset_sample"]
        if pd.isna(qon): continue
        qon = int(qon); base = np.median(sig[max(0,qon-11):qon+1])
        for f in AMP:
            v = row[PK[f]]
            if pd.notna(v) and 0 <= int(v) < len(sig): tmp[f].append(sig[int(v)]-base)
    for f in AMP:
        if tmp[f]: amp_rec[f].append(np.median(tmp[f]))

rows=[]
print(f"sinus lead II amplitudes (n={len(amp_rec['Ramp'])} records)")
print(f"{'feat':6s} {'our':>7s} {'Table6':>7s} {'Fig5~':>7s}")
for f in AMP:
    our = float(np.mean(amp_rec[f]))
    rows.append({"feature":f,"our_mV":round(our,3),"table6_mV":T6[f],"preprint_fig5_mV":PRE[f]})
    print(f"{f:6s} {our:7.3f} {T6[f]:7.3f} {PRE[f]:7.3f}")
pd.DataFrame(rows).to_csv(os.path.join(DATA,"amplitudes_sinus_leadII.csv"), index=False)
print("\nsaved amplitudes_sinus_leadII.csv")
