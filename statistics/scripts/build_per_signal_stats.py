"""
build_per_signal_stats.py  -  Two-level population statistics for MedalCare-XL, compared
to the paper (published Table 6, exact) and the PREPRINT Figure 5 (lead II, estimated).

Level 1 (per-signal) collapse each recording's ~13 beats to ONE median per (record, lead)
                      for each interval  -> per_signal_median.csv
Level 2 (per-lead/class) mean, SD, n across recordings, per lead, per class
                      -> per_lead_class_summary.csv
Comparison sinus per-lead means vs Table 6 (exact) and, for lead II, vs preprint Fig 5
                      -> comparison_sinus_vs_paper.csv

Uses the ECGdeli interval columns already stored in the master CSV.
Run python3 build_per_signal_stats.py
"""
import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(ROOT,"statistics","data")
FID  = os.path.join(ROOT, "ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

# master-CSV interval columns -> our feature names
COLMAP = {"Pdur":"ecgdeli_pdur_ms","QRSdur":"ecgdeli_qrsdur_ms","Tdur":"ecgdeli_tdur_ms",
          "PQint":"ecgdeli_pq_ms","PRint":"ecgdeli_pr_ms","QTint":"qt_interval_ms",
          "QTpeak":"qt_peak_ms","RRint":"ecgdeli_rr_ms"}
FEATS = list(COLMAP)

# Published Table 6 'sim' means (lead-wise) [Pdur,QRSdur,Tdur,PQint,QTint,RRint]
T6 = {"I":[124.06,131.31,178.12,128.07,310.54,758.15],"II":[128.09,126.10,182.33,127.18,317.08,758.02],
 "III":[164.52,126.80,183.16,171.88,306.94,757.99],"aVR":[127.42,128.81,179.38,126.19,318.73,757.97],
 "aVL":[154.50,128.62,182.76,169.37,299.19,758.08],"aVF":[141.00,125.02,184.05,142.60,310.43,758.06],
 "V1":[140.78,129.05,180.72,160.57,303.89,758.06],"V2":[155.93,136.12,176.15,181.65,287.71,757.90],
 "V3":[154.23,132.80,179.01,174.20,285.50,758.01],"V4":[140.60,127.32,180.05,148.55,290.09,758.03],
 "V5":[128.69,123.89,177.39,126.73,310.74,758.12],"V6":[123.44,126.55,174.63,118.63,320.51,758.06]}
T6_ORDER = ["Pdur","QRSdur","Tdur","PQint","QTint","RRint"]

# Preprint Figure 5 (lead II) approximate peak (mode) read off the figure  [ms / mV]
# NOTE estimated by eye from the density peaks, the preprint has no raw table.
PRE_FIG5_II = {"Pdur":128,"QRSdur":125,"Tdur":185,"PRint":190,"QTint":385,"RRint":760}

# ---------- load only the columns we need ----------
usecols = ["disease_class","record_id","lead"] + list(COLMAP.values())
df = pd.read_csv(FID, usecols=usecols, na_values=["","None"],
                 dtype={c:"float32" for c in COLMAP.values()})
df = df.rename(columns={v:k for k,v in COLMAP.items()})

# ---------- LEVEL 1 per (class, record, lead) median over beats ----------
lvl1 = df.groupby(["disease_class","record_id","lead"], observed=True)[FEATS].median().reset_index()
lvl1.to_csv(os.path.join(DATA,"per_signal_median.csv"), index=False)
print(f"Level 1: per_signal_median.csv  -> {len(lvl1):,} rows (record x lead)")

# ---------- LEVEL 2 per (class, lead) mean/SD/n across records ----------
agg = lvl1.groupby(["disease_class","lead"], observed=True)[FEATS].agg(["mean","std","count"])
agg.columns = [f"{f}_{s}" for f,s in agg.columns]; agg = agg.reset_index()
agg.to_csv(os.path.join(DATA,"per_lead_class_summary.csv"), index=False)
print(f"Level 2: per_lead_class_summary.csv -> {len(agg)} rows (class x lead)")

# ---------- COMPARISON sinus per-lead means vs Table 6 (+ Fig5 for lead II) ----------
sin = lvl1[lvl1.disease_class=="sinus"]
rows=[]
for ld in LEADS:
    s = sin[sin.lead==ld]
    row = {"lead":ld}
    for i,f in enumerate(T6_ORDER):
        our = s[f].mean(); row[f"{f}_our"]=round(our,1); row[f"{f}_T6"]=T6[ld][i]
        row[f"{f}_dT6"]=round(our-T6[ld][i],1)
    rows.append(row)
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(DATA,"comparison_sinus_vs_table6.csv"), index=False)

print("\n=== sinus per-lead: our mean  vs Table 6  (Δ) ===")
print(f"{'lead':4s}"+ "".join(f"{f:>22s}" for f in T6_ORDER))
for _,r in comp.iterrows():
    print(f"{r['lead']:4s}"+ "".join(f"  {r[f+'_our']:6.1f}/{r[f+'_T6']:6.1f}({r[f+'_dT6']:+5.1f})" for f in T6_ORDER))

print("\n=== lead II: our mean vs Table 6 vs preprint Fig 5 (estimated) ===")
s2 = sin[sin.lead=="II"]
print(f"{'feature':8s} {'our':>8s} {'Table6':>8s} {'Fig5~':>8s}")
for f in ["Pdur","QRSdur","Tdur","PQint","PRint","QTint","QTpeak","RRint"]:
    t6 = T6["II"][T6_ORDER.index(f)] if f in T6_ORDER else None
    fg = PRE_FIG5_II.get(f)
    print(f"{f:8s} {s2[f].mean():8.1f} {('%.1f'%t6) if t6 else '   -':>8s} {(str(fg)) if fg else '   -':>8s}")
print("\nKEY: QTint our ~%.0f matches preprint Fig 5 (~385), NOT Table 6 (317)." % s2.QTint.mean())
