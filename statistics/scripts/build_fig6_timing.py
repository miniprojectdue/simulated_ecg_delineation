"""
build_fig6_timing.py  -  Per-class (healthy vs disease) comparison for the TIMING features
the preprint Figure 6 highlights, in the disease-relevant leads. Uses the Level-1
per-signal medians (no signal reads needed).

Fig 6 timing panels reproduced
  RBBB / LBBB  -> QRS duration (lead II)
  AV block     -> PR interval  (lead II)
  IAB / LAE    -> P duration   (lead II)
  FAM          -> P duration   (lead V6)
  MI           -> QT interval  (lead V4)

Reads per_signal_median.csv
Writes fig6_timing_by_class.csv
"""
import pandas as pd, numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
df = pd.read_csv(os.path.join(DATA,"per_signal_median.csv"))

# feature -> Level-1 column
FCOL = {"Pdur":"Pdur","QRSdur":"QRSdur","PRint":"PRint","QTint":"QTint"}
# (disease class, feature, lead, Fig-6 expected direction vs healthy)
PANELS = [
 ("rbbb","QRSdur","II","up (wider QRS)"),
 ("lbbb","QRSdur","II","up (wider QRS)"),
 ("avblock","PRint","II","up (prolonged PR)"),
 ("iab","Pdur","II","up (wider P)"),
 ("lae","Pdur","II","up (wider P)"),
 ("fam","Pdur","V6","up (wider/abnormal P)"),
 ("mi","QTint","V4","up (prolonged QT)"),
]
sin = df[df.disease_class=="sinus"]
rows=[]
print(f"{'class':8s} {'feat':6s} {'lead':4s} {'healthy':>8s} {'disease':>8s} {'shift':>7s}   Fig6 expects")
for cls,feat,lead,exp in PANELS:
    h = sin[sin.lead==lead][FCOL[feat]].mean()
    d = df[(df.disease_class==cls)&(df.lead==lead)][FCOL[feat]].mean()
    rows.append({"class":cls,"feature":feat,"lead":lead,
                 "healthy_mean":round(h,1),"disease_mean":round(d,1),"shift":round(d-h,1),
                 "fig6_expected":exp})
    print(f"{cls:8s} {feat:6s} {lead:4s} {h:8.1f} {d:8.1f} {d-h:+7.1f}   {exp}")
pd.DataFrame(rows).to_csv(os.path.join(DATA,"fig6_timing_by_class.csv"), index=False)
print("\nsaved fig6_timing_by_class.csv")
