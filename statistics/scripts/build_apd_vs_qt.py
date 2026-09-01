"""
build_apd_vs_qt.py  -  Cross-check our ECGdeli QT against the TRUE repolarisation parameter
(ventricular APD) used by the simulation to generate each recording. If our delineation tracks
the physiology, longer APD should give longer QT. APD is the closest thing to timing ground
truth in the parameter files (which are simulation INPUTS, not output fiducials).

Reads /tmp/apd_raw.txt  (APD.min/max grepped per ventricular file)
        ../ecgdeli_labelling/medalcare_manifest.csv  (path_param_ventricular -> record_id)
        per_signal_median.csv                        (our per-record QT)
Writes apd_by_run.csv, apd_vs_qt.csv, apd_vs_qt.png
"""
import pandas as pd, numpy as np, os, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")

# 1) APD per parameter-file path  (each line is "pathAPD.min,val")
rows={}
for line in open("/tmp/apd_raw.txt"):
    left,val = line.strip().rsplit(",",1)
    path,key = left.rsplit(":",1)
    rel = "WP2_largeDataset_ParameterFiles/"+path
    rows.setdefault(rel,{})[key]=float(val)
apd = pd.DataFrame([{"path_param_ventricular":k,"APD_min":v.get("APD.min"),"APD_max":v.get("APD.max")}
                    for k,v in rows.items()])

# 2) map param path -> record_id via manifest
man = pd.DataFrame(csv.DictReader(open(os.path.join(ROOT,"ecgdeli_labelling","data","input","medalcare_manifest.csv"))))
apd = apd.merge(man[["record_id","disease_class","path_param_ventricular"]], on="path_param_ventricular", how="inner")
apd.to_csv(os.path.join(DATA,"apd_by_run.csv"), index=False)
print(f"APD joined to {len(apd)} records")

# 3) our per-record QT (lead II)
qt = pd.read_csv(os.path.join(DATA,"per_signal_median.csv"))
qt2 = qt[qt.lead=="II"][["record_id","disease_class","QTint"]].rename(columns={"QTint":"our_QT_leadII"})
m = apd.merge(qt2, on=["record_id","disease_class"], how="inner").dropna(subset=["APD_max","our_QT_leadII"])
m.to_csv(os.path.join(DATA,"apd_vs_qt.csv"), index=False)

r_max = np.corrcoef(m.APD_max, m.our_QT_leadII)[0,1]
r_min = np.corrcoef(m.APD_min, m.our_QT_leadII)[0,1]
print(f"records compared: {len(m)}")
print(f"Pearson r  (our QT vs APD.max) = {r_max:.3f}")
print(f"Pearson r  (our QT vs APD.min) = {r_min:.3f}")
print(f"APD.max mean {m.APD_max.mean():.1f} ms | our QT(II) mean {m.our_QT_leadII.mean():.1f} ms")
# healthy only
h = m[m.disease_class=="sinus"]
print(f"sinus: r(QT,APD.max)={np.corrcoef(h.APD_max,h.our_QT_leadII)[0,1]:.3f} (n={len(h)})")

# 4) scatter (sampled) with regression line
s = m.sample(min(4000,len(m)), random_state=0)
fig,ax=plt.subplots(figsize=(8,6))
ax.scatter(s.APD_max, s.our_QT_leadII, s=6, alpha=0.25, color="#4C78A8")
b,a = np.polyfit(m.APD_max, m.our_QT_leadII, 1)
xs=np.linspace(m.APD_max.min(), m.APD_max.max(),50)
ax.plot(xs, a+b*xs, color="#D62728", lw=2, label=f"fit: slope={b:.2f}, r={r_max:.2f}")
ax.set_xlabel("simulation APD.max (ms) — true repolarisation parameter")
ax.set_ylabel("our ECGdeli QT, lead II (ms)")
ax.set_title("Our labelled QT vs the simulation's true APD (all runs)\nPositive slope = delineation tracks the generative physiology")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(FIG,"apd_vs_qt.png"), dpi=140, bbox_inches="tight")
print("saved apd_by_run.csv, apd_vs_qt.csv, apd_vs_qt.png")
