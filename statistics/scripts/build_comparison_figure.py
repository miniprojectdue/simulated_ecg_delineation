"""
build_comparison_figure.py  -  Lead-II timing densities (our per-signal medians) with
markers for our mean, published Table 6, and the preprint Figure 5 estimate. Shows that
QT matches the preprint (~385) and not Table 6 (317), while everything else agrees with both.

Reads per_signal_median.csv
Writes leadII_vs_paper_preprint.png
"""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); FIG = os.path.join(HERE, "..", "figures")
df = pd.read_csv(os.path.join(DATA,"per_signal_median.csv"))
s = df[(df.disease_class=="sinus") & (df.lead=="II")]

# feature (our col, Table6 II value, preprint Fig5 II estimate, title, unit)
PANELS = [("Pdur",128.09,128,"P duration"),("QRSdur",126.10,125,"QRS duration"),
          ("Tdur",182.33,185,"T duration"),("PRint",None,190,"PR interval (P–R peak)"),
          ("QTint",317.08,385,"QT interval"),("RRint",758.02,760,"RR interval")]

def kde(d, xs):
    d = np.asarray(d,float); d=d[np.isfinite(d)]; sd=d.std()
    h=1.06*sd*len(d)**-0.2 if sd>0 else 1.0
    u=(xs[:,None]-d[None,:])/h
    return np.exp(-0.5*u*u).sum(1)/(len(d)*h*np.sqrt(2*np.pi))

fig, ax = plt.subplots(2,3, figsize=(14,7))
for a,(f,t6,fg,title) in zip(ax.ravel(), PANELS):
    v = s[f].dropna().values
    lo,hi = np.quantile(v,0.005), np.quantile(v,0.995)
    lo=min(lo, (t6 or lo)-10, fg-10); hi=max(hi,(t6 or hi)+10, fg+10)
    xs=np.linspace(lo,hi,300)
    a.fill_between(xs, kde(v,xs), color="#4C78A8", alpha=0.45, label="Our labels (density)")
    a.axvline(np.nanmean(v), color="#1F4E79", lw=2, label=f"Our mean ({np.nanmean(v):.0f})")
    a.axvline(fg, color="#2CA02C", lw=2, ls="--", label=f"Preprint Fig 5 (~{fg})")
    if t6 is not None:
        a.axvline(t6, color="#D62728", lw=2, ls=":", label=f"Table 6 ({t6:.0f})")
    a.set_title(title, fontsize=10); a.set_xlabel("ms"); a.set_yticks([])
    a.legend(fontsize=7)
fig.suptitle("Healthy sinus, lead II (one median value per recording): our labels vs preprint Figure 5 vs Table 6\n"
             "The directly comparable features agree with both; only QT separates — ours matches the preprint (~385), "
             "not Table 6 (317). The PR panel uses the preprint's P–R-peak convention (Table 6 reports the different PQ interval).",
             fontsize=9)
plt.tight_layout()
out=os.path.join(FIG,"leadII_vs_paper_preprint.png")
plt.savefig(out, dpi=140, bbox_inches="tight"); print("saved", out)
