"""
build_gaussian_overlays.py  -  Healthy-sinus lead-II density overlays (preprint Figure 5 style),
showing BOTH references on every panel the published Table 6 Gaussian (dashed red) and the
preprint Figure 5 reference (solid green).

The preprint publishes only probability-density plots (no numeric mean/SD table), so the preprint
CENTRES are read from Figure 5 and its WIDTHS reuse the Table 6 SD (the closest available spread)
for the PR panel, which Table 6 does not report, the width uses our own observed SD. Only QT differs
materially between the two references, and our density sits on the preprint centre (~385), not Table 6 (~317).

Timing densities come from per_signal_median.csv (one median value per record), amplitudes are
recomputed from the raw signals for a sample of sinus records.

Writes fig_repro_timing_leadII.png, fig_repro_amp_leadII.png  (statistics/figures + reports/figures)
"""
import os, csv
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
REPFIG = os.path.join(ROOT, "reports", "figures")
MASTER = os.path.join(ROOT, "dataset_curation", "data", "assembled", "master_labels.csv")
SIGIDX = os.path.join(ROOT, "dataset_curation", "data", "review", "signals_index.csv")
N_AMP = 200                       # sinus records sampled for amplitude densities

# ---- lead-II references (mean, sd) ------------------------------------------------------------
# NOTE on the AV interval the preprint's Figure 5 plots PRint (P onset -> R peak, ~190 ms) while
# the published Table 6 reports PQint (P onset -> QRS onset, ~127 ms) — different measurements. This
# panel matches the PREPRINT convention, so it shows our PRint vs the preprint only, the Table 6 PQ
# comparison (our PQint 124 vs Table 6 127) is in the Section 4.4 table. Table 6 has no PRint entry.
T6 = {"Pdur": (128.09, 14.00), "QRSdur": (126.10, 13.73), "Tdur": (182.33, 25.94),
      "PRint": None,           "QTint": (317.08, 23.61), "RRint": (758.02, 54.41)}
# Preprint Figure 5 (simulated/red curve) centres read from the figure, widths reuse Table 6 SD
# (our observed SD for PRint, which Table 6 lacks). Of the directly comparable features only QT departs.
PP = {"Pdur": (128, 14.00), "QRSdur": (125, 13.73), "Tdur": (185, 25.94),
      "PRint": (190, 24.6),  "QTint": (385, 23.61), "RRint": (760, 54.41)}
TIMING = ["Pdur", "QRSdur", "Tdur", "PRint", "QTint", "RRint"]
TTITLE = ["P wave duration", "QRS duration", "T wave duration", "PR interval (P–R peak)", "QT interval", "RR interval"]

# amplitudes (mean), widths use a nominal SD proxy since neither reference publishes lead-II amp SD
T6A = {"Pamp": 0.09, "Qamp": 0.06, "Ramp": 0.59, "Samp": -0.20, "Tamp": 0.49}
PPA = {"Pamp": 0.10, "Qamp": 0.05, "Ramp": 0.70, "Samp": -0.20, "Tamp": 0.45}
AMP = ["Pamp", "Qamp", "Ramp", "Samp", "Tamp"]
ATITLE = ["P amplitude", "Q amplitude", "R amplitude", "S amplitude", "T amplitude"]
PK = {"Pamp": "p_peak_sample", "Qamp": "q_peak_sample", "Ramp": "r_peak_sample",
      "Samp": "s_peak_sample", "Tamp": "t_peak_sample"}

def kde(data, xs):
    d = np.asarray(data, float); d = d[np.isfinite(d)]; sd = d.std()
    h = 1.06 * sd * len(d) ** (-0.2) if sd > 0 else 1.0
    u = (xs[:, None] - d[None, :]) / h
    return np.exp(-0.5 * u * u).sum(1) / (len(d) * h * np.sqrt(2 * np.pi))

def gauss(xs, mu, sd):
    return np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

def panel(a, v, t6, pp, title, unit, our_sd_for_pp=None):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    centres = [m for m in [v.mean(), t6[0] if t6 else None, pp[0] if pp else None] if m is not None]
    sds = [s for s in [v.std(), t6[1] if t6 else None, (pp[1] if pp else None)] if s is not None]
    lo = min([v.min()] + [c - 3.5 * max(sds) for c in centres])
    hi = max([np.quantile(v, 0.99)] + [c + 3.5 * max(sds) for c in centres])
    xs = np.linspace(lo, hi, 400)
    a.fill_between(xs, kde(v, xs), color="#4C78A8", alpha=0.45, label="Our labels")
    a.axvline(v.mean(), color="#4C78A8", ls="--", lw=1)
    if t6 is not None:
        a.plot(xs, gauss(xs, *t6), color="#D62728", lw=1.8, ls="--", label="Published Table 6")
    if pp is not None:
        sd_pp = pp[1] if pp[1] is not None else (our_sd_for_pp or v.std())
        a.plot(xs, gauss(xs, pp[0], sd_pp), color="#2CA02C", lw=2.0, label="Preprint Fig 5")
    a.set_title(title, fontsize=10); a.set_xlabel(unit); a.tick_params(axis='y', labelsize=7); a.legend(fontsize=7)

# ================= TIMING (from per_signal_median.csv) =========================================
psm = pd.read_csv(os.path.join(DATA, "per_signal_median.csv"))
s = psm[(psm.disease_class == "sinus") & (psm.lead == "II")]
fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for i, (a, f, ti) in enumerate(zip(ax.ravel(), TIMING, TTITLE)):
    panel(a, s[f].values, T6[f], PP[f], ti, "ms", our_sd_for_pp=s[f].std())
    if i % 3 == 0:
        a.set_ylabel("Probability density")
fig.suptitle("Healthy sinus, lead II (one median per record): our density vs published Table 6 (dashed) "
             "and preprint Figure 5 (solid).\nOf the directly comparable features only QT separates the two "
             "references — our density sits on the preprint (~385 ms), not Table 6 (~317 ms). The AV panel "
             "shows PRint (P–R peak) vs the preprint; Table 6's PQ interval is a different measure (see §4.4).",
             fontsize=10)
plt.tight_layout()
for d in (FIG, REPFIG):
    plt.savefig(os.path.join(d, "fig_repro_timing_leadII.png"), dpi=140, bbox_inches="tight")
plt.close(); print("saved fig_repro_timing_leadII.png")

# ================= AMPLITUDES (recomputed from signals) ========================================
idx = {r["record_id"]: r for r in csv.DictReader(open(SIGIDX, newline=""))}
# reproducible random sample of sinus records (fixed seed 2026, sorted eligible list) — matches
# build_amplitudes_alllead.py so the figure and Table 8 use the same recordings.
elig = sorted(r for r, v in idx.items() if v["disease_class"] == "sinus")
want = np.random.default_rng(2026).choice(elig, size=min(N_AMP, len(elig)), replace=False).tolist()
wantset = set(want); beats = {}
for ch in pd.read_csv(MASTER, usecols=["record_id", "disease_class", "lead", "qrs_onset_sample"] + list(PK.values()),
                      dtype=str, na_filter=False, chunksize=400000):
    ch = ch[(ch.disease_class == "sinus") & (ch.lead == "II") & (ch.record_id.isin(wantset))]
    for rid, g in ch.groupby("record_id"):
        beats.setdefault(rid, []).extend(g.to_dict("records"))
LROW = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}
amp_rec = {f: [] for f in AMP}
for rid in want:
    path = os.path.join(ROOT, idx[rid]["path_raw"])
    try:
        M = pd.read_csv(path, header=None).to_numpy()
    except Exception:
        continue
    if M.shape[0] != 12:
        M = M.T
    sig = M[LROW["II"]]
    tmp = {f: [] for f in AMP}
    for row in beats[rid]:
        qon = row["qrs_onset_sample"]
        if qon in ("", "None"):
            continue
        qon = int(float(qon)); base = np.median(sig[max(0, qon - 11):qon + 1])
        for f in AMP:
            v = row[PK[f]]
            if v not in ("", "None") and 0 <= int(float(v)) < len(sig):
                tmp[f].append(sig[int(float(v))] - base)
    for f in AMP:
        if tmp[f]:
            amp_rec[f].append(np.median(tmp[f]))
print("amplitude records used:", len(amp_rec["Ramp"]))

fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for i, (a, f, ti) in enumerate(zip(ax.ravel(), AMP, ATITLE)):
    v = np.asarray(amp_rec[f], float); v = v[np.isfinite(v)]
    sd_proxy = max(v.std(), 0.02)
    panel(a, v, (T6A[f], sd_proxy), (PPA[f], sd_proxy), ti, "mV")
    if i % 3 == 0:
        a.set_ylabel("Probability density")
ax.ravel()[5].axis("off")
fig.suptitle("Healthy sinus, lead II amplitudes (one median per record): our density vs published Table 6 "
             "(dashed) and preprint Figure 5 (solid).\nThe two references agree on amplitudes and both overlap "
             "our densities; the P wave runs slightly low, while Q, R, S and T track the references.", fontsize=11)
plt.tight_layout()
for d in (FIG, REPFIG):
    plt.savefig(os.path.join(d, "fig_repro_amp_leadII.png"), dpi=140, bbox_inches="tight")
plt.close(); print("saved fig_repro_amp_leadII.png")
