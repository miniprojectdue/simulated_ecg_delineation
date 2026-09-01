import os, sys
import pandas as pd, numpy as np

SEED       = 20260725
PER_CLASS  = 100          # 50 calibration + 50 test
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)

RECON  = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "dataset_curation", "data", "global",
                                                            "reconciled_global_fiducials_corrected.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "Gold", "data")
UNITS  = os.path.join(ROOT, "manual_labelling", "data", "final_data_units.csv")
CORR   = [os.path.join(ROOT, "manual_labelling", "corrections", "final_data_units_corrections.csv"),
          os.path.join(ROOT, "manual_labelling", "corrections", "all_units_worklist_corrections.csv")]

FKEYS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
         "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
BOUNDS = ["p_onset_sample","p_offset_sample","qrs_onset_sample","qrs_offset_sample",
          "t_onset_sample","t_offset_sample"]
# exactly the columns medalcare_label_ecg.m / gold_label_ecg.m require, in order
REQ = (["record_id","disease_class","lead","beat_id","fs_hz","n_samples"] + FKEYS +
       ["p_present","qrs_present","t_present","beat_start_sample","beat_end_sample",
        "flags","also_delineator","priority","path_raw"])
EXTRA = ["gold_split","qc_status","n_recon_leads","ref_lead_mad_ms","p_dur_ms","pq_seg_ms","qt_ms"]

def main():
    g = pd.read_csv(RECON, low_memory=False)
    print(f"reconciled records in: {len(g)}")

    # ---- hold out anything already reviewed under the old per-lead protocol ----
    seen = set()
    for c in CORR:
        if os.path.isfile(c):
            seen |= set(pd.read_csv(c, usecols=["record_id"])["record_id"])
    before = len(g)
    g = g[~g["record_id"].isin(seen)].copy()
    print(f"held out {before - len(g)} records already reviewed per-lead; {len(g)} eligible")

    # ---- stratified sample, PER_CLASS per disease class ----
    rng = np.random.RandomState(SEED)
    picks = []
    for cls, d in g.groupby("disease_class", sort=True):
        n = min(PER_CLASS, len(d))
        if n < PER_CLASS:
            print(f"  WARNING: class {cls} has only {len(d)} eligible records, taking all")
        picks.append(d.iloc[rng.choice(len(d), size=n, replace=False)])
    S = pd.concat(picks, ignore_index=True)

    # alternate calibration/test within each class so the halves match on everything
    S = S.sort_values(["disease_class", "record_id"]).reset_index(drop=True)
    S["gold_split"] = ["calibration" if i % 2 == 0 else "test"
                       for cls, d in S.groupby("disease_class", sort=True) for i in range(len(d))]

    # ---- reference lead: the lead whose own boundaries sit closest to the reconciled label ----
    # R-PEAK-ALIGNED comparison. final_data_units.csv holds one representative beat per
    # record-lead, chosen per lead, and that beat coincides with the reconciled record-level
    # representative beat only 37.7 per cent of the time. Comparing raw sample positions
    # therefore differs by a whole RR interval on the other 62.3 per cent and produces a
    # spurious ~750 ms disagreement. Every boundary is expressed relative to its own row's
    # R peak first, which removes the beat offset and leaves the morphology difference that
    # the reference-lead choice is actually about.
    u = pd.read_csv(UNITS, usecols=["record_id","lead","r_peak_sample"] + BOUNDS)
    u = u[u["record_id"].isin(set(S["record_id"]))].copy()
    ur = u[BOUNDS].sub(u["r_peak_sample"], axis=0)
    gr = S.set_index("record_id")[BOUNDS].sub(S.set_index("record_id")["r_peak_sample"], axis=0)
    ur.index = u["record_id"].values
    dev = (ur - gr).abs().mean(axis=1) * 2.0                               # ms at 500 Hz
    u = u.assign(mad_ms=dev.values).sort_values(["record_id","mad_ms"])
    best = u.groupby("record_id", as_index=False).first()[["record_id","lead","mad_ms"]]
    S = S.merge(best, on="record_id", how="left")
    S["lead"] = S["lead"].fillna("II")
    S["ref_lead_mad_ms"] = S["mad_ms"].round(1)

    # ---- tool columns ----
    S["beat_id"]          = S["rep_beat_id"]
    S["also_delineator"]  = 0
    S["priority"]         = 1
    S["p_dur_ms"]  = ((S["p_offset_sample"]   - S["p_onset_sample"])   * 2).round(0)
    S["pq_seg_ms"] = ((S["qrs_onset_sample"]  - S["p_offset_sample"])  * 2).round(0)
    S["qt_ms"]     = ((S["t_offset_sample"]   - S["qrs_onset_sample"]) * 2).round(0)
    S["flags"] = ("GOLD " + S["gold_split"].str[:5] + " | " + S["n_recon_leads"].astype(int).astype(str)
                  + " leads | P " + S["p_dur_ms"].astype(int).astype(str)
                  + "ms PQ " + S["pq_seg_ms"].astype(int).astype(str)
                  + "ms QT " + S["qt_ms"].astype(int).astype(str) + "ms")

    miss = [c for c in REQ if c not in S.columns]
    if miss:
        sys.exit("BUG: worklist is missing tool-required columns: " + ", ".join(miss))
    out = S[REQ + [c for c in EXTRA if c in S.columns]].copy()
    for c in FKEYS + ["beat_id","beat_start_sample","beat_end_sample","n_samples","fs_hz"]:
        out[c] = out[c].round().astype("Int64")

    os.makedirs(OUTDIR, exist_ok=True)
    cal = out[out["gold_split"] == "calibration"].sort_values(["disease_class","record_id"])
    tst = out[out["gold_split"] == "test"].sort_values(["disease_class","record_id"])
    assert set(cal["record_id"]).isdisjoint(set(tst["record_id"])), "calibration and test overlap"
    assert not out["record_id"].duplicated().any(), "duplicate record in gold set"
    cal.to_csv(os.path.join(OUTDIR, "gold_worklist_calibration.csv"), index=False)
    tst.to_csv(os.path.join(OUTDIR, "gold_worklist_test.csv"), index=False)

    print(f"\nwrote {len(cal)} calibration + {len(tst)} test records -> {os.path.relpath(OUTDIR, ROOT)}")
    print(pd.crosstab(out["disease_class"], out["gold_split"]).to_string())
    print(f"\nreference-lead agreement with the reconciled label (ms): "
          f"median {out['ref_lead_mad_ms'].median():.1f}, p90 {out['ref_lead_mad_ms'].quantile(.9):.1f}")
    print("lead chosen as reference:", out["lead"].value_counts().to_dict())
    print("qc_status mix:", out["qc_status"].value_counts().to_dict())

if __name__ == "__main__":
    main()
