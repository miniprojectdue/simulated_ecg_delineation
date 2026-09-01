import os, sys
import pandas as pd, numpy as np

SEED      = 20260725
PER_SPLIT = 50            # per disease class, per split
SPLITS    = ["calibration", "test"]

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
PREV   = {s: os.path.join(ROOT, "Gold", "data", "gold_worklist_%s.csv" % s) for s in SPLITS}
DELETED = os.path.join(ROOT, "Gold", "corrections", "deleted_records.csv")

FKEYS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
         "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
BOUNDS = ["p_onset_sample","p_offset_sample","qrs_onset_sample","qrs_offset_sample",
          "t_onset_sample","t_offset_sample"]
REQ = (["record_id","disease_class","lead","beat_id","fs_hz","n_samples"] + FKEYS +
       ["p_present","qrs_present","t_present","beat_start_sample","beat_end_sample",
        "flags","also_delineator","priority","path_raw"])
EXTRA = ["gold_split","qc_status","n_recon_leads","ref_lead_mad_ms","p_dur_ms","pq_seg_ms","qt_ms"]


def main():
    g = pd.read_csv(RECON, low_memory=False)
    print("reconciled records in: %d" % len(g))

    # ---- hold out anything already reviewed under the old per-lead protocol ----
    seen = set()
    for c in CORR:
        if os.path.isfile(c):
            seen |= set(pd.read_csv(c, usecols=["record_id"])["record_id"])
    before = len(g)
    g = g[~g["record_id"].isin(seen)].copy()
    print("held out %d records already reviewed per-lead, %d eligible" % (before - len(g), len(g)))

    # ---- records the reviewer excluded in the tool ----
    dropped = set()
    if os.path.isfile(DELETED):
        dropped = set(pd.read_csv(DELETED)["record_id"])
        print("excluded %d record(s) the reviewer deleted in the tool" % len(dropped))
    g = g[~g["record_id"].isin(dropped)].copy()

    # ---- the clean tier is the only pool from here on ----
    pool = g[g["qc_status"] == "clean"].copy()
    print("clean pool: %d records" % len(pool))
    print(pool["disease_class"].value_counts().sort_index().to_string())

    # ---- what the previous draw already picked ----
    keep, prev_all = {}, set()
    for s in SPLITS:
        if os.path.isfile(PREV[s]):
            p = pd.read_csv(PREV[s], usecols=["record_id","disease_class","qc_status"])
            prev_all |= set(p["record_id"])
            k = p[(p["qc_status"] == "clean") & (~p["record_id"].isin(dropped))]
            keep[s] = k
            print("previous %-11s %3d records, %3d clean and kept, %3d replaced"
                  % (s, len(p), len(k), len(p) - len(k)))
        else:
            keep[s] = pd.DataFrame(columns=["record_id","disease_class"])

    kept_ids = set().union(*[set(k["record_id"]) for k in keep.values()])
    avail = pool[~pool["record_id"].isin(prev_all)]          # never re-use a dropped record

    # ---- top each class and split back up to PER_SPLIT ----
    rng = np.random.RandomState(SEED)
    rows = []
    short = []
    for cls in sorted(pool["disease_class"].unique()):
        cand = avail[avail["disease_class"] == cls]
        cand = cand.sort_values("record_id").reset_index(drop=True)
        order = rng.permutation(len(cand))
        cur = 0
        for s in SPLITS:
            have = keep[s][keep[s]["disease_class"] == cls]["record_id"].tolist()
            need = PER_SPLIT - len(have)
            if need > len(cand) - cur:
                short.append((cls, s, need - (len(cand) - cur)))
                need = len(cand) - cur
            new = cand.iloc[order[cur:cur + need]]["record_id"].tolist()
            cur += need
            for rid in have + new:
                rows.append((rid, s))
            print("  %-8s %-11s kept %2d + new %2d = %2d" % (cls, s, len(have), len(new), len(have) + len(new)))
    if short:
        for cls, s, n in short:
            print("  WARNING: class %s split %s is %d records short of %d" % (cls, s, n, PER_SPLIT))

    pick = pd.DataFrame(rows, columns=["record_id","gold_split"])
    S = pick.merge(g, on="record_id", how="left")
    assert S["disease_class"].notna().all(), "a picked record is missing from the reconciled table"

    # ---- reference lead: the lead whose own boundaries sit closest to the reconciled label ----
    # R-peak-aligned, exactly as in build_gold_worklist.py, since final_data_units.csv holds a
    # per-lead representative beat that coincides with the record-level one only 37.7% of the time.
    u = pd.read_csv(UNITS, usecols=["record_id","lead","r_peak_sample"] + BOUNDS)
    u = u[u["record_id"].isin(set(S["record_id"]))].copy()
    # aligned by explicit reindex rather than by DataFrame subtraction, since subtracting a
    # duplicate-labelled frame from a unique-labelled one returns rows in sorted label order
    # and the positional assign that follows would then attach each deviation to the wrong lead
    gr = S.set_index("record_id")
    gr = gr[BOUNDS].sub(gr["r_peak_sample"], axis=0)
    G = gr.reindex(u["record_id"].values).to_numpy(dtype=float)
    U = u[BOUNDS].to_numpy(dtype=float) - u["r_peak_sample"].to_numpy(dtype=float)[:, None]
    dev = np.abs(U - G).mean(axis=1) * 2.0
    u = u.assign(mad_ms=dev).sort_values(["record_id","mad_ms"])
    best = u.groupby("record_id", as_index=False).first()[["record_id","lead","mad_ms"]]
    S = S.merge(best, on="record_id", how="left")
    S["lead"] = S["lead"].fillna("II")
    S["ref_lead_mad_ms"] = S["mad_ms"].round(1)

    # ---- tool columns ----
    S["beat_id"]         = S["rep_beat_id"]
    S["also_delineator"] = 0
    S["priority"]        = 1
    S["p_dur_ms"]  = ((S["p_offset_sample"]  - S["p_onset_sample"])  * 2).round(0)
    S["pq_seg_ms"] = ((S["qrs_onset_sample"] - S["p_offset_sample"]) * 2).round(0)
    S["qt_ms"]     = ((S["t_offset_sample"]  - S["qrs_onset_sample"]) * 2).round(0)
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

    cal = out[out["gold_split"] == "calibration"].sort_values(["disease_class","record_id"])
    tst = out[out["gold_split"] == "test"].sort_values(["disease_class","record_id"])
    assert set(cal["record_id"]).isdisjoint(set(tst["record_id"])), "calibration and test overlap"
    assert not out["record_id"].duplicated().any(), "duplicate record in gold set"
    assert (out["qc_status"] == "clean").all(), "a non-clean record survived the filter"

    os.makedirs(OUTDIR, exist_ok=True)
    cal.to_csv(os.path.join(OUTDIR, "gold_worklist_calibration.csv"), index=False)
    tst.to_csv(os.path.join(OUTDIR, "gold_worklist_test.csv"), index=False)

    print("\nwrote %d calibration + %d test records -> %s" % (len(cal), len(tst), OUTDIR))
    print(pd.crosstab(out["disease_class"], out["gold_split"]).to_string())
    print("\nqc_status mix: %s" % out["qc_status"].value_counts().to_dict())
    print("carried over from the first draw: %d" % len(kept_ids & set(out["record_id"])))
    print("reference-lead agreement with the reconciled label (ms): median %.1f, p90 %.1f"
          % (out["ref_lead_mad_ms"].median(), out["ref_lead_mad_ms"].quantile(.9)))
    print("lead chosen as reference: %s" % out["lead"].value_counts().to_dict())


if __name__ == "__main__":
    main()
