
import os
import subprocess
import sys

import numpy as np
import pandas as pd

def _repo_root():
    """Walk up from this file to the folder holding config/paths.yaml."""
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(here, "config", "paths.yaml")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise RuntimeError("could not locate the repository root above %s" % __file__)
        here = parent


ROOT = _repo_root() + "/"
WORK = "/sessions/rcw-01eznxqjqyb87mcdkyb8z92s/work/"

CORR   = ROOT + "Gold/corrections/gold_worklist_calibration_corrections.csv"
SEED   = ROOT + "Gold/data/gold_worklist_calibration_seeded.csv"
UNITS  = ROOT + "manual_labelling/data/all_units_worklist.csv"
MASTER = ROOT + "dataset_curation/data/assembled/master_labels.csv"
RECON  = WORK + "recon_corrected.csv"
OUT    = ROOT + "Gold/data/gold_reviewed_labels.csv"
TMP    = "/tmp/_gold_master_rows.csv"

PTS = ["p_onset", "p_peak", "p_offset", "qrs_onset", "q_peak", "r_peak", "s_peak",
       "qrs_offset", "t_onset", "t_peak", "t_offset"]
COLS = [p + "_sample" for p in PTS]
WRITE = "--write" in sys.argv


def master_rows(record_ids):
    """Pull the master label rows for a handful of records without reading a 1 GB file.

    A plain read of master_labels.csv runs the device out of memory, and a chunked read of
    2.6 million rows costs most of a minute. A fixed string grep on the record identifier
    does the same job in under a second, and the header is prepended by hand.
    """
    ids = "/tmp/_gold_ids.txt"
    with open(ids, "w") as fh:
        fh.write("\n".join(sorted(record_ids)) + "\n")
    head = open(MASTER).readline()
    with open(TMP, "w") as fh:
        fh.write(head)
        fh.flush()  # grep writes to the same descriptor, so the header has to land first
        subprocess.run(["grep", "-F", "-f", ids, MASTER], stdout=fh,
                       env=dict(os.environ, LC_ALL="C"), check=False)
    return pd.read_csv(TMP, usecols=["record_id", "lead", "beat_id", "beat_start_sample",
                                     "beat_end_sample", "split", "mi_subclass", "qc_status"],
                       low_memory=False)


def main():
    C = pd.read_csv(CORR)
    print("corrections: %d rows, %d records, %d unique record and lead"
          % (len(C), C.record_id.nunique(), len(C[["record_id", "lead"]].drop_duplicates())))
    if len(C) != len(C[["record_id", "lead"]].drop_duplicates()):
        raise SystemExit("the corrections file holds more than one row for some unit")

    # ---- 1. the beat window, the split and the MI subclass -------------------
    M = master_rows(set(C.record_id))
    M = M.rename(columns={"qc_status": "qc_status_master"})
    n0 = len(C)
    C = C.merge(M, on=["record_id", "lead", "beat_id"], how="left")
    if len(C) != n0:
        raise SystemExit("the master table holds duplicate beats for some reviewed unit")
    lost = C["beat_start_sample"].isna()
    print("beat window found for %d of %d units" % (len(C) - int(lost.sum()), len(C)))
    if lost.any():
        # Two records were pulled out of the corpus after the review had already touched
        # them, so their corrections point at beats that no longer exist. Those rows are
        # dropped and named. A record that is still in the corpus but missing the reviewed
        # beat would be a real defect, so that case stops the run instead.
        gone = set(M.record_id)
        orphan = C.loc[lost]
        removed = orphan[~orphan.record_id.isin(gone)]
        broken = orphan[orphan.record_id.isin(gone)]
        for _, r in removed.iterrows():
            print("  dropped, record no longer in the corpus: %s %s beat %d"
                  % (r.record_id, r.lead, r.beat_id))
        if len(broken):
            print(broken[["record_id", "lead", "beat_id"]].to_string())
            raise SystemExit("the record is present but the reviewed beat is not, "
                             "refusing to guess a beat window")
        C = C[~lost].reset_index(drop=True)
    for c in ("beat_start_sample", "beat_end_sample", "beat_id"):
        C[c] = C[c].astype(int)

    # ---- 2. the raw path and the unit level quality columns ------------------
    U = pd.read_csv(UNITS, usecols=["record_id", "lead", "path_raw", "qc_status",
                                    "rep_qc_status", "unit_worst_status"], low_memory=False)
    U = U.drop_duplicates(subset=["record_id", "lead"], keep="first")
    C = C.merge(U, on=["record_id", "lead"], how="left")
    orphan_path = int(C["path_raw"].isna().sum())
    if orphan_path:
        # The units worklist dropped a lead here, so rebuild the path from the record name.
        # WP2_largeDataset_Noise/<class dir>/<train or test>/run_S<nn>/run_<nnnnnn>_raw.csv is
        # the fixed layout, and the record identifier carries every field of it.
        byrec = C.dropna(subset=["path_raw"]).drop_duplicates("record_id").set_index("record_id")["path_raw"]
        C["path_raw"] = C["path_raw"].fillna(C["record_id"].map(byrec))
        print("path rebuilt from a sibling lead on %d units, %d still missing"
              % (orphan_path, int(C["path_raw"].isna().sum())))

    # ---- 3. the gold half and the record level quality -----------------------
    S = pd.read_csv(SEED, usecols=["record_id", "lead", "gold_split", "path_raw"])
    C = C.merge(S[["record_id", "gold_split"]].drop_duplicates("record_id"),
                on="record_id", how="left")
    C["seeded_lead"] = C.merge(S[["record_id", "lead"]].assign(s=1), on=["record_id", "lead"],
                               how="left")["s"].fillna(0).astype(int)
    unseeded = int(C["gold_split"].isna().sum())
    C["gold_split"] = C["gold_split"].fillna("unseeded")
    print("units on the seeded lead %d, on a lead the reviewer chose instead %d, "
          "on a record the reseed dropped %d"
          % (int(C.seeded_lead.sum()), int((C.seeded_lead == 0).sum()) - unseeded, unseeded))

    R = pd.read_csv(RECON, usecols=["record_id", "qc_status"], low_memory=False)
    R = R.rename(columns={"qc_status": "record_qc_status"})
    C = C.merge(R, on="record_id", how="left")
    C["record_qc_status"] = C["record_qc_status"].fillna("not_reconciled")

    # ---- 4. presence flags and the phantom Q sanity check --------------------
    # The estimator writes q_peak equal to qrs_onset when it finds no Q, and the reviewer
    # answered that by blanking the landmark and clearing the flag. Anything left where the
    # flag is set but the sample is blank, or the other way round, is a data entry slip.
    for f, p in (("q_peak_sample", "q_present"), ("r_peak_sample", "r_present"),
                 ("s_peak_sample", "s_present"), ("p_peak_sample", "p_present"),
                 ("t_peak_sample", "t_present")):
        blank, flag = C[f].isna(), C[p] == 1
        n_bad = int((blank & flag).sum() + (~blank & ~flag).sum())
        print("  %-18s blank %3d  present flag %3d  disagreements %d"
              % (f, int(blank.sum()), int(flag.sum()), n_bad))

    # ---- 5. order, flagged and never repaired -------------------------------
    chain = ["p_onset_sample", "p_peak_sample", "p_offset_sample", "qrs_onset_sample",
             "qrs_offset_sample", "t_onset_sample", "t_peak_sample", "t_offset_sample"]
    ok = pd.Series(True, index=C.index)
    for a, b in zip(chain, chain[1:]):
        both = C[a].notna() & C[b].notna()
        ok &= ~(both & (C[a] > C[b]))
    inside = pd.Series(True, index=C.index)
    for f in ("q_peak_sample", "r_peak_sample", "s_peak_sample"):
        m = C[f].notna()
        inside &= ~(m & ((C[f] < C["qrs_onset_sample"]) | (C[f] > C["qrs_offset_sample"])))
    # The two failures are kept apart. chain_ok is the boundary sequence, and it breaking
    # means a segment was read as starting before the one ahead of it ended. peaks_inside_ok
    # is the weaker check that Q, R and S sit within their own complex, and it breaks by a
    # single sample on units the reseed never covered. order_ok is both together and is the
    # flag training should filter on.
    C["chain_ok"] = ok.astype(int)
    C["peaks_inside_ok"] = inside.astype(int)
    C["order_ok"] = (ok & inside).astype(int)
    print("boundary order violations %d, QRS peaks outside the complex %d, units passing both %d"
          % (int((~ok).sum()), int((~inside).sum()), int(C.order_ok.sum())))
    for _, r in C.loc[C.order_ok == 0].iterrows():
        print("    %-42s %-3s beat %d  Poff %s QRSon %s Q %s QRSoff %s Ton %s"
              % (r.record_id, r.lead, r.beat_id, r.p_offset_sample, r.qrs_onset_sample,
                 r.q_peak_sample, r.qrs_offset_sample, r.t_onset_sample))

    oob = int(((C[COLS] < 0).any(axis=1)
               | C[COLS].gt(C["n_samples"] - 1, axis=0).any(axis=1)).sum())

    # ---- 6. the shared schema -----------------------------------------------
    C["label_source"] = "human_reviewed"
    C["qrs_onset_offset_ms"] = 0.0
    C["in_gold"] = 1
    C["reviewed_at"] = C["edited_at"]
    order = (["record_id", "disease_class", "mi_subclass", "lead", "beat_id", "split",
              "fs_hz", "n_samples", "path_raw", "beat_start_sample", "beat_end_sample"]
             + COLS
             + ["p_present", "q_present", "r_present", "s_present", "qrs_present", "t_present",
                "label_source", "qrs_onset_offset_ms", "qc_status", "record_qc_status",
                "rep_qc_status", "unit_worst_status", "in_gold", "gold_split", "seeded_lead",
                "chain_ok", "peaks_inside_ok", "order_ok", "reviewed_at"])
    C = C[order]

    print()
    print("rows %d  records %d  out of record values %d" % (len(C), C.record_id.nunique(), oob))
    print("class mix: %s" % C.disease_class.value_counts().to_dict())
    print("gold half: %s" % C.gold_split.value_counts().to_dict())
    print("unit qc: %s" % C.qc_status.value_counts(dropna=False).to_dict())
    print("landmarks present per unit, mean %.2f of 11" % C[COLS].notna().sum(axis=1).mean())

    if not WRITE:
        print("\ndry run, nothing written. Re-run with --write.")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    C.to_csv(OUT, index=False)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
