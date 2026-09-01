#!/usr/bin/env python3
"""
build_clean_unit_labels.py  -  assemble the training label table for ml_modelling.

One row per clean unit, a unit being one record and one lead at its representative beat.
There are 194,645 of them and they are the training examples the delineation model sees.

What goes into a row, in order,

  1. The unit itself, taken from manual_labelling/data/all_units_worklist.csv filtered to
     qc_status == clean. That file still carries the original ECGdeli fiducials and none of
     them survive into the output, only the beat window, the presence flags and the paths.

  2. The measured fiducials, the estimator that re-derives every landmark from the raw
     twelve lead signal through the spatial magnitude and velocity curves. Two runs cover
     the set, measured_units.jsonl for the record level clean tier and pend_*.jsonl for the
     units that are clean in their own lead inside a record flagged elsewhere.

  3. The phantom Q rule. The estimator writes q_peak equal to qrs_onset when it finds no Q
     wave, so that value is a placeholder rather than a measurement. The gold review settled
     what it means. The reviewer blanked Q on 255 units and added one on none, and split by
     the placeholder the 103 placeholder units moved 30.0 ms against the review while the 41
     units with a real Q moved 0.0 ms. Every placeholder therefore becomes q_present 0 with
     a blank q_peak, so the model learns absence rather than a landmark sitting on the QRS
     onset.

  4. The gold offsets from Gold/data/gold_offsets_validated.csv, which is where the 400
     reviewed records enter the label table. Only qrs_onset earned a correction and only per
     disease class. Offsets are negative or zero, so the onset can only move earlier, and
     the only order that can break is P offset landing after it. Those are squeezed back
     under the same rule apply_gold_offsets.py uses, boundaries move and peaks stay fixed.

Nothing is overwritten. The output is a new file and every input is left alone. 
"""
import glob
import json
import os
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

UNITS   = ROOT + "manual_labelling/data/all_units_worklist.csv"
OFFSETS = ROOT + "Gold/data/gold_offsets_validated.csv"
GOLDIDS = ROOT + "Gold/data/gold_excluded_record_ids.csv"
RECON   = WORK + "recon_corrected.csv"
JSONL   = [WORK + "measured_units.jsonl"] + sorted(glob.glob(WORK + "pend_*.jsonl"))
OUT     = ROOT + "dataset_curation/data/assembled/clean_units_labels.csv"

PTS = ["p_onset", "p_peak", "p_offset", "qrs_onset", "q_peak", "r_peak", "s_peak",
       "qrs_offset", "t_onset", "t_peak", "t_offset"]
MS_PER_SAMPLE = 2.0
WRITE = "--write" in sys.argv


def main():
    # ---- 1. the clean units -------------------------------------------------
    keep = ["record_id", "disease_class", "lead", "beat_id", "fs_hz", "n_samples",
            "beat_start_sample", "beat_end_sample", "p_present", "qrs_present",
            "t_present", "path_raw", "qc_status", "rep_qc_status", "unit_worst_status"]
    U = pd.read_csv(UNITS, usecols=keep, low_memory=False)
    U = U[U["qc_status"] == "clean"].reset_index(drop=True)
    print("clean units: %d over %d records" % (len(U), U.record_id.nunique()))

    # ---- 2. the measured fiducials ------------------------------------------
    recs = []
    for path in JSONL:
        n0 = len(recs)
        with open(path) as fh:
            for line in fh:
                o = json.loads(line)
                if o.get("error"):
                    continue
                recs.append(o)
        print("  %-24s %7d units" % (os.path.basename(path), len(recs) - n0))
    M = pd.DataFrame(recs)
    M = M.drop_duplicates(subset=["record_id", "lead"], keep="last")
    cols = ["record_id", "lead"] + PTS + [c for c in
            ("conf_p", "conf_r", "conf_s", "conf_t", "clamped", "capped") if c in M.columns]
    M = M[cols]
    print("measured units available: %d" % len(M))

    n0 = len(U)
    U = U.merge(M, on=["record_id", "lead"], how="inner")
    if len(U) != n0:
        raise SystemExit("%d clean units have no measured value, refusing to guess"
                         % (n0 - len(U)))
    for p in PTS:
        U[p] = U[p].astype(int)

    # ---- 3. the phantom Q ---------------------------------------------------
    phantom = U["q_peak"] == U["qrs_onset"]
    U["q_present"] = (~phantom).astype(int)
    U.loc[phantom, "q_peak"] = np.nan
    print("phantom Q marked absent: %d of %d units, %.1f%%"
          % (phantom.sum(), len(U), 100.0 * phantom.mean()))

    # ---- 4. the gold offsets ------------------------------------------------
    V = pd.read_csv(OFFSETS)
    V = V[V["scope"] != "none"]
    if not len(V):
        print("no landmark earned a correction, labels pass through unchanged")
    U["qrs_onset_offset_ms"] = 0.0
    moved = {}
    for _, r in V.iterrows():
        f = r["fiducial"]
        if f != "qrs_onset_sample":
            raise SystemExit("this script only knows how to apply qrs_onset, got %s. "
                             "Re-check the offset table before trusting the output." % f)
        m = (U["disease_class"] == r["disease_class"]) if r["scope"] == "class" \
            else pd.Series(True, index=U.index)
        off = float(r["offset_ms"])
        if not off:
            continue
        shift = np.round(np.where(m, off / MS_PER_SAMPLE, 0.0)).astype(int)
        U["qrs_onset"] = U["qrs_onset"] + shift
        U.loc[m, "qrs_onset_offset_ms"] = off
        moved[r["disease_class"] if r["scope"] == "class" else "ALL"] = (off, int(m.sum()))
    for c in sorted(moved):
        print("  qrs_onset %-8s %+5.1f ms on %6d units" % (c, moved[c][0], moved[c][1]))

    # P offset is the only landmark the shift can overtake, since every offset is negative
    squeezed = U["p_offset"] > U["qrs_onset"]
    U.loc[squeezed, "p_offset"] = U.loc[squeezed, "qrs_onset"]
    pulled = U["p_peak"] > U["p_offset"]
    U.loc[pulled, "p_peak"] = U.loc[pulled, "p_offset"]
    pulled2 = U["p_onset"] > U["p_offset"]
    U.loc[pulled2, "p_onset"] = U.loc[pulled2, "p_offset"]
    print("order repair: p_offset squeezed on %d units, p_peak on %d, p_onset on %d"
          % (squeezed.sum(), pulled.sum(), pulled2.sum()))

    # ---- 5. provenance and splits -------------------------------------------
    R = pd.read_csv(RECON, usecols=["record_id", "split", "mi_subclass", "qc_status"],
                    low_memory=False).rename(columns={"qc_status": "record_qc_status"})
    U = U.merge(R, on="record_id", how="left")
    if U["split"].isna().any():
        raise SystemExit("%d units have no train or val split" % int(U["split"].isna().sum()))

    G = pd.read_csv(GOLDIDS, usecols=["record_id", "gold_split"])
    U = U.merge(G, on="record_id", how="left")
    U["in_gold"] = U["gold_split"].notna().astype(int)
    U["gold_split"] = U["gold_split"].fillna("")
    U["label_source"] = "measured_debiased"

    U = U.rename(columns={p: p + "_sample" for p in PTS})
    order = (["record_id", "disease_class", "mi_subclass", "lead", "beat_id", "split",
              "fs_hz", "n_samples", "path_raw", "beat_start_sample", "beat_end_sample"]
             + [p + "_sample" for p in PTS]
             + ["p_present", "q_present", "qrs_present", "t_present",
                "label_source", "qrs_onset_offset_ms", "qc_status", "record_qc_status",
                "rep_qc_status", "unit_worst_status", "in_gold", "gold_split"]
             + [c for c in ("conf_p", "conf_r", "conf_s", "conf_t", "clamped", "capped")
                if c in U.columns])
    U = U[order]

    # ---- 6. checks ----------------------------------------------------------
    chain = ["p_onset_sample", "p_offset_sample", "qrs_onset_sample",
             "qrs_offset_sample", "t_onset_sample", "t_offset_sample"]
    bad = 0
    for a, b in zip(chain, chain[1:]):
        bad += int((U[a] > U[b]).sum())
    oob = int(((U[[p + "_sample" for p in PTS]] < 0).any(axis=1)
               | (U[[p + "_sample" for p in PTS]].gt(U["n_samples"] - 1, axis=0)).any(axis=1)).sum())
    print()
    print("rows %d  records %d  boundary order violations %d  out of record values %d"
          % (len(U), U.record_id.nunique(), bad, oob))
    print("split: %s" % U["split"].value_counts().to_dict())
    print("gold units flagged: %d over %d records" % (U["in_gold"].sum(),
                                                      U.loc[U.in_gold == 1, "record_id"].nunique()))
    print("trainable after dropping gold: %d units over %d records"
          % ((U.in_gold == 0).sum(), U.loc[U.in_gold == 0, "record_id"].nunique()))
    print("class mix: %s" % U["disease_class"].value_counts().to_dict())

    if not WRITE:
        print("\ndry run, nothing written. Re-run with --write.")
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    U.to_csv(OUT, index=False)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
