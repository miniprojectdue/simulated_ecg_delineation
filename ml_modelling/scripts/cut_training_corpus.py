#!/usr/bin/env python3
"""
cut_training_corpus.py  -  split the label tables into the sets the model sees.

    python3 ml_modelling/scripts/cut_training_corpus.py [--write]

    pretrain    dataset_curation/data/assembled/clean_units_labels.csv, minus every record a
                human has touched. Estimator labels, de-biased by the validated per-class
                QRS onset offsets, carrying the phantom Q rule.
    finetune    Gold/data/gold_reviewed_labels.csv, the reviewer's own labels.
    test        external, MonoAlg3D, not in this repository and deliberately so.

Held-out records are dropped whole, never unit by unit. The fine-tuning set keeps the
reviewer's own train and validation split rather than inventing one.

Both output tables gain win_start_sample and win_end_sample: the beat window opened out far
enough to hold every landmark the unit carries with a 40 ms margin. The beat window itself is
left alone and stays in the table.
"""

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
CLEAN    = ROOT + "dataset_curation/data/assembled/clean_units_labels.csv"
REVIEWED = ROOT + "Gold/data/gold_reviewed_labels.csv"
GOLDIDS  = ROOT + "Gold/data/gold_excluded_record_ids.csv"
OUTDIR   = ROOT + "ml_modelling/data/"

WRITE = "--write" in sys.argv

PTS = ["p_onset", "p_peak", "p_offset", "qrs_onset", "q_peak", "r_peak", "s_peak",
       "qrs_offset", "t_onset", "t_peak", "t_offset"]
COLS = [p + "_sample" for p in PTS]
PAD_MS = 40.0


def windows(name, T):
    """widen the crop until it covers every landmark the unit carries.

    The representative beat window was cut from one R peak to the next, which is the right
    way to pick a beat and the wrong way to crop one. A T offset routinely lands after the
    window closes and a P onset routinely lands before it opens, so a third of the units
    carry a landmark the loader would never show the network. Nothing here moves a label.
    The window is opened to the earliest landmark and closed after the latest, a 40 ms
    margin is added on each side so a boundary is never sitting on the edge of the crop,
    and the result is clipped to the record. Consecutive windows may now overlap, which is
    fine, the crops are independent samples and not a partition of the trace.
    """
    lo = T[COLS].min(axis=1)
    hi = T[COLS].max(axis=1)
    pad = PAD_MS / (1000.0 / T["fs_hz"])
    T["win_start_sample"] = np.minimum(T.beat_start_sample, lo.fillna(T.beat_start_sample))
    T["win_end_sample"] = np.maximum(T.beat_end_sample, hi.fillna(T.beat_end_sample))
    T["win_start_sample"] = np.maximum(0, np.floor(T.win_start_sample - pad)).astype(int)
    T["win_end_sample"] = np.minimum(T.n_samples - 1,
                                     np.ceil(T.win_end_sample + pad)).astype(int)

    out = pd.Series(False, index=T.index)
    for c in COLS:
        m = T[c].notna()
        out |= m & ((T[c] < T.win_start_sample) | (T[c] > T.win_end_sample))
    beat = (T.beat_end_sample - T.beat_start_sample) * (1000.0 / T.fs_hz)
    win = (T.win_end_sample - T.win_start_sample) * (1000.0 / T.fs_hz)
    print("  %s window, beat median %.0f ms, crop median %.0f ms, widest %.0f ms"
          % (name, beat.median(), win.median(), win.max()))
    print("  landmarks left outside the crop %d" % int(out.sum()))
    if int(out.sum()):
        raise SystemExit("the padded window still clips a landmark")
    return T


def main():
    K = pd.read_csv(CLEAN, low_memory=False)
    L = pd.read_csv(REVIEWED, low_memory=False)
    G = pd.read_csv(GOLDIDS, usecols=["record_id"])
    print("pretrain candidates %d units over %d records"
          % (len(K), K.record_id.nunique()))
    print("reviewed            %d units over %d records" % (len(L), L.record_id.nunique()))

    held = set(G.record_id) | set(L.record_id)
    print("held out records: %d gold, %d reviewed, %d together"
          % (len(set(G.record_id)), len(set(L.record_id)), len(held)))
    extra = sorted(set(L.record_id) - set(G.record_id))
    if extra:
        print("reviewed but not on the gold exclusion list, so held out by name:")
        for r in extra:
            print("    %s" % r)

    P = K[~K.record_id.isin(held)].reset_index(drop=True)
    print()
    print("pretrain  %d units over %d records" % (len(P), P.record_id.nunique()))
    print("  split %s" % P.split.value_counts().to_dict())
    print("  class %s" % P.disease_class.value_counts().to_dict())
    print("  lead  %d distinct, %.1f units per record" % (P.lead.nunique(),
                                                          len(P) / P.record_id.nunique()))
    print("  Q present on %d units, %.1f%%" % (P.q_present.sum(), 100.0 * P.q_present.mean()))

    print("finetune  %d units over %d records" % (len(L), L.record_id.nunique()))
    print("  split %s" % L.split.value_counts().to_dict())
    print("  class %s" % L.disease_class.value_counts().to_dict())
    print("  order clean %d, flagged %d" % (int(L.order_ok.sum()), int((L.order_ok == 0).sum())))

    # ---- the checks that matter ---------------------------------------------
    overlap_rec = set(P.record_id) & set(L.record_id)
    pk = set(zip(P.record_id, P.lead))
    lk = set(zip(L.record_id, L.lead))
    print()
    print("records shared between pretrain and finetune %d, units shared %d"
          % (len(overlap_rec), len(pk & lk)))
    if overlap_rec:
        raise SystemExit("a record appears in both sets, that is leakage")

    # A record must sit entirely inside one split, otherwise the validation score is
    # measuring memorisation of a torso rather than delineation.
    for name, T in (("pretrain", P), ("finetune", L)):
        straddle = T.groupby("record_id")["split"].nunique()
        n = int((straddle > 1).sum())
        print("%s records straddling the train and validation line: %d" % (name, n))
        if n:
            raise SystemExit("split is not clean at record level in %s" % name)

    print()
    P = windows("pretrain", P)
    L = windows("finetune", L)

    if not WRITE:
        print("\ndry run, nothing written. Re-run with --write.")
        return
    os.makedirs(OUTDIR, exist_ok=True)
    P.to_csv(OUTDIR + "pretrain_units.csv", index=False)
    L.to_csv(OUTDIR + "finetune_units.csv", index=False)
    pd.DataFrame({"record_id": sorted(held)}).to_csv(OUTDIR + "held_out_record_ids.csv",
                                                     index=False)
    print("\nwrote %spretrain_units.csv" % OUTDIR)
    print("wrote %sfinetune_units.csv" % OUTDIR)
    print("wrote %sheld_out_record_ids.csv" % OUTDIR)


if __name__ == "__main__":
    main()
