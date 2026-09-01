#!/usr/bin/env python3
"""
apply_qrs_polarity.py  -  fold the clinical QRS names back into the label table.


Takes the output of rename_qrs_polarity.py and rewrites the Q, R and S columns of
dataset_curation/data/assembled/clean_units_labels.csv. The positions are the estimator's own
and none of them move. What changes is which of the three a landmark is called, and whether it
is called anything at all, since a complex with no positive deflection has no R and a complex
with no negative deflection ahead of the R has no Q.

r_present and s_present join the presence flags that were already there for P, Q, QRS and T, 
so a training loss can mask every landmark the same way. qrs_pattern
carries the morphology as a string, upper case for a deflection at least half the height of the
largest in the complex and lower case for a smaller one, which gives RS, rS, Q, R, Rs, QR and
the rest of the usual vocabulary.

"""
import glob
import json
import os
import shutil
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

UNITS = ROOT + "dataset_curation/data/assembled/clean_units_labels.csv"
BAK   = ROOT + "dataset_curation/data/assembled/clean_units_labels.prepolarity.bak.csv"
JSONL = sorted(glob.glob(WORK + "pol4_*.jsonl"))
WRITE = "--write" in sys.argv


def main():
    U = pd.read_csv(UNITS, low_memory=False)
    print("units %d over %d records" % (len(U), U.record_id.nunique()))

    recs = []
    for p in JSONL:
        with open(p) as fh:
            for line in fh:
                o = json.loads(line)
                if o.get("error"):
                    continue
                recs.append(o)
    P = pd.DataFrame(recs).drop_duplicates(subset=["record_id", "lead"], keep="last")
    print("renamed units available %d from %d shards" % (len(P), len(JSONL)))

    # the old Q presence flag is the phantom Q rule's answer and it is about to be replaced,
    # so its rate is recorded here and the column is dropped before the merge to keep the
    # names clean
    before = {"Q": float(U.q_present.mean()), "R": 1.0, "S": 1.0}
    U = U.drop(columns=["q_present"])

    n0 = len(U)
    U = U.merge(P[["record_id", "lead", "q_peak", "r_peak", "s_peak", "q_present",
                   "r_present", "s_present", "qrs_pattern"]],
                on=["record_id", "lead"], how="inner")
    if len(U) != n0:
        raise SystemExit("%d units have no renamed reading, refusing to write a mixed table"
                         % (n0 - len(U)))
    moved = {}
    for name, old, new in (("q", "q_peak_sample", "q_peak"),
                           ("r", "r_peak_sample", "r_peak"),
                           ("s", "s_peak_sample", "s_peak")):
        both = U[old].notna() & U[new].notna()
        moved[name] = (int((both & (U[old] != U[new])).sum()),
                       int((U[old].notna() & U[new].isna()).sum()),
                       int((U[old].isna() & U[new].notna()).sum()))
        U[old] = U[new]
    U = U.drop(columns=["q_peak", "r_peak", "s_peak"])
    U["q_present"] = U["q_present"].astype(int)
    U["r_present"] = U["r_present"].astype(int)
    U["s_present"] = U["s_present"].astype(int)

    print()
    print("  %-4s %10s %10s %10s" % ("", "renamed", "withdrawn", "added"))
    for k in ("q", "r", "s"):
        print("  %-4s %10d %10d %10d" % (k.upper(), moved[k][0], moved[k][1], moved[k][2]))
    print()
    print("presence after the rename, Q %.1f%% R %.1f%% S %.1f%%, Q was %.1f%%"
          % (100 * U.q_present.mean(), 100 * U.r_present.mean(), 100 * U.s_present.mean(),
             100 * before["Q"]))
    print("morphology: %s" % U.qrs_pattern.value_counts().head(12).to_dict())

    # every named peak has to sit inside its own complex, which is the one invariant the
    # rename could break if a position were carried across from the wrong unit
    bad = 0
    for c in ("q_peak_sample", "r_peak_sample", "s_peak_sample"):
        m = U[c].notna()
        bad += int((m & ((U[c] < U.qrs_onset_sample) | (U[c] > U.qrs_offset_sample))).sum())
    order = 0
    for a, b in (("q_peak_sample", "r_peak_sample"), ("r_peak_sample", "s_peak_sample")):
        m = U[a].notna() & U[b].notna()
        order += int((m & (U[a] > U[b])).sum())
    print("peaks outside their complex %d, Q after R or R after S %d" % (bad, order))
    if bad or order:
        raise SystemExit("the rename produced an impossible complex")

    # the three peak flags belong together where q_present used to sit, which is between the
    # P flag and the QRS flag, and the morphology string reads best next to qrs_present
    cols = list(U.columns)
    for c in ("q_present", "r_present", "s_present", "qrs_pattern"):
        cols.remove(c)
    i = cols.index("qrs_present")
    cols[i:i] = ["q_present", "r_present", "s_present"]
    cols.insert(cols.index("qrs_present") + 1, "qrs_pattern")
    U = U[cols]

    if not WRITE:
        print("\ndry run, nothing written. Re-run with --write.")
        return
    if not os.path.exists(BAK):
        shutil.copy2(UNITS, BAK)
        print("\nkept the previous table at %s" % os.path.basename(BAK))
    U.to_csv(UNITS, index=False)
    print("wrote %s" % UNITS)


if __name__ == "__main__":
    main()
