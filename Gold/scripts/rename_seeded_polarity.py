
import os

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
SEED = ROOT + "Gold/data/gold_worklist_calibration_seeded.csv"
UNITS = ROOT + "dataset_curation/data/assembled/clean_units_labels.csv"
OUT = ROOT + "Gold/data/gold_seeded_qrs_polarity.csv"
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def pattern(q, r, s, amp):
    big = max([abs(a) for a in amp.values()] or [1.0])
    out = []
    for name, pos in (("q", q), ("r", r), ("s", s)):
        if pos is None:
            continue
        out.append(name.upper() if abs(amp[pos]) >= 0.5 * big else name)
    return "".join(out) or "flat"


def main():
    S = pd.read_csv(SEED, low_memory=False)
    if "path_raw" not in S.columns:
        K = pd.read_csv(UNITS, usecols=["record_id", "path_raw"],
                        low_memory=False).drop_duplicates("record_id")
        S = S.merge(K, on="record_id", how="left")
    S = S.sort_values(["path_raw", "record_id", "lead"]).reset_index(drop=True)
    print("seeded units %d over %d records" % (len(S), S.record_id.nunique()))
    miss = int(S.path_raw.isna().sum())
    if miss:
        print("no raw path for %d units, they are records the reseed dropped" % miss)
        S = S[S.path_raw.notna()].reset_index(drop=True)

    path, M, rows = None, None, []
    for _, r in S.iterrows():
        try:
            if path != r.path_raw:
                A = np.loadtxt(ROOT + r.path_raw, delimiter=",")
                if A.shape[0] > A.shape[1]:
                    A = A.T
                path, M = r.path_raw, A
            x = M[LEADS.index(r.lead)].astype(float)

            qon, qoff = int(r.qrs_onset_sample), int(r.qrs_offset_sample)
            base = float(x[min(max(qon, 0), len(x) - 1)])

            # the same two placeholders the corpus rename skips, a Q sitting on the QRS onset
            # is the no Q marker and an S sitting on the offset is the unresolved parking spot
            pos = []
            if pd.notna(r.q_peak_sample) and int(r.q_peak_sample) != qon:
                pos.append(int(r.q_peak_sample))
            if pd.notna(r.r_peak_sample):
                pos.append(int(r.r_peak_sample))
            if pd.notna(r.s_peak_sample) and int(r.s_peak_sample) < qoff:
                pos.append(int(r.s_peak_sample))
            pos = sorted(set(p for p in pos if 0 <= p < len(x)))
            amp = {p: float(x[p]) - base for p in pos}

            up = [p for p in pos if amp[p] > 0]
            dn = [p for p in pos if amp[p] <= 0]
            if up:
                rp = up[0]
                q = max([p for p in dn if p < rp], default=None)
                s = min([p for p in dn if p > rp], default=None)
            else:
                rp, s = None, None
                q = min(dn) if dn else None

            rows.append(dict(record_id=r.record_id, lead=r.lead,
                             q_peak_sample=q, r_peak_sample=rp, s_peak_sample=s,
                             q_present=int(q is not None), r_present=int(rp is not None),
                             s_present=int(s is not None),
                             qrs_pattern=pattern(q, rp, s, amp)))
        except Exception as exc:                              # noqa: BLE001
            rows.append(dict(record_id=r.record_id, lead=r.lead,
                             error="%s: %s" % (type(exc).__name__, exc)))

    R = pd.DataFrame(rows)
    err = R.get("error")
    if err is not None:
        n = int(err.notna().sum())
        print("errors %d" % n)
        R = R[err.isna()].drop(columns=["error"])
    print("renamed %d units" % len(R))
    print("Q present %.1f%%  R present %.1f%%  S present %.1f%%"
          % (100 * R.q_present.mean(), 100 * R.r_present.mean(), 100 * R.s_present.mean()))
    print("morphology %s" % R.qrs_pattern.value_counts().head(10).to_dict())

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    R.to_csv(OUT, index=False)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
