
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_common import (FKEYS, BOUNDS, gold_path, load_pair, boot_ci, paired_delta_ci)

MIN_BIAS_MS = 2.0          # one sample at 500 Hz, the finest claim the data supports
MIN_CLASSES = 2            # classes that must still lean after the global offset
K_FOLDS = 5
FOLD_SEED = 20260728

# q_peak_sample can never be offset. The estimator writes q_peak equal to qrs_onset when it
# finds no Q wave, so that value is a placeholder and not a measurement. 317 of the 399
# seeded calibration units carry it. Split by whether the seed held the placeholder, the
# 103 placeholder units move 30.0 ms and the 41 units with a real Q move 0.0 ms, and the
# reviewer blanked Q on 255 units while adding one on none. The apparent global bias is
# therefore the placeholder being scored as a landmark, and applying it would displace every
# genuine Q peak in the corpus by roughly one half of a QRS width.
NEVER_ADOPT = {"q_peak_sample"}

WORKLIST = sys.argv[1] if len(sys.argv) > 1 else gold_path("data", "gold_worklist_calibration_seeded.csv")
CORR     = sys.argv[2] if len(sys.argv) > 2 else gold_path("corrections", "gold_worklist_calibration_corrections.csv")
OFFCSV   = gold_path("data", "gold_offsets.csv")
OUTCSV   = gold_path("data", "gold_offsets_validated.csv")
OUTTXT   = gold_path("results", "crossvalidate_gold_offsets.txt")


def biased(x, seed):
    """Median signed residual with a bootstrap interval, and whether it clears the bar."""
    m, lo, hi = boot_ci(x, stat=np.median, seed=seed)
    sig = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0) and abs(m) >= MIN_BIAS_MS)
    return m, lo, hi, sig


def make_folds(classes, k=K_FOLDS, seed=FOLD_SEED):
    """
    Stratified folds, so every fold holds roughly ten records of every disease class.

    Unstratified folds on 50 records per class would leave a fold with too few of some
    class to fit a per class offset at all, and the comparison between the global and the
    per class candidate would then be decided by which fold a record landed in.
    """
    rng = np.random.RandomState(seed)
    fold = np.empty(len(classes), dtype=int)
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def main():
    if not os.path.isfile(OFFCSV):
        raise SystemExit("No %s. Run fit_gold_offsets.py first." % OFFCSV)
    O = pd.read_csv(OFFCSV)

    print("cross validating the offsets on the reviewed half, %d folds" % K_FOLDS)
    print("  worklist   : %s" % WORKLIST)
    print("  corrections: %s" % CORR)
    R, _ = load_pair(WORKLIST, CORR)

    gmap = O[O["scope"] == "global"].set_index("fiducial")["offset_ms"].to_dict()
    cmap = {(r["fiducial"], r["disease_class"]): r["offset_ms"]
            for _, r in O[O["scope"] == "class"].iterrows()}

    rows, lines, notes = [], [], []
    w = lines.append
    w("Gold offset cross validation, %d folds over the reviewed half" % K_FOLDS)
    w("=============================================================")
    w("Every record is scored by an offset fitted on the other folds, so no record")
    w("contributes to the offset that judges it. A landmark is adopted only if a majority")
    w("of folds found a bias of at least %.1f ms with a bootstrap interval clear of zero," % MIN_BIAS_MS)
    w("and the pooled out of sample absolute error did not get significantly worse.")
    w("Landmarks on the never adopt list are held back whatever the numbers say, and that")
    w("list currently holds %s." % ", ".join(sorted(NEVER_ADOPT)))
    w("")
    w("%-20s %5s %16s %9s %8s %9s %9s  %s" % (
        "landmark", "n", "bias [95% CI]", "residual", "MAE0", "MAEcv", "dMAE", "adopted"))

    for f in FKEYS:
        sub = R[(R["fiducial"] == f) & R["usable"]].copy().reset_index(drop=True)
        if len(sub) < 40:
            continue
        d = sub["residual_ms"].values
        cls = sub["disease_class"].values
        n = len(d)
        fold = make_folds(cls)

        b0, b0lo, b0hi, _ = biased(d, seed=101)
        off_cv = np.zeros(n)
        votes, fold_offsets = [], []

        for kf in range(K_FOLDS):
            te = fold == kf
            tr = ~te
            dtr, ctr = d[tr], cls[tr]
            og = float(np.median(dtr))
            fold_offsets.append(og)
            _, _, _, has_bias = biased(dtr, seed=1000 + kf)
            adopt_k = "none"
            if has_bias:
                lean = 0
                for c in np.unique(ctr):
                    m = ctr == c
                    if m.sum() >= 12:
                        _, _, _, s = biased(dtr[m] - og, seed=2000 + kf)
                        lean += int(s)
                adopt_k = "class" if lean >= MIN_CLASSES else "global"
            votes.append(adopt_k)
            if adopt_k == "global":
                off_cv[te] = og
            elif adopt_k == "class":
                per = {c: float(np.median(dtr[ctr == c])) for c in np.unique(ctr)}
                off_cv[te] = [per.get(c, og) for c in cls[te]]

        e_none = np.abs(d)
        e_cv = np.abs(d - off_cv)
        dmae, mlo, mhi, _ = paired_delta_ci(e_cv, e_none, seed=303)

        # majority vote over folds, ties and a lack of quorum both fall back to none
        tally = {v: votes.count(v) for v in set(votes)}
        adopt = max(tally, key=lambda v: tally[v])
        if tally[adopt] <= K_FOLDS // 2:
            adopt = "none"
        vetoed = False
        if adopt != "none" and f in NEVER_ADOPT:
            vetoed, adopt = True, "none"
            notes.append("%s: held out error fell, but the landmark is on the never adopt "
                         "list and the apparent bias is the no Q placeholder rather than a "
                         "measurement, so nothing is applied." % f)
        if adopt != "none" and np.isfinite(mlo) and mlo > 0:
            vetoed, adopt = True, "none"
            notes.append("%s: the offset removed a real bias but made out of sample absolute "
                         "error worse, so it was vetoed." % f)
        if adopt == "none":
            off_cv = np.zeros(n)
            e_cv = e_none
            dmae = 0.0

        b1 = float(np.median(d - off_cv))
        spread = max(fold_offsets) - min(fold_offsets)
        w("%-20s %5d %6.1f [%5.1f,%5.1f] %9.1f %8.1f %9.1f %9.1f  %s%s" % (
            f, n, b0, b0lo, b0hi, b1, e_none.mean(), e_cv.mean(), dmae,
            adopt, " (vetoed)" if vetoed else ""))
        notes.append("%s: fold offsets ranged %.1f to %.1f ms, a spread of %.1f ms, and the "
                     "fold votes were %s." % (f, min(fold_offsets), max(fold_offsets), spread,
                                              ", ".join(votes)))

        og_full = float(gmap.get(f, 0.0))
        if adopt == "class":
            for c in sorted(np.unique(cls)):
                m = cls == c
                rows.append({"fiducial": f, "scope": "class", "disease_class": c,
                             "offset_ms": round(float(cmap.get((f, c), og_full)), 2),
                             "n_test": int(m.sum()),
                             "bias_before_ms": round(float(np.median(d[m])), 2),
                             "bias_after_ms": round(float(np.median((d - off_cv)[m])), 2),
                             "mae_before_ms": round(float(e_none[m].mean()), 2),
                             "mae_after_ms": round(float(e_cv[m].mean()), 2)})
        else:
            rows.append({"fiducial": f, "scope": adopt, "disease_class": "ALL",
                         "offset_ms": round(og_full, 2) if adopt == "global" else 0.0,
                         "n_test": n,
                         "bias_before_ms": round(b0, 2), "bias_after_ms": round(b1, 2),
                         "mae_before_ms": round(float(e_none.mean()), 2),
                         "mae_after_ms": round(float(e_cv.mean()), 2)})

    V = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTTXT), exist_ok=True)
    V.to_csv(OUTCSV, index=False)

    kept = V[V["scope"] != "none"]["fiducial"].nunique()
    w("")
    w("%d of %d landmarks earned a correction. The rest keep their reconciled label." % (kept, len(FKEYS)))
    b = V[V["fiducial"].isin(BOUNDS)]
    if len(b):
        w("Boundaries only, mean absolute residual bias %.1f ms before and %.1f ms after,"
          % (b["bias_before_ms"].abs().mean(), b["bias_after_ms"].abs().mean()))
        w("and mean absolute error %.1f ms before and %.1f ms after."
          % (b["mae_before_ms"].mean(), b["mae_after_ms"].mean()))
    w("")
    w("The offsets written out are the ones fitted on the full reviewed half, since that is")
    w("the better estimate of each number. What is cross validated is the decision to adopt")
    w("them at all, and the fold to fold spread reported below is the evidence that the")
    w("number does not depend on which records happened to be in the fit.")
    w("")
    w("LIMITATION. These records are not a fresh sample. They were reviewed knowing which")
    w("landmarks the estimator was likely to get wrong, so the benefit reported here is an")
    w("upper bound. A genuinely untouched half would be stronger evidence and the design")
    w("still holds 400 such records in gold_worklist_test_seeded.csv if there is ever time.")
    for nline in notes:
        w("")
        w("NOTE " + nline)
    txt = "\n".join(lines)
    open(OUTTXT, "w").write(txt + "\n")
    print()
    print(txt)
    print()
    print("wrote %s" % OUTCSV)
    print("wrote %s" % OUTTXT)
    print()
    print("next: python3 Gold/scripts/apply_gold_offsets.py --recon "
          "dataset_curation/data/global/reconciled_global_fiducials.csv")


if __name__ == "__main__":
    main()
