import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_common import (FKEYS, BOUNDS, gold_path, load_pair, boot_ci, paired_delta_ci)

MIN_BIAS_MS = 2.0          # one sample at 500 Hz, the finest claim the data supports
MIN_CLASSES = 2            # classes that must still lean after the global offset

WORKLIST = sys.argv[1] if len(sys.argv) > 1 else gold_path("data", "gold_worklist_test.csv")
CORR     = sys.argv[2] if len(sys.argv) > 2 else gold_path("corrections", "gold_worklist_test_corrections.csv")
OFFCSV   = gold_path("data", "gold_offsets.csv")
OUTCSV   = gold_path("data", "gold_offsets_validated.csv")
OUTTXT   = gold_path("results", "validate_gold_offsets.txt")


def biased(x, seed):
    """Median signed residual with a bootstrap interval, and whether it clears the bar."""
    m, lo, hi = boot_ci(x, stat=np.median, seed=seed)
    sig = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0) and abs(m) >= MIN_BIAS_MS)
    return m, lo, hi, sig


def main():
    if not os.path.isfile(OFFCSV):
        raise SystemExit("No %s. Run fit_gold_offsets.py first." % OFFCSV)
    O = pd.read_csv(OFFCSV)

    print("validating on the TEST half, which the fit has never seen")
    print("  worklist   : %s" % WORKLIST)
    print("  corrections: %s" % CORR)
    R, _ = load_pair(WORKLIST, CORR)

    gmap = O[O["scope"] == "global"].set_index("fiducial")["offset_ms"].to_dict()
    cmap = {(r["fiducial"], r["disease_class"]): r["offset_ms"]
            for _, r in O[O["scope"] == "class"].iterrows()}

    rows, lines, notes = [], [], []
    w = lines.append
    w("Gold offset validation, held-out half")
    w("=====================================")
    w("Decided on residual BIAS, not on mean absolute error. A landmark is corrected only")
    w("if the reviewed labels lean consistently to one side of the reconciled ones by at")
    w("least %.1f ms with a bootstrap interval clear of zero. Mean absolute error is shown" % MIN_BIAS_MS)
    w("throughout and holds a veto, but it does not make the decision.")
    w("")
    w("%-20s %5s %16s %8s %8s %9s %9s  %s" % (
        "landmark", "n", "bias [95% CI]", "residual", "MAE0", "MAEfinal", "dMAE", "adopted"))

    for f in FKEYS:
        sub = R[(R["fiducial"] == f) & R["usable"]].copy()
        if sub.empty:
            continue
        d = sub["residual_ms"].values
        n = len(d)
        og = float(gmap.get(f, 0.0))
        oc_raw = sub["disease_class"].map(lambda c: cmap.get((f, c), np.nan)).values.astype(float)
        oc = np.where(np.isfinite(oc_raw), oc_raw, og)

        # step 1 - is there a bias at all
        b0, b0lo, b0hi, has_bias = biased(d, seed=101)

        adopt = "none"
        nclass_lean = 0
        if has_bias:
            # step 2 - does bias survive the global offset inside individual classes
            rg = d - og
            for cls in sorted(sub["disease_class"].unique()):
                m = sub["disease_class"].values == cls
                if m.sum() >= 12:
                    _, _, _, s = biased(rg[m], seed=202)
                    nclass_lean += int(s)
            adopt = "class" if nclass_lean >= MIN_CLASSES else "global"

        off_vec = {"none": np.zeros(n), "global": np.full(n, og), "class": oc}[adopt]
        e_final = np.abs(d - off_vec)
        e_none = np.abs(d)
        dmae, mlo, mhi, _ = paired_delta_ci(e_final, e_none, seed=303)

        # step 3 - veto an offset that makes absolute error significantly worse
        vetoed = False
        if adopt != "none" and np.isfinite(mlo) and mlo > 0:
            vetoed, adopt = True, "none"
            off_vec = np.zeros(n)
            e_final = e_none
            dmae, mlo, mhi, _ = 0.0, 0.0, 0.0, n
            notes.append("%s: offset removed a real bias but worsened absolute error, so it was vetoed." % f)

        b1, _, _, _ = biased(d - off_vec, seed=404)
        w("%-20s %5d %6.1f [%5.1f,%5.1f] %8.1f %8.1f %9.1f %9.1f  %s%s" % (
            f, n, b0, b0lo, b0hi, b1, e_none.mean(), e_final.mean(), dmae,
            adopt, "" if not vetoed else " (vetoed)"))

        if adopt == "class":
            for cls in sorted(sub["disease_class"].unique()):
                m = sub["disease_class"].values == cls
                rows.append({"fiducial": f, "scope": "class", "disease_class": cls,
                             "offset_ms": round(float(cmap.get((f, cls), og)), 2), "n_test": int(m.sum()),
                             "bias_before_ms": round(float(np.median(d[m])), 2),
                             "bias_after_ms": round(float(np.median((d - off_vec)[m])), 2),
                             "mae_before_ms": round(float(e_none[m].mean()), 2),
                             "mae_after_ms": round(float(e_final[m].mean()), 2)})
        else:
            rows.append({"fiducial": f, "scope": adopt, "disease_class": "ALL",
                         "offset_ms": round(og, 2) if adopt == "global" else 0.0, "n_test": n,
                         "bias_before_ms": round(b0, 2), "bias_after_ms": round(b1, 2),
                         "mae_before_ms": round(float(e_none.mean()), 2),
                         "mae_after_ms": round(float(e_final.mean()), 2)})

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
    w("Expect the bias column to shrink far more than the error column. That is the point.")
    w("The scatter left over is reviewer and label noise, it has no preferred direction and")
    w("a model trained on 16,000 records averages it away. The lean is what it cannot.")
    for nline in notes:
        w("")
        w("NOTE " + nline)
    w("")
    w("These numbers are the honest estimate of what de-biasing buys, since the offsets")
    w("were fitted on records that appear nowhere in this table. Do not re-fit and re-run.")
    txt = "\n".join(lines)
    open(OUTTXT, "w").write(txt + "\n")
    print()
    print(txt)
    print()
    print("wrote %s" % OUTCSV)
    print("wrote %s" % OUTTXT)


if __name__ == "__main__":
    main()
