import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_common import (FKEYS, BOUNDS, OUTLIER_MS, gold_path, load_pair, boot_ci)

WORKLIST = sys.argv[1] if len(sys.argv) > 1 else gold_path("data", "gold_worklist_calibration.csv")
CORR     = sys.argv[2] if len(sys.argv) > 2 else gold_path("corrections", "gold_worklist_calibration_corrections.csv")
OUTCSV   = gold_path("data", "gold_offsets.csv")
OUTTXT   = gold_path("results", "fit_gold_offsets.txt")


def summarise(sub, scope, cls):
    use = sub[sub["usable"]]
    d = use["residual_ms"].values
    off, lo, hi = boot_ci(d, stat=np.median)
    base = float(np.mean(np.abs(d))) if len(d) else np.nan
    resid = float(np.mean(np.abs(d - off))) if len(d) else np.nan
    return {
        "scope": scope,
        "disease_class": cls,
        "fiducial": sub["fiducial"].iloc[0],
        "is_boundary": sub["fiducial"].iloc[0] in BOUNDS,
        "n": int(len(d)),
        "n_outlier": int(sub["outlier"].sum()),
        "n_absent_ref": int(sub["absent_ref"].sum()),
        "n_absent_rev": int(sub["absent_rev"].sum()),
        "offset_ms": round(off, 2) if np.isfinite(off) else np.nan,
        "ci_lo_ms": round(lo, 2) if np.isfinite(lo) else np.nan,
        "ci_hi_ms": round(hi, 2) if np.isfinite(hi) else np.nan,
        # a CI that straddles zero means this landmark shows no bias worth correcting
        "nonzero": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        "mae_before_ms": round(base, 2) if np.isfinite(base) else np.nan,
        "mae_after_ms": round(resid, 2) if np.isfinite(resid) else np.nan,
    }


def main():
    print("fitting offsets on the CALIBRATION half")
    print("  worklist   : %s" % WORKLIST)
    print("  corrections: %s" % CORR)
    R, M = load_pair(WORKLIST, CORR)

    recs = []
    for f in FKEYS:
        sub = R[R["fiducial"] == f]
        recs.append(summarise(sub, "global", "ALL"))
        for cls, s2 in sub.groupby("disease_class", sort=True):
            recs.append(summarise(s2, "class", cls))
    O = pd.DataFrame(recs)

    os.makedirs(os.path.dirname(OUTCSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUTTXT), exist_ok=True)
    O.to_csv(OUTCSV, index=False)

    lines = []
    w = lines.append
    w("Gold calibration offsets")
    w("========================")
    w("records reviewed: %d of %d in the calibration half" % (M["record_id"].nunique(), len(pd.read_csv(WORKLIST))))
    w("residuals beyond %.0f ms excluded as wrong-feature rather than mis-placed" % OUTLIER_MS)
    w("")
    w("A positive offset means the reviewer moved the landmark LATER than the label had it.")
    w("Add the offset to the label to remove the bias. An interval marked no in the")
    w("distinguishable column has a confidence interval covering zero, so there is no bias")
    w("to correct and that landmark should be left alone.")
    w("")
    w("GLOBAL")
    w("%-20s %5s %9s %18s %6s %10s %10s" % ("landmark", "n", "offset", "95% CI", "signif", "MAE pre", "MAE post"))
    g = O[O["scope"] == "global"]
    for _, r in g.iterrows():
        w("%-20s %5d %8.1f  [%6.1f, %6.1f] %6s %9.1f %9.1f" % (
            r["fiducial"], r["n"], r["offset_ms"], r["ci_lo_ms"], r["ci_hi_ms"],
            "yes" if r["nonzero"] else "no", r["mae_before_ms"], r["mae_after_ms"]))
    w("")
    w("PER CLASS, boundaries only")
    for f in BOUNDS:
        w("")
        w("  %s" % f)
        sub = O[(O["scope"] == "class") & (O["fiducial"] == f)].sort_values("offset_ms")
        for _, r in sub.iterrows():
            w("    %-10s n=%3d  %7.1f ms  [%6.1f, %6.1f]  %-3s  MAE %5.1f -> %5.1f" % (
                r["disease_class"], r["n"], r["offset_ms"], r["ci_lo_ms"], r["ci_hi_ms"],
                "yes" if r["nonzero"] else "no", r["mae_before_ms"], r["mae_after_ms"]))
    w("")
    w("These are CANDIDATES. None of them is adopted until validate_gold_offsets.py has")
    w("scored them on the test half, which this script has never seen.")
    txt = "\n".join(lines)
    open(OUTTXT, "w").write(txt + "\n")
    print()
    print(txt)
    print()
    print("wrote %s" % OUTCSV)
    print("wrote %s" % OUTTXT)
    print()
    print("next: python3 Gold/scripts/validate_gold_offsets.py")


if __name__ == "__main__":
    main()
