import os
import numpy as np
import pandas as pd

MS_PER_SAMPLE = 2.0            # 500 Hz

FKEYS = ["p_onset_sample", "p_peak_sample", "p_offset_sample",
         "qrs_onset_sample", "q_peak_sample", "r_peak_sample", "s_peak_sample",
         "qrs_offset_sample", "t_onset_sample", "t_peak_sample", "t_offset_sample"]

# Boundaries are what the delineator gets wrong and what the offsets are estimated for.
# Peaks are validated by Check_Position_ECG_Waves.m and are reported for completeness only.
BOUNDS = ["p_onset_sample", "p_offset_sample", "qrs_onset_sample",
          "qrs_offset_sample", "t_onset_sample", "t_offset_sample"]
PEAKS = [f for f in FKEYS if f not in BOUNDS]

# A residual this large is not a boundary-placement disagreement. It means the reviewer
# and the label are on different beats, or a landmark was dragged onto the wrong feature.
# Such rows are excluded from the offset and counted separately rather than silently kept,
# since a single 700 ms outlier would move a mean and a trimmed mean alike.
OUTLIER_MS = 200.0


def repo_root(start):
    r = os.path.abspath(start)
    while r != os.path.dirname(r) and not os.path.isfile(os.path.join(r, "config", "paths.yaml")):
        r = os.path.dirname(r)
    return r


ROOT = repo_root(os.path.dirname(os.path.abspath(__file__)))


def gold_path(*parts):
    return os.path.join(ROOT, "Gold", *parts)


def load_pair(worklist_csv, corrections_csv, verbose=True):
    """
    Join the reconciled label (worklist) to the reviewed label (corrections) on record_id
    and return a tidy residual frame.

    The worklist carries exactly one row per record, holding the reconciled global label
    that the model will be trained on. The corrections file carries one row per record the
    reviewer touched or marked with r. Only rows with reviewed == 1 are used.
    """
    W = pd.read_csv(worklist_csv)
    if not os.path.isfile(corrections_csv):
        raise SystemExit(
            "No corrections file at %s\n"
            "Review the worklist in MATLAB first with\n"
            "    addpath('Gold/tool'); gold_label_ecg('%s')"
            % (corrections_csv, os.path.relpath(worklist_csv, ROOT))
        )
    C = pd.read_csv(corrections_csv)

    if W["record_id"].duplicated().any():
        raise SystemExit("BUG: worklist has duplicate record_id, it must be one row per record")

    if "reviewed" in C.columns:
        n_all = len(C)
        C = C[pd.to_numeric(C["reviewed"], errors="coerce").fillna(0) == 1].copy()
        if verbose and len(C) < n_all:
            print("  dropped %d corrections row(s) not marked reviewed" % (n_all - len(C)))
    # Pair on record AND lead. A correction is a review of one lead's trace, so the
    # worklist row it has to be scored against is the row for that same lead. Pairing on
    # record_id alone mates a review of one lead to the seed of another, and de-duplicating
    # on record_id alone throws the matching lead away whenever the non-matching one
    # happens to sort last. On this corpus that mates the wrong lead for records the reviewer
    # saw on two leads, and those two leads disagree by up to 276 ms on P onset.
    # On the calibration half the record_id-only rule misroutes 5 of those 29.
    key = ["record_id", "lead"] if ("lead" in W.columns and "lead" in C.columns) else ["record_id"]
    n_rev = len(C)
    C = C.drop_duplicates(subset=key, keep="last")

    M = W.merge(C, on=key, how="inner", suffixes=("_ref", "_rev"))
    if verbose:
        print("  paired on %s" % " + ".join(key))
        if n_rev - len(M):
            print("  %d reviewed row(s) name a unit the worklist does not carry, not scored"
                  % (n_rev - len(M)))
    cov = len(M) / max(1, len(W))
    if verbose:
        print("  worklist %d records, reviewed %d, coverage %.1f%%" % (len(W), len(M), 100 * cov))
        if cov < 0.95:
            print("  WARNING: coverage below 95%. Records inspected and left unchanged are")
            print("           only recorded if you pressed r on them. If you did not, the")
            print("           offsets below are conditioned on 'needed an edit' and are")
            print("           biased away from zero. Treat them as an upper bound.")
    if M.empty:
        raise SystemExit("No reviewed records in common between worklist and corrections.")

    rows = []
    for f in FKEYS:
        ref = pd.to_numeric(M[f + "_ref"], errors="coerce")
        rev = pd.to_numeric(M[f + "_rev"], errors="coerce")
        d = (rev - ref) * MS_PER_SAMPLE
        rows.append(pd.DataFrame({
            "record_id": M["record_id"].values,
            "disease_class": M["disease_class_ref"].values,
            "fiducial": f,
            "ref_ms": ref.values * MS_PER_SAMPLE,
            "rev_ms": rev.values * MS_PER_SAMPLE,
            "residual_ms": d.values,
        }))
    R = pd.concat(rows, ignore_index=True)

    R["absent_ref"] = R["ref_ms"].isna()
    R["absent_rev"] = R["rev_ms"].isna()
    R["usable"] = (~R["absent_ref"]) & (~R["absent_rev"]) & (R["residual_ms"].abs() <= OUTLIER_MS)
    R["outlier"] = (~R["absent_ref"]) & (~R["absent_rev"]) & (R["residual_ms"].abs() > OUTLIER_MS)
    return R, M


def boot_ci(x, stat=np.median, n=5000, seed=20260725, alpha=0.05):
    """Percentile bootstrap interval. Returns (point, lo, hi)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) < 8:
        return float(stat(x)), np.nan, np.nan
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(x), size=(n, len(x)))
    bs = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2)))


def paired_delta_ci(err_a, err_b, n=5000, seed=20260726, alpha=0.05):
    """
    Paired bootstrap on mean(|a|) - mean(|b|) over the same records.

    Pairing is what makes this test sensitive. Both candidates are scored on the identical
    set of records, so the record-to-record variation in how hard a record is cancels out
    and what remains is the effect of the offset alone.
    """
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8:
        return (float(np.mean(a) - np.mean(b)) if len(a) else np.nan), np.nan, np.nan, len(a)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(n, len(a)))
    bs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    d = float(a.mean() - b.mean())
    return d, float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))), len(a)


ORDER_CHAIN = ["p_onset_sample", "p_peak_sample", "p_offset_sample",
               "qrs_onset_sample", "q_peak_sample", "r_peak_sample", "s_peak_sample",
               "qrs_offset_sample", "t_onset_sample", "t_peak_sample", "t_offset_sample"]


def count_disorder(df, chain=ORDER_CHAIN):
    """
    Count rows where each adjacent landmark pair sits in the wrong order.

    Run this before and after offsetting. The corpus already contains disorder of its own,
    since the delineator snaps a boundary onto its neighbour when it cannot find the feature,
    so a raw count after the fact would blame the offsets for something they did not do.
    """
    out = {}
    for j in range(1, len(chain)):
        a = pd.to_numeric(df[chain[j - 1]], errors="coerce").values
        b = pd.to_numeric(df[chain[j]], errors="coerce").values
        ok = np.isfinite(a) & np.isfinite(b)
        n = int(np.sum(ok & (b < a)))
        if n:
            out["%s > %s" % (chain[j - 1], chain[j])] = n
    return out


def coincidence_split(before, moved, left, right, ms_per_sample=MS_PER_SAMPLE):
    """
    Split the rows a yielding landmark moved on by how much headroom it had to begin with.

    A landmark that was already sitting exactly on its neighbour was never an independent
    measurement. It is the delineator writing the neighbour's position twice, which the
    PQ headroom table in the dissertation documents at 40 to 57 per cent of beats. Moving
    it along with the neighbour preserves what it actually encoded. A landmark that had
    real headroom and got squeezed is a different case and is worth seeing separately.
    """
    gap = (pd.to_numeric(before[right], errors="coerce").values
           - pd.to_numeric(before[left], errors="coerce").values) * ms_per_sample
    m = np.asarray(moved, dtype=bool)
    return {
        "coincident": int(np.sum(m & (gap <= 0))),
        "squeezed": int(np.sum(m & (gap > 0))),
        "untouched_with_headroom": int(np.sum((~m) & (gap > 0))),
    }


def repair_order(df, corrected=(), chain=ORDER_CHAIN, strict=False):
    """
    Restore monotonic landmark order after offsets have been applied.

    Who is allowed to move matters more here than it looks, and getting it wrong silently
    throws the correction away. In this corpus the P offset already sits at or after the
    QRS onset in 40 to 57 per cent of beats depending on class, which is the whole point of
    the PQ headroom table in the dissertation. So a validated correction that pulls the QRS
    onset earlier lands on top of the P offset almost every time. Clamp the QRS onset back
    to the P offset and the correction is cancelled, the median shift comes out at zero and
    the output looks untouched while claiming to be de-biased.

    The rule is therefore a rank. Peaks never move, since the delineator validates peak
    positions and the QC found them reliable. Corrected boundaries never move either, since
    the held-out half produced evidence for exactly where they belong. Everything else
    yields. When a corrected boundary collides with an uncorrected neighbour, the neighbour
    is the one that gives way, which is the right call given that one of the two has
    evidence behind it and the other has none.

    strict=True restores the old behaviour, where every boundary can be clamped and a
    correction can lose to an uncorrected neighbour. Useful only as a comparison.

    Returns the repaired frame and a diagnostics dict.
    """
    arr = df[chain].astype("float64").values.copy()
    before = arr.copy()
    n = arr.shape[0]
    corrected = set(corrected)
    # a landmark is pinned if it is a peak, or a boundary the test half validated
    pinned = np.array([(c in PEAKS) or ((not strict) and (c in corrected)) for c in chain])

    # forward pass, push a yielding landmark right if it sits before an earlier one
    run = np.full(n, -np.inf)
    for j in range(len(chain)):
        cur = arr[:, j]
        if not pinned[j]:
            cur = np.where(np.isfinite(cur) & (cur < run), run, cur)
            arr[:, j] = cur
        run = np.where(np.isfinite(cur), np.maximum(run, cur), run)

    # backward pass, pull a yielding landmark left if it sits after a later one
    run = np.full(n, np.inf)
    for j in range(len(chain) - 1, -1, -1):
        cur = arr[:, j]
        if not pinned[j]:
            cur = np.where(np.isfinite(cur) & (cur > run), run, cur)
            arr[:, j] = cur
        run = np.where(np.isfinite(cur), np.minimum(run, cur), run)

    # anything still out of order is a collision between two pinned landmarks, which no
    # reordering can resolve without discarding evidence. Counted and reported, not hidden.
    V = arr.copy()
    unresolved = 0
    for j in range(1, len(chain)):
        a, b = V[:, j - 1], V[:, j]
        ok = np.isfinite(a) & np.isfinite(b)
        unresolved += int(np.sum(ok & (b < a)))

    moved = np.isfinite(before) & (arr != before)
    out = df.copy()
    out[chain] = arr
    diag = {
        "moved_total": int(moved.sum()),
        "moved_by_landmark": {c: int(moved[:, j].sum()) for j, c in enumerate(chain) if moved[:, j].sum()},
        "unresolved_pairs": unresolved,
        "pinned": [c for c, p in zip(chain, pinned) if p],
    }
    return out, diag
