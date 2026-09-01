
import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gold_common import (FKEYS, BOUNDS, MS_PER_SAMPLE, ORDER_CHAIN, ROOT, gold_path,
                         repair_order, count_disorder, coincidence_split)

VALID = gold_path("data", "gold_offsets_validated.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", required=True, help="reconciled global labels CSV to de-bias")
    ap.add_argument("--out", default=gold_path("data", "reconciled_global_debiased.csv"))
    ap.add_argument("--offsets", default=VALID)
    ap.add_argument("--keep-gold", action="store_true",
                    help="keep the 800 gold records in the output (default drops them)")
    a = ap.parse_args()

    if not os.path.isfile(a.offsets):
        raise SystemExit("No %s. Run validate_gold_offsets.py first." % a.offsets)
    if os.path.abspath(a.out) == os.path.abspath(a.recon):
        raise SystemExit("Refusing to overwrite the input. Choose a different --out.")

    V = pd.read_csv(a.offsets)
    G = pd.read_csv(a.recon, low_memory=False)
    print("reconciled records in: %d" % len(G))

    gold = pd.concat([pd.read_csv(gold_path("data", f))[["record_id", "disease_class", "gold_split"]]
                      for f in ("gold_worklist_calibration.csv", "gold_worklist_test.csv")],
                     ignore_index=True)
    gold.to_csv(gold_path("data", "gold_excluded_record_ids.csv"), index=False)
    print("gold records held out of training: %d" % len(gold))

    before = G[FKEYS].astype("float64").copy()
    disorder_before = count_disorder(G)
    applied, skipped, corrected = [], [], []
    for f in FKEYS:
        rows = V[V["fiducial"] == f]
        if rows.empty or (rows["scope"] == "none").all():
            skipped.append(f)
            continue
        if (rows["scope"] == "global").any():
            off = float(rows[rows["scope"] == "global"]["offset_ms"].iloc[0]) / MS_PER_SAMPLE
            G[f] = G[f] + off
            applied.append("%s global %+.1f ms" % (f, off * MS_PER_SAMPLE))
        else:
            m = rows.set_index("disease_class")["offset_ms"].to_dict()
            off = G["disease_class"].map(m).astype("float64").fillna(0.0) / MS_PER_SAMPLE
            G[f] = G[f] + off
            applied.append("%s per class %+.1f to %+.1f ms" % (
                f, min(m.values()), max(m.values())))
        corrected.append(f)

    # A landmark that earned an offset is pinned during the repair. The held-out half is the
    # only evidence in this pipeline about where that boundary belongs, and an uncorrected
    # neighbour carries none, so the neighbour is what gives way when the two collide.
    G, diag = repair_order(G, corrected=corrected)
    for f in FKEYS:
        G[f] = G[f].round()

    shift = ((G[FKEYS].astype("float64") - before).abs() * MS_PER_SAMPLE)
    print()
    print("applied : %s" % ("; ".join(applied) if applied else "nothing"))
    print("unchanged: %s" % (", ".join(skipped) if skipped else "none"))
    print()
    print("order repair")
    print("  pinned, never moved : %s" % (", ".join(diag["pinned"]) or "none"))
    print("  values moved        : %d" % diag["moved_total"])
    for k, v in sorted(diag["moved_by_landmark"].items(), key=lambda kv: -kv[1]):
        j = ORDER_CHAIN.index(k)
        moved_mask = G[k].values != (before[k].values + 0.0)
        parts = ""
        if j + 1 < len(ORDER_CHAIN) and ORDER_CHAIN[j + 1] in corrected:
            s = coincidence_split(before, moved_mask, k, ORDER_CHAIN[j + 1])
            parts = "  (%d were already sitting on %s, %d had headroom and were squeezed)" % (
                s["coincident"], ORDER_CHAIN[j + 1], s["squeezed"])
        elif j > 0 and ORDER_CHAIN[j - 1] in corrected:
            s = coincidence_split(before, moved_mask, ORDER_CHAIN[j - 1], k)
            parts = "  (%d were already sitting on %s, %d had headroom and were squeezed)" % (
                s["coincident"], ORDER_CHAIN[j - 1], s["squeezed"])
        print("    %-20s %8d%s" % (k, v, parts))

    # A yielding landmark that was already coincident with its corrected neighbour was never
    # an independent measurement. The delineator snapped it there when it could not find the
    # feature, which the PQ headroom table records at 40 to 57 per cent of beats, so moving
    # it along preserves exactly what it encoded. A landmark with real headroom keeps it.
    disorder_after = count_disorder(G)
    new_pairs = {k: v - disorder_before.get(k, 0) for k, v in disorder_after.items()
                 if v > disorder_before.get(k, 0)}
    if disorder_before:
        print("  disorder already in the input, not caused by these offsets")
        for k, v in sorted(disorder_before.items(), key=lambda kv: -kv[1]):
            print("    %-44s %8d" % (k, v))
    if new_pairs:
        print("  disorder the offsets created and the repair could not resolve")
        for k, v in sorted(new_pairs.items(), key=lambda kv: -kv[1]):
            print("    %-44s %8d" % (k, v))
        print("    A landmark ran out of room between two neighbours that both hold evidence,")
        print("    a validated offset or a delineator-checked peak, so no reordering resolves")
        print("    it without throwing evidence away. Reported rather than quietly clamped.")
        print("    Treat these rows as missing for the affected landmark if the count is")
        print("    large enough to matter. Here it is %.2f%% of the corpus."
              % (100.0 * sum(new_pairs.values()) / max(1, len(G))))
    elif not disorder_after:
        print("  landmark order is intact on every row")
    print()
    print("median absolute shift per landmark, ms, over the records the offset addresses")
    for f in BOUNDS:
        # A per class offset can be zero for classes that carry most of the corpus, and the
        # median over every record then reads zero even though the correction landed. Score
        # each landmark on the records it was meant to move, and report the reach alongside.
        moved = shift[f][shift[f] > 0]
        med = moved.median() if len(moved) else 0.0
        reach = 100.0 * len(moved) / max(1, len(shift))
        note = ""
        if f in corrected and not len(moved):
            note = "   <-- correction cancelled, check the repair"
        print("  %-20s %6.1f   on %5.1f%% of records%s" % (f, med, reach, note))

    if not a.keep_gold:
        n0 = len(G)
        G = G[~G["record_id"].isin(set(gold["record_id"]))].copy()
        print()
        print("dropped %d gold records, %d remain for train and val" % (n0 - len(G), len(G)))

    d = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(d, exist_ok=True)
    G.to_csv(a.out, index=False)
    print()
    print("wrote %s" % a.out)
    print("the input %s is unchanged" % a.recon)


if __name__ == "__main__":
    main()
