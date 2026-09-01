#!/usr/bin/env python3
"""
repair_labels.py  -  reconcile the QRS with the per-lead T review, once.

    python3 repair_labels.py [--write]

The per-lead pass was scoped to the T wave. Some QRS boundaries were also moved
while it ran, and on several records the reviewer's own per-lead T onsets then
fell before the QRS offset they had set. A boundary pair that overlaps has no
meaning in a segmentation and the exported table needs a total order, so it has
to be reconciled before anything is scored.

Three things are done and nothing else. Every T reading placed during the review
is preserved except where step three has no alternative.

    1. QRS reverted to the first pass ONLY where the review contradicts itself,
       meaning the QRS offset sits later than the earliest T onset the same
       reviewer placed across that record's twelve leads. Those two readings
       cannot both stand and the T was read twelve times to the QRS's once, so
       the QRS is the one that gives way. Every other QRS correction is kept.
    2. QRS forced identical across the twelve leads of a record, using the value
       from step one. Three records carry a one to four millisecond disagreement
       between leads, which is a record-level quantity holding two values.
    3. T onset clamped to the QRS offset wherever it still falls earlier. These
       are the residual overlaps that are not explained by the QRS edits. State
       the rule in the write-up rather than leaving it to be inferred.
    4. A T peak that ends up outside its own T window is moved to the largest
       absolute departure from baseline inside that window, which is the rule
       propagate_to_all_leads already used for a derived peak. A peak outside the
       wave it belongs to is a slip rather than a reading, and every one moved is
       listed with its before and after so the change is not silent.

The corrections file is backed up before it is rewritten and every change is
listed, so the diff is auditable rather than a black box.
"""
import argparse, collections, csv, os, shutil, sys, time

QON, QOFF, TON = 'qrs_onset_sample', 'qrs_offset_sample', 't_onset_sample'
TPK, TOFF = 't_peak_sample', 't_offset_sample'
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def num(v):
    v = (v or '').strip()
    return int(float(v)) if v else None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', default=os.path.join(here, 'labels', 'smith2026_manual_corrections.csv'))
    p.add_argument('--baseline', required=True,
                   help='the pre-review backup the QRS is reverted to')
    p.add_argument('--signals', default=os.path.join(here, 'test_export', 'signals'))
    p.add_argument('--write', action='store_true')
    a = p.parse_args()
    a_sig = a.signals

    with open(a.labels, newline='') as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())
    base = {}
    with open(a.baseline, newline='') as f:
        for r in csv.DictReader(f):
            base.setdefault(r['record_id'], (num(r[QON]), num(r[QOFF])))

    per = collections.defaultdict(list)
    for r in rows:
        per[r['record_id']].append(r)

    reverted, unified, clamped = [], [], []
    for rec in sorted(per):
        rs = per[rec]
        # The reviewer's own reading is kept unless it contradicts itself. A QRS
        # offset later than the earliest T onset that same reviewer placed across
        # the twelve leads is the one case where the two readings cannot both
        # stand, and there the record falls back to the first pass, which is the
        # magnitude rule. Everywhere else the review is the more considered
        # reading and is left alone.
        have = collections.Counter((num(r[QON]), num(r[QOFF])) for r in rs)
        mine = have.most_common(1)[0][0]
        tons = [num(r[TON]) for r in rs if num(r[TON]) is not None]
        conflict = bool(tons) and min(tons) < mine[1]
        want = mine
        if conflict:
            fallback = base.get(rec)
            if fallback and None not in fallback:
                want = fallback
                reverted.append((rec, mine, want, sum(1 for t in tons if t < mine[1])))
        if len(have) > 1:
            unified.append((rec, sorted(have), want))
        for r in rs:
            r[QON], r[QOFF] = str(want[0]), str(want[1])
            t = num(r[TON])
            if t is not None and t < want[1]:
                clamped.append((rec, r['lead'], t, want[1]))
                r[TON] = str(want[1])

    print('QRS reverted to the first pass on %d record(s), where the review '
          'contradicted itself' % len(reverted))
    print('   %-24s %8s %10s %s' % ('record', 'yours', 'reverted', 'leads that disagreed'))
    for rec, mine, want, k in sorted(reverted, key=lambda x: -x[3]):
        print('   %-24s %8d %10d   %d of 12' % (rec, mine[1], want[1], k))
    kept = sum(1 for rec in per
               if base.get(rec) and (num(per[rec][0][QON]), num(per[rec][0][QOFF])) != base[rec])
    print('\nQRS edits kept exactly as reviewed on the other records')
    print('\nQRS made identical across leads on %d record(s)' % len(unified))
    for rec, had, want in unified:
        print('   %-24s %s -> %s' % (rec, had, want))
    print('\nT onset clamped to the QRS offset on %d unit(s)' % len(clamped))
    if clamped:
        moves = sorted(w - t for _, _, t, w in clamped)
        print('   moved by a median of %d ms, max %d ms' % (moves[len(moves) // 2], moves[-1]))
        byrec = collections.Counter(c[0] for c in clamped)
        for rec, k in byrec.most_common():
            print('   %-24s %2d lead(s)' % (rec, k))

    # ---- 4. a T peak outside its own window ---------------------------
    moved = []
    for r in rows:
        # NOT named a, which is the parsed arguments and would be shadowed here.
        t0, t1, tp = num(r[TON]), num(r[TOFF]), num(r[TPK])
        if None in (t0, t1, tp) or t0 <= tp <= t1:
            continue
        path = os.path.join(a_sig, r['record_id'] + '_raw.csv')
        if not os.path.exists(path):
            print('cannot check %s %s, no signal at %s' % (r['record_id'], r['lead'], path))
            continue
        import numpy as np
        A = np.loadtxt(path, delimiter=',')
        if A.shape[0] != 12:
            A = A.T
        V = A[LEADS.index(r['lead'])].astype(float)
        n = len(V)
        base = float(np.median(V[int(0.85 * n):]))
        lo, hi = max(0, t0 // 2), min(n - 1, t1 // 2)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(np.abs(V[lo:hi + 1] - base)))
        moved.append((r['record_id'], r['lead'], tp, k * 2))
        r[TPK] = str(k * 2)
    print('\nT peak moved back inside its own window on %d unit(s)' % len(moved))
    for rec, lead, was, now in moved:
        print('   %-24s %-4s %4d -> %4d ms' % (rec, lead, was, now))

    left = sum(1 for r in rows
               if num(r[TON]) is not None and num(r[TON]) < num(r[QOFF]))
    print('\nremaining overlaps after the repair: %d' % left)
    if left:
        raise SystemExit('the repair did not resolve everything, refusing to write')

    if not a.write:
        print('\ndry run, nothing written. Re-run with --write.')
        return
    bak = '%s.bak_repair_%s' % (a.labels, time.strftime('%Y%m%d_%H%M%S'))
    shutil.copy2(a.labels, bak)
    with open(a.labels, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print('\nbacked up to %s\nrewrote %s' % (bak, a.labels))


if __name__ == '__main__':
    main()
