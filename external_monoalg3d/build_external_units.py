#!/usr/bin/env python3
"""
build_external_units.py  -  build the model-ready external test table.


Reads the reviewer's corrections file directly and writes the table the
ml_modelling loader consumes. There is no propagation step any more.

Every lead of every record in scope is reviewed by hand, so there is nothing left
to derive. The T boundaries are per-lead readings the reviewer placed, the peaks
are per-lead calls the reviewer made or accepted, and the QRS is the record-level
boundary the earlier pass agreed. propagate_to_all_leads existed to manufacture
the eleven leads nobody had looked at, and with all twelve reviewed it would only
be a chance to overwrite real review with a rule.

What is still done here is the conversion.

    scope       records named in smith2026_subset.csv, rows with reviewed = 1,
                and leads not flagged exclude
    landmarks   the 1000 Hz sample index i maps to the 500 Hz grid as (i + 1) // 2,
                which is round-half-up and reproduces the earlier export exactly
    _ms columns the original 1000 Hz index carried through unchanged, since one
                sample is one millisecond at 1000 Hz, so the reference keeps its
                full resolution and no rounding error is added to the label side
    window      the beat span plus 20 samples on each side, which is the rule the
                training corpus was cut with. The loader centres the window in the
                crop, so this is what puts the beat where the network expects it.
            
    signals     already decimated to 12 x 501 by export_test_set_500hz, so
                nothing is resampled here

Nothing is overwritten. Run with --write to actually write.
"""
import argparse
import collections
import csv
import os

LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LANDMARKS = ['p_onset', 'p_peak', 'p_offset',
             'qrs_onset', 'q_peak', 'r_peak', 's_peak', 'qrs_offset',
             't_onset', 't_peak', 't_offset']
PRESENCE = ['p_present', 'qrs_present', 'q_present', 'r_present', 's_present', 't_present']

SCHEMA = (['record_id', 'disease_class', 'lead', 'beat_id', 'split', 'fs_hz', 'n_samples',
           'path_raw', 'win_start_sample', 'win_end_sample']
          + [n + '_sample' for n in LANDMARKS]
          + [n + '_ms' for n in LANDMARKS]
          + PRESENCE)
EXTRA = ['label_quality', 'label_source']

ORDER = ['qrs_onset', 'qrs_offset', 't_onset', 't_peak', 't_offset']
PAIRS = [('q_peak', 'q_present'), ('r_peak', 'r_present'), ('s_peak', 's_present'),
         ('t_offset', 't_present')]
TLEAD_TAG = 'manual_tlead_perlead_T'


def blank(value):
    text = (value or '').strip()
    return text == '' or text.lower() in ('nan', 'none')


def to_500(value):
    """A 1000 Hz sample index on the 500 Hz grid, round half up. Blank stays blank."""
    if blank(value):
        return ''
    return str((int(round(float(value.strip()))) + 1) // 2)


def carry_ms(value):
    """The original 1000 Hz index is already the position in milliseconds."""
    if blank(value):
        return ''
    return str(int(round(float(value.strip()))))


def read_ids(path):
    """The record_id column of a small CSV. Empty list when the file is absent."""
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return [r['record_id'].strip() for r in csv.DictReader(f)
                if r.get('record_id', '').strip()]


def build(labels_csv, subset, signals_dir, fs_out, n_out, window='landmarks', margin=20):
    with open(labels_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    out, skipped = [], collections.Counter()
    for r in rows:
        rec = r['record_id'].strip()
        if subset and rec not in subset:
            skipped['not in subset'] += 1
            continue
        if (r.get('reviewed') or '').strip() != '1':
            skipped['not reviewed'] += 1
            continue
        if (r.get('exclude') or '').strip() == '1':
            skipped['lead excluded'] += 1
            continue
        item = {
            'record_id': rec,
            'disease_class': r.get('disease_class', '').strip(),
            'lead': r['lead'].strip(),
            'beat_id': (r.get('beat_id') or '1').strip(),
            'split': 'test',
            'fs_hz': str(fs_out),
            'n_samples': str(n_out),
            'path_raw': os.path.join(signals_dir, rec + '_raw.csv'),
            'win_start_sample': '0',
            'win_end_sample': str(n_out - 1),
        }
        for name in LANDMARKS:
            item[name + '_sample'] = to_500(r.get(name + '_sample', ''))
            item[name + '_ms'] = carry_ms(r.get(name + '_sample', ''))
        present = [int(item[n + '_sample']) for n in LANDMARKS if item[n + '_sample'] != '']
        if window == 'landmarks' and present:
            item['win_start_sample'] = str(max(0, min(present) - margin))
            item['win_end_sample'] = str(min(n_out - 1, max(present) + margin))
        for name in PRESENCE:
            item[name] = (r.get(name) or '').strip()
        src = (r.get('label_source') or '').strip()
        item['label_source'] = src
        item['label_quality'] = 'reviewed' if src == TLEAD_TAG else 'record_level'
        out.append(item)
    return out, skipped


def check(built, subset, expect_leads=12):
    """Integrity of the table itself. Returns a list of problems, empty when clean."""
    bad = []
    per = collections.defaultdict(list)
    for r in built:
        per[r['record_id']].append(r)

    if subset:
        missing = sorted(set(subset) - set(per))
        if missing:
            bad.append('%d record(s) in the subset produced no rows, first is %s'
                       % (len(missing), missing[0]))

    for rec in sorted(per):
        rows = per[rec]
        leads = [r['lead'] for r in rows]
        if len(set(leads)) != len(leads):
            bad.append('%s has a duplicated lead' % rec)
        short = expect_leads - len(rows)
        if short > 0:
            bad.append('%s has %d lead(s) of %d, missing %s'
                       % (rec, len(rows), expect_leads,
                          ' '.join(sorted(set(LEADS) - set(leads)))))
        qrs = {(r['qrs_onset_sample'], r['qrs_offset_sample']) for r in rows}
        if len(qrs) > 1:
            bad.append('%s carries %d different QRS boundary sets across its leads'
                       % (rec, len(qrs)))

    wide = [r for r in built
            if int(r['win_end_sample']) - int(r['win_start_sample']) + 1 > 900]
    if wide:
        bad.append('%d unit(s) carry a window wider than 900 samples. The loader places the crop '
                   'by centring the window in it, so a window that is not the beat span puts the '
                   'beat somewhere the network was not trained to look. Use --window landmarks.'
                   % len(wide))

    for r in built:
        tag = '%s %s' % (r['record_id'], r['lead'])
        seq = [(n, int(r[n + '_sample'])) for n in ORDER if r[n + '_sample'] != '']
        for (na, va), (nb, vb) in zip(seq, seq[1:]):
            if vb < va:
                bad.append('%s is out of order, %s at %d before %s at %d'
                           % (tag, na, va, nb, vb))
        for land, flag in PAIRS:
            has = r[land + '_sample'] != ''
            says = r[flag] == '1'
            if has != says:
                bad.append('%s has %s = %r but %s = %r'
                           % (tag, land, r[land + '_sample'], flag, r[flag]))
    return bad


def report(built):
    per = collections.defaultdict(list)
    for r in built:
        per[r['record_id']].append(r)
    varied = sum(1 for v in per.values()
                 if len({(x['t_onset_sample'], x['t_offset_sample']) for x in v}) > 1)
    print('records %d, leads per record %.1f, units %d'
          % (len(per), len(built) / max(len(per), 1), len(built)))
    print('per-lead T on %d of %d records' % (varied, len(per)))
    print('by class: %s' % dict(collections.Counter(
        r['disease_class'] for r in built if r['lead'] == 'I')))
    print('label_quality: %s' % dict(collections.Counter(r['label_quality'] for r in built)))
    widths = sorted(int(r['win_end_sample']) - int(r['win_start_sample']) + 1 for r in built)
    print('window width min %d, median %d, max %d samples (crop is 1280)'
          % (widths[0], widths[len(widths) // 2], widths[-1]))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--labels', default=os.path.join(here, 'labels', 'smith2026_manual_corrections.csv'))
    p.add_argument('--subset', default=os.path.join(here, 'smith2026_subset.csv'))
    p.add_argument('--signals', default=os.path.join(here, 'test_export', 'signals'))
    p.add_argument('--out', default=os.path.join(here, 'test_export', 'smith2026_test_units_alllead.csv'))
    p.add_argument('--fs-out', type=int, default=500)
    p.add_argument('--n-out', type=int, default=501)
    p.add_argument('--window', choices=['landmarks', 'record'], default='landmarks')
    p.add_argument('--margin', type=int, default=20)
    p.add_argument('--write', action='store_true')
    a = p.parse_args()

    subset = read_ids(a.subset)
    print('subset: %d records from %s' % (len(subset), a.subset) if subset
          else 'no subset file, every reviewed record is in scope')

    built, skipped = build(a.labels, set(subset), os.path.abspath(a.signals),
                           a.fs_out, a.n_out, a.window, a.margin)
    for reason, n in sorted(skipped.items()):
        print('skipped %d row(s), %s' % (n, reason))
    if not built:
        raise SystemExit('no rows survived the filters, nothing to write')
    report(built)

    problems = check(built, subset)
    if problems:
        print('\n%d integrity problem(s):' % len(problems))
        for line in problems[:40]:
            print('  ' + line)
        if len(problems) > 40:
            print('  ... and %d more' % (len(problems) - 40))
        raise SystemExit('refusing to write')
    print('integrity checks passed')

    missing = sorted({r['path_raw'] for r in built if not os.path.exists(r['path_raw'])})
    if missing:
        print('warning, %d signal file(s) not found, first is %s' % (len(missing), missing[0]))

    if not a.write:
        print('\ndry run, nothing written. Re-run with --write.')
        return
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA + EXTRA)
        w.writeheader()
        w.writerows(built)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
