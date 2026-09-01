#!/usr/bin/env python3
"""
apply_deletions.py  -  Permanently remove excluded records from the dataset.

Reads the exclusion list written by the corrector tool's Delete button
(deleted_records.csv, one record_id per row, whole-record) and streams each
target CSV, dropping every row whose record_id is on the list. Raw signal files
are kept (this is an exclusion, not a raw-file wipe). Each edited file is backed
up to a .bak copy first.

Targets (whichever exist)
    dataset_curation/data/assembled/master_labels.csv
    dataset_curation/data/review/signals_index.csv
    manual_labelling/data/*.csv          <- EVERY worklist in that folder, so the one the
                                            corrector tool is actually loaded with stays in
                                            sync (final_data_units.csv, all_units_worklist.csv,
                                            or any later worklist) without editing this script
    dataset_curation/data/global/reconciled_global_fiducials_corrected.csv

A file with no record_id column, or one where nothing matched, is left untouched.

Where it sits in the pipeline
    correct + delete in the tool  ->  merge_manual_corrections.py
    ->  apply_deletions.py  (this)  ->  build_edited_global.py

"""
import os, sys, csv, shutil, glob
csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)

if len(sys.argv) > 1:
    DELFILE = sys.argv[1]
else:
    # the tool writes corrections to manual_labelling/corrections (its path math is
    # data/../corrections). Fall back to data/corrections for older batch runs.
    _cands = [os.path.join(ROOT, "manual_labelling", "corrections", "deleted_records.csv"),
              os.path.join(ROOT, "manual_labelling", "data", "corrections", "deleted_records.csv")]
    DELFILE = next((c for c in _cands if os.path.isfile(c)), _cands[0])

# every worklist CSV sitting in manual_labelling/data, whichever one the tool was run with
# (final_data_units.csv is the current worklist; all_units_worklist.csv is the older full list).
# Discovered by glob so a new worklist never silently keeps the deleted records.
WORKLISTS = sorted(glob.glob(os.path.join(ROOT, "manual_labelling", "data", "*.csv")))

TARGETS = [
    os.path.join(ROOT, "dataset_curation", "data", "assembled", "master_labels.csv"),
    os.path.join(ROOT, "dataset_curation", "data", "review", "signals_index.csv"),
] + WORKLISTS + [
    os.path.join(ROOT, "dataset_curation", "data", "global", "reconciled_global_fiducials_corrected.csv"),
]

def load_deleted(path):
    if not os.path.isfile(path):
        sys.exit("no deletions file at %s (nothing to do)" % path)
    recs = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            r = (row.get("record_id") or "").strip()
            if r:
                recs.add(r)
    return recs

def _discard(tmp):
    """Drop an unused temp file. Some mounts refuse unlink, so fall back to truncating it."""
    try:
        os.remove(tmp)
    except OSError:
        try:
            open(tmp, "w").close()
        except OSError:
            pass


def filter_file(path, del_set):
    """Rewrite path without rows whose record_id is in del_set. Returns (kept, removed) or None."""
    if not os.path.isfile(path):
        print("  skip (not found) %s" % os.path.relpath(path, ROOT))
        return None
    tmp = path + ".tmp"
    kept = removed = 0
    with open(path, newline="") as fin, open(tmp, "w", newline="") as fout:
        first = fin.readline()
        header = next(csv.reader([first]))
        fout.write(first)
        if "record_id" not in header:
            _discard(tmp)
            print("  skip (no record_id column) %s" % os.path.relpath(path, ROOT))
            return None
        ri = header.index("record_id")
        if ri == 0:
            # record_id is the first field, so the row can be tested and copied without parsing it
            for line in fin:
                if line[:line.index(",")] in del_set:
                    removed += 1
                else:
                    fout.write(line); kept += 1
        else:
            r = csv.reader(fin); w = csv.writer(fout)
            for row in r:
                if row and row[ri] in del_set:
                    removed += 1
                else:
                    w.writerow(row); kept += 1
    if removed == 0:
        _discard(tmp)
        print("  unchanged (0 removed) %s" % os.path.relpath(path, ROOT))
        return (kept, 0)
    shutil.copy(path, path + ".bak")
    os.replace(tmp, path)
    print("  %-52s kept %d, removed %d (backup .bak)" % (os.path.relpath(path, ROOT), kept, removed))
    return (kept, removed)

def main():
    deleted = load_deleted(DELFILE)
    print("excluded records to apply: %d  (from %s)" % (len(deleted), os.path.relpath(DELFILE, ROOT)))
    if not deleted:
        sys.exit("deletions file is empty, nothing to do")
    for t in TARGETS:
        filter_file(t, deleted)
    print("done. Raw signal files were kept. Re-run build_edited_global.py to refresh the global target.")

if __name__ == "__main__":
    main()
