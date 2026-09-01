"""Run the fiducial estimator over every lead of every clean record.

Reuses measure.measure() unchanged. Two additions.

  1. A one entry cache on the raw CSV loader. The twelve units of a record
     share one file, and the units of a record sit next to each other in
     final_data_units.csv even though the file is not globally sorted, so a
     single slot is enough to collapse twelve loads into one.
  2. Streaming. Each worker owns the records whose crc32 falls in its
     residue class and processes each row as it is read, holding nothing.
     An earlier version grouped every clean row in memory first, which on a
     3 GB device meant four workers competing for more RAM than existed and
     the run collapsed into swap at roughly one unit per second.

  3. Resume. A worker's unit sequence is fully determined by the residue
     class, so restarting one is a matter of skipping as many units as its
     output file already holds and appending from there. The device reaps
     long lived background processes after roughly half a minute, which
     killed two full runs at around nine thousand units per worker with no
     traceback, so the run is driven as a series of short resumable passes
     rather than one long one.

  python3 measure_all.py <worker_index> <n_workers> <out.jsonl>
"""
import csv, json, os, sys, time, zlib
import numpy as np

WORK = "/sessions/rcw-01eznxqjqyb87mcdkyb8z92s/work"
sys.path.insert(0, WORK)
import measure as M                      # ROOT, LEADS, PTS, measure()

# Both inputs can be overridden so the same estimator can be run over a different unit
# list without cloning the script. MEAS_RECON set to "-" turns the record level clean
# filter off, which is what you want when the unit list has already been filtered, for
# instance to the units that are clean in their own lead inside a record that is flagged
# elsewhere. With neither variable set the behaviour is exactly as it was.
UNITS = os.environ.get("MEAS_UNITS", M.ROOT + "manual_labelling/data/final_data_units.csv")
RECON = os.environ.get("MEAS_RECON", WORK + "/recon_corrected.csv")  # record level qc status

# ---- one entry raw CSV cache -------------------------------------------
# measure() does A = M[:, w0:w1+1].astype(float), which always copies, and it
# never writes into the loaded array, so handing out the same object is safe.
_cache = {"path": None, "arr": None}
_real_loadtxt = np.loadtxt


def _cached_loadtxt(path, **kw):
    if _cache["path"] != path:
        _cache["path"] = path
        _cache["arr"] = _real_loadtxt(path, **kw)
    return _cache["arr"]


M.np.loadtxt = _cached_loadtxt


def clean_records():
    if RECON == "-":
        return None
    keep = set()
    with open(RECON, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("qc_status") == "clean":
                keep.add(r["record_id"])
    return keep


def main():
    wi, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    keep = clean_records()
    sys.stderr.write("worker %d  %s clean records\n"
                     % (wi, "no filter" if keep is None else len(keep)))
    sys.stderr.flush()

    already = 0
    if os.path.exists(out):
        with open(out) as f:
            already = sum(1 for _ in f)
    sys.stderr.write("worker %d  resuming past %d units\n" % (wi, already))
    sys.stderr.flush()

    fh = open(out, "a")
    done = errs = seen = 0
    t0 = time.time()

    with open(UNITS, newline="") as f:
        for row in csv.DictReader(f):
            rid = row["record_id"]
            if keep is not None and rid not in keep:
                continue
            if zlib.crc32(rid.encode()) % nw != wi:
                continue
            seen += 1
            if seen <= already:          # written by an earlier pass
                continue
            try:
                cur = {k: int(float(row[k + "_sample"])) for k in M.PTS}
            except (ValueError, KeyError):
                fh.write(json.dumps(dict(record_id=rid, lead=row["lead"],
                                         error="unparsable sample column")) + "\n")
                errs += 1
                continue
            # the row IS the ECGdeli label, so it serves as both the current
            # value and the window anchor
            eco = dict(cur)
            present = {"p": int(float(row["p_present"])),
                       "t": int(float(row["t_present"]))}
            try:
                rec, diag = M.measure(row["path_raw"], row["lead"], cur, eco,
                                      present, int(float(row["n_samples"])))
            except Exception as e:
                fh.write(json.dumps(dict(record_id=rid, lead=row["lead"],
                                         error=repr(e)[:200])) + "\n")
                errs += 1
                continue
            o = dict(record_id=rid, lead=row["lead"],
                     beat_id=int(float(row["beat_id"])),
                     disease_class=row["disease_class"])
            for k in M.PTS:
                o[k] = int(rec[k])
                o["d_" + k] = int(rec[k]) - cur[k]
            o["conf_p"] = diag["conf_p"]
            o["conf_r"] = diag["conf_r"]
            o["conf_s"] = diag["conf_s"]
            o["conf_t"] = diag["conf_t"]
            o["clamped"] = diag["clamped"]
            o["capped"] = diag["capped"]
            fh.write(json.dumps(o) + "\n")
            done += 1
            if done % 500 == 0:
                fh.flush()
                sys.stderr.write("worker %d  %d units  %d errors  %.0f s\n"
                                 % (wi, done, errs, time.time() - t0))
                sys.stderr.flush()

    fh.close()
    sys.stderr.write("worker %d DONE %d units %d errors %.0f s\n"
                     % (wi, done, errs, time.time() - t0))


if __name__ == "__main__":
    main()
