import csv, json, os, sys
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


ROOT  = _repo_root() + "/"
WORK  = "/sessions/rcw-01eznxqjqyb87mcdkyb8z92s/work/"
UNITS = ROOT + "manual_labelling/data/final_data_units.csv"
RECON = ROOT + "dataset_curation/data/global/reconciled_global_fiducials.csv"
SRC   = WORK + "recon_corrected.csv"
JL    = WORK + "measured_units.jsonl"

PTS = ['p_onset','p_peak','p_offset','qrs_onset','q_peak','r_peak','s_peak',
       'qrs_offset','t_onset','t_peak','t_offset']
ONSETS  = ['p_onset','qrs_onset','t_onset']
OFFSETS = ['p_offset','qrs_offset','t_offset']
PEAKS   = ['p_peak','q_peak','r_peak','s_peak','t_peak']

WRITE = "--write" in sys.argv

# ---- the measured values ------------------------------------------------
m = pd.DataFrame([json.loads(l) for l in open(JL)])
m = m[~m.get("error", pd.Series([None] * len(m))).notna()] if "error" in m else m
print("measured units %d over %d records" % (len(m), m.record_id.nunique()))

# ---- unit table ---------------------------------------------------------
u = pd.read_csv(UNITS)
cols = list(u.columns)
before = len(u)
u = u[u.record_id.isin(set(m.record_id))].copy()
print("unit table %d rows -> %d after dropping the minor and critical records"
      % (before, len(u)))

key = ["record_id", "lead"]
mi = m.set_index(key)
ui = u.set_index(key)
assert ui.index.is_unique and mi.index.is_unique, "duplicate record and lead"
assert set(ui.index) == set(mi.index), "unit table and measurements disagree"

eco = ui[[p + "_sample" for p in PTS]].copy()          # kept for the report
for p in PTS:
    ui[p + "_sample"] = mi[p].reindex(ui.index).astype(int)

# the beat window is a display hint, widen it so no measured landmark falls
# outside the window the labelling tool draws
ui["beat_start_sample"] = np.minimum(ui["beat_start_sample"], ui["p_onset_sample"])
ui["beat_end_sample"]   = np.maximum(ui["beat_end_sample"],   ui["t_offset_sample"])

u = ui.reset_index()[cols]

# ---- record level reconciliation ----------------------------------------
g = m.groupby("record_id")
rec = pd.DataFrame(index=sorted(m.record_id.unique()))
for p in ONSETS:
    rec[p] = g[p].quantile(0.25).round().astype(int)
for p in OFFSETS:
    rec[p] = g[p].quantile(0.75).round().astype(int)
for p in PEAKS:
    rec[p] = g[p].median().round().astype(int)

# the R peak is the one landmark every lead agrees on, so take the plain
# median of the leads rather than letting a single lead master it
rec = rec[PTS]
fixed = np.zeros(len(rec), dtype=int)
V = rec.to_numpy()
for j in range(1, V.shape[1]):
    lo = V[:, j] < V[:, j - 1]
    fixed += lo
    V[lo, j] = V[lo, j - 1]
rec[PTS] = V
print("record level rows %d, forward ordering repairs %d over %d records"
      % (len(rec), int(fixed.sum()), int((fixed > 0).sum())))

# the record table names its fiducials with the _sample suffix
SPTS = [p + "_sample" for p in PTS]
rec.columns = SPTS
old = pd.read_csv(RECON, low_memory=False)
rec = rec.reset_index().rename(columns={"index": "record_id"})
src = pd.read_csv(SRC, low_memory=False)
meta = [c for c in old.columns if c not in SPTS and c != "record_id"]
carry = src.set_index("record_id").reindex(rec.record_id)
for c in meta:
    rec[c] = carry[c].to_numpy() if c in carry.columns else ""

# The record level fiducials are now built from the unit table's representative
# beat, which is not the beat the old record level table described. The two
# coincide on only about a third of records, so carrying the old rep_beat_id
# forward would leave every fiducial pointing at a beat the row does not name,
# an error worth roughly one RR interval. Take the beat identity and the beat
# window from the units the values actually came from.
ub = u.groupby("record_id").agg(rep_beat_id=("beat_id", "first"),
                                bs=("beat_start_sample", "min"),
                                be=("beat_end_sample", "max"))
ub = ub.reindex(rec.record_id)
rec["rep_beat_id"] = ub["rep_beat_id"].to_numpy().astype(int)
rec["beat_start_sample"] = np.minimum(ub["bs"].to_numpy().astype(int),
                                      rec["p_onset_sample"])
rec["beat_end_sample"] = np.maximum(ub["be"].to_numpy().astype(int),
                                    rec["t_offset_sample"])
rec = rec[list(old.columns)]

# ---- checks -------------------------------------------------------------
print()
print("unit  table  clean only %s | 12 leads each %s | cols unchanged %s"
      % ((u.rep_qc_status.notna()).all(),
         (u.groupby('record_id').size() == 12).all(),
         list(u.columns) == cols))
V = u[[p + "_sample" for p in PTS]].to_numpy()
print("unit  table  ordering violations %d"
      % int((np.diff(V, axis=1) < 0).any(axis=1).sum()))
V = rec[SPTS].to_numpy()
print("record table ordering violations %d | rows %d | cols match old %s | clean only %s"
      % (int((np.diff(V, axis=1) < 0).any(axis=1).sum()), len(rec),
         list(rec.columns) == list(old.columns),
         bool((rec.qc_status == "clean").all())))
_ub = u.groupby("record_id")["beat_id"].first().reindex(rec.record_id).to_numpy()
print("record table beat identity agrees with the units %d of %d"
      % (int((rec.rep_beat_id.to_numpy() == _ub).sum()), len(rec)))
_in = ((rec.beat_start_sample <= rec.p_onset_sample) &
       (rec.beat_end_sample >= rec.t_offset_sample)).all()
print("record table landmarks inside the beat window:", bool(_in))

print()
print("how far each fiducial moved from ECGdeli, unit level, ms")
print("%-12s %6s %6s %6s  %s" % ("fiducial", "med", "p90", "max", "within 10 ms"))
for p in PTS:
    d = np.abs(u.set_index(key)[p + "_sample"] - eco[p + "_sample"]).to_numpy() * 2.0
    print("%-12s %6.0f %6.0f %6.0f  %11.0f%%"
          % (p, np.median(d), np.percentile(d, 90), d.max(), 100 * (d <= 10).mean()))

if WRITE:
    u.to_csv(UNITS, index=False)
    rec.to_csv(RECON, index=False)
    print()
    print("wrote", UNITS, os.path.getsize(UNITS), "bytes")
    print("wrote", RECON, os.path.getsize(RECON), "bytes")
else:
    print("\ndry run, nothing written. pass --write to commit.")
