#!/usr/bin/env python3
"""
build_corrector_worklist.py  -  Build the tier-1 critical manual-labelling worklists.

From the source review list (dataset_curation/data/review/require_manual_label_priority.csv) it
  1. selects the tier-1 critical units (rule-based genuine errors),
  2. orders them easy -> hard by disease class, keeping each recording's leads together,
     and batches them (~250 units, recordings never split) -> require_manual_label_critical_ordered.csv,
  3. for each unit selects a representative interior beat and pulls that beat's full row
     (fiducial *_sample columns, presence flags, beat window) from master_labels.csv, joins the raw
     signal path, and writes one beat-level worklist per batch that the corrector tool can load.

Outputs (this module's data/ folder)
  data/require_manual_label_critical_ordered.csv   (the ordered plan)
  data/corrector_batches/critical_batch_NN.csv     (tool-ready, one representative beat per unit)

"""
import os, csv
import pandas as pd

# cross-lead fiducial flag per (record,lead) - QC secondary signal
def _load_clflag(path):
    import csv as _csv
    fl=set()
    prim=("R","S","J","Ton","Tpk","Toff")
    for d in _csv.DictReader(open(path, newline="")):
        if any(d["flag_"+k]=="1" for k in prim): fl.add((d["record_id"],d["lead"]))
    return fl

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)
DC       = os.path.join(ROOT, "dataset_curation", "data")
PRIORITY = os.path.join(DC, "review", "require_manual_label.csv")
CLFILE   = os.path.join(ROOT, "ecgdeli_labelling", "data", "qc", "crosslead_fiducial_flags.csv")
SIGIDX   = os.path.join(DC, "review", "signals_index.csv")
MASTER   = os.path.join(DC, "assembled", "master_labels.csv")
MLDATA   = os.path.join(ROOT, "manual_labelling", "data")
PLAN_OUT = os.path.join(MLDATA, "require_manual_label_critical_ordered.csv")
BATCHDIR = os.path.join(MLDATA, "corrector_batches")
os.makedirs(BATCHDIR, exist_ok=True)
clflag = _load_clflag(CLFILE)

CLASS_ORDER = ["sinus", "avblock", "iab", "lae", "fam", "rbbb", "lbbb", "mi"]   # easy -> hard
LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
FKEYS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
         "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
NEED  = ["record_id","disease_class","lead","beat_id","fs_hz","n_samples","beat_start_sample",
         "beat_end_sample","p_present","qrs_present","t_present","qc_flags"] + FKEYS
TOOLCOLS = ["record_id","disease_class","lead","beat_id","fs_hz","n_samples"] + FKEYS + \
           ["p_present","qrs_present","t_present","beat_start_sample","beat_end_sample",
            "flags","also_delineator","priority","path_raw"]
TARGET_BATCH = 250

# 1) select + order + batch the tier-1 critical units
c = pd.read_csv(PRIORITY)
c = c[c.priority_tier == "1_critical"].copy()
c["_cls"] = c.disease_class.map({k: i for i, k in enumerate(CLASS_ORDER)})
c["_lead"] = c.lead.map({k: i for i, k in enumerate(LEADS)})
c = c.sort_values(["_cls", "record_id", "_lead"]).reset_index(drop=True)
c["order_index"] = range(1, len(c) + 1)
batch = []; b = 1; n = 0; prev = None
for row in c.itertuples():
    if n >= TARGET_BATCH and row.record_id != prev:
        b += 1; n = 0
    batch.append(b); n += 1; prev = row.record_id
c["batch"] = batch
plan_cols = ["batch","order_index","record_id","disease_class","lead","path_raw","n_beats",
             "frac_critical","example_beat","example_qc_flags"]
c[plan_cols].to_csv(PLAN_OUT, index=False)
print(f"ordered plan: {len(c):,} units in {c.batch.nunique()} batches -> {PLAN_OUT}")

# 2) representative interior beat per unit
def target_beat(nb):
    nb = int(nb)
    return max(2, min(nb - 1, round(nb / 2))) if nb >= 3 else 1
c["target_beat"] = c.n_beats.map(target_beat)
want = {(r.record_id, r.lead, int(r.target_beat)): (int(r.batch), int(r.order_index)) for r in c.itertuples()}
want_recs = set(r for r, _, _ in want)
path_by_rec = {r["record_id"]: r["path_raw"] for r in csv.DictReader(open(SIGIDX, newline=""))}

# 3) pull those beat rows from master_labels
rows = {}
for ch in pd.read_csv(MASTER, usecols=NEED, dtype=str, na_filter=False, chunksize=400000):
    ch = ch[ch.record_id.isin(want_recs)]
    for row in ch.itertuples(index=False):
        d = row._asdict()
        k = (d["record_id"], d["lead"], int(d["beat_id"]) if d["beat_id"] not in ("", "None") else -1)
        if k in want:
            rows[k] = d
print(f"matched {len(rows):,}/{len(want):,} representative beats")

# 4) write one tool-ready worklist per batch
recs = []
for k, (bnum, order_index) in want.items():
    d = rows.get(k)
    if d is None:
        continue
    out = {col: d.get(col, "") for col in TOOLCOLS if col in d}
    out["flags"] = (d.get("qc_flags", "") or "").replace(",", ";")
    out["also_delineator"] = "1" if (d["record_id"], d["lead"]) in clflag else "0"
    out["priority"] = 1
    out["path_raw"] = path_by_rec.get(d["record_id"], "")
    out["_batch"] = bnum; out["_order"] = order_index
    recs.append(out)
recs.sort(key=lambda r: r["_order"])
for bnum in sorted({r["_batch"] for r in recs}):
    fn = os.path.join(BATCHDIR, f"critical_batch_{bnum:02d}.csv")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TOOLCOLS, extrasaction="ignore"); w.writeheader()
        w.writerows(r for r in recs if r["_batch"] == bnum)
print(f"wrote {len({r['_batch'] for r in recs})} batch files to {BATCHDIR}  ({len(recs):,} beats)")
