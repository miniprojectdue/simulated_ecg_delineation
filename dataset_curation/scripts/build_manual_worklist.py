#!/usr/bin/env python3
"""
build_manual_worklist.py  -  Build require_manual_label.csv: the (record, lead) units that need
manual fiducial labelling, tiered by confidence, from master_labels.csv.

A unit (record, lead) is included when a MAJORITY (>=50%) of its beats are rule-based CRITICAL,
or when the unit carries a cross-lead fiducial flag together with at least one rule-based
critical beat. QC now rests on the cited rule-based screen and the cross-lead consistency check
only. The earlier cross-method signal is no longer used.

priority_tier
  1_critical   - majority of beats are rule-based CRITICAL (definite delineation errors), label first
  2_crosslead  - below that threshold but flagged by both a rule-based critical beat and the
                 cross-lead consistency check, lower priority
"""
import csv, os
from collections import defaultdict, Counter
csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
SRC = os.path.join(ROOT,"dataset_curation","data","assembled","master_labels.csv")
IDX = os.path.join(ROOT,"dataset_curation","data","review","signals_index.csv")
CL  = os.path.join(ROOT,"ecgdeli_labelling","data","qc","crosslead_fiducial_flags.csv")
OUT = os.path.join(ROOT,"dataset_curation","data","review","require_manual_label.csv")
PRIMARY = ("R","S","J","Ton","Tpk","Toff")   # QRS and T landmarks drive the flag; P landmarks are context

clflag = set()
for r in csv.DictReader(open(CL, newline="")):
    if any(r["flag_"+k] == "1" for k in PRIMARY): clflag.add((r["record_id"], r["lead"]))

u = defaultdict(lambda: {"tot":0,"crit":0,"dc":"","exb":"","exf":""})
for r in csv.DictReader(open(SRC)):
    d = u[(r["record_id"], r["lead"])]; d["tot"] += 1; d["dc"] = r["disease_class"]
    if r["qc_status"] == "critical":
        d["crit"] += 1
        if not d["exb"]: d["exb"], d["exf"] = r["beat_id"], r["qc_flags"]

path = {r["record_id"]: r["path_raw"] for r in csv.DictReader(open(IDX))}
rows = []
for (rid, ld), d in u.items():
    fc = d["crit"]/d["tot"]; cl = 1 if (rid, ld) in clflag else 0
    if fc < 0.5 and not (d["crit"] > 0 and cl): continue
    tier = "1_critical" if fc >= 0.5 else "2_crosslead"
    rows.append({"record_id":rid,"disease_class":d["dc"],"lead":ld,"path_raw":path.get(rid,""),
        "n_beats":d["tot"],"n_critical":d["crit"],"crosslead_flag":cl,
        "frac_critical":round(fc,2),"priority_tier":tier,
        "example_beat":d["exb"],"example_qc_flags":d["exf"]})
rows.sort(key=lambda d: (d["priority_tier"], -d["frac_critical"]))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} units, {len({r['record_id'] for r in rows})} signals")
print(dict(Counter(r["priority_tier"] for r in rows)))
