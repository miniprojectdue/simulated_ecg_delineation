#!/usr/bin/env python3
"""
build_crosslead_priority.py  -  Cross-lead QC prioritisation for the worklist.

Adds two columns to all_units_worklist.csv, in place and non-destructively:
  crosslead_flag      1 if the unit has a cross-lead fiducial flag (any QRS or T landmark
                      outside its V2/V5 tolerance), else 0
  priority_crosslead  1 = persistent rule-based problem (unit qc_status == critical)
                      2 = below the persistent threshold but carries BOTH a rule-based
                          critical beat (unit_worst_status == critical) AND a cross-lead flag
                      3 = everything else

This replaces the earlier cross-method secondary signal. QC now rests on the cited
rule-based screen plus the cross-lead consistency check only. Idempotent - re-running
just refreshes the two columns.
"""
import csv, os
csv.field_size_limit(10**7)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=HERE
while ROOT!=os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT,"ecgdeli_labelling")):
    ROOT=os.path.dirname(ROOT)
WL=os.path.join(ROOT,"manual_labelling","data","all_units_worklist.csv")
CL=os.path.join(ROOT,"ecgdeli_labelling","data","qc","crosslead_fiducial_flags.csv")
PRIMARY=("R","S","J","Ton","Tpk","Toff")   # QRS and T landmarks drive the flag; P landmarks are context

clflag=set()
with open(CL,newline="") as f:
    for r in csv.DictReader(f):
        if any(r["flag_"+k]=="1" for k in PRIMARY): clflag.add((r["record_id"],r["lead"]))

with open(WL,newline="") as f:
    rd=csv.DictReader(f); cols=[c for c in rd.fieldnames if c not in ("crosslead_flag","priority_crosslead")]
    rows=list(rd)
out=cols+["crosslead_flag","priority_crosslead"]; n={"1":0,"2":0,"3":0}
for r in rows:
    cl=1 if (r["record_id"],r["lead"]) in clflag else 0
    r["crosslead_flag"]=str(cl)
    if r.get("qc_status","")=="critical": p="1"
    elif r.get("unit_worst_status","")=="critical" and cl: p="2"
    else: p="3"
    r["priority_crosslead"]=p; n[p]+=1
tmp=WL+".tmp"
with open(tmp,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=out,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
os.replace(tmp,WL)
print(f"priority_crosslead 1={n['1']} 2={n['2']} 3={n['3']}  crosslead_flag units={len(clflag)}")
