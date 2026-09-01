#!/usr/bin/env python3
"""
crosslead_fiducial_qc.py - Per-fiducial cross-lead consistency QC against V2/V5.

All 12 leads of a MedalCare-XL record are simultaneous projections of one simulated
beat, so each fiducial's timing RELATIVE TO QRS ONSET (t=0, a globally synchronous
event) is comparable across leads even when the representative beat index differs.
We reference every landmark to that beat's QRS onset, take the V2/V5 mean as the
per-record reference for each landmark, and measure each lead's absolute deviation.
Tolerances are set PER FIDUCIAL at the 98th percentile of the observed deviations,
because boundary landmarks (QRS onset/offset, T-offset) are near-synchronous and tight
while peak landmarks (R, Q, S, T-peak) legitimately shift with the lead projection.
QC signal only - it changes no label.
"""
import csv, os, statistics, math
from collections import defaultdict


csv.field_size_limit(10**7)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=HERE
while ROOT!=os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT,"manual_labelling")):
    ROOT=os.path.dirname(ROOT)
SRC=os.path.join(ROOT,"manual_labelling","data","all_units_worklist.csv")
OUTDIR=os.path.join(ROOT,"ecgdeli_labelling","data","qc"); os.makedirs(OUTDIR,exist_ok=True)
OUT=os.path.join(OUTDIR,"crosslead_fiducial_flags.csv")
SUM=os.path.join(OUTDIR,"crosslead_fiducial_summary.txt")
REF_LEADS=("V2","V5")
# fiducial -> worklist column; all measured as (col - qrs_onset) in ms
FID={"Pon":"p_onset_sample","Ppk":"p_peak_sample","Poff":"p_offset_sample",
     "Q":"q_peak_sample","R":"r_peak_sample","S":"s_peak_sample",
     "J":"qrs_offset_sample","Ton":"t_onset_sample","Tpk":"t_peak_sample","Toff":"t_offset_sample"}
ORDER=["Pon","Ppk","Poff","Q","R","S","J","Ton","Tpk","Toff"]
LEADS=["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def gi(v):
    v=(v or "").strip()
    if v in ("","None","NaN","nan"): return None
    try: return float(v)
    except: return None
def pct(a,p):
    a=sorted(a); k=(len(a)-1)*p/100.0; lo=math.floor(k); hi=math.ceil(k)
    return a[int(k)] if lo==hi else a[lo]*(hi-k)+a[hi]*(k-lo)

val_of={}; cls={}
with open(SRC,newline="") as f:
    for r in csv.DictReader(f):
        rec=r["record_id"]; lead=r["lead"]; cls[rec]=r.get("disease_class","")
        ms=1000.0/(gi(r.get("fs_hz")) or 500.0)
        qon=gi(r.get("qrs_onset_sample"))
        d={}
        if qon is not None:
            for fid,col in FID.items():
                v=gi(r.get(col))
                if v is not None: d[fid]=(v-qon)*ms
        val_of[(rec,lead)]=d

recs=defaultdict(list)
for (rec,lead) in val_of: recs[rec].append(lead)

# pass 1: reference per record per fiducial (mean of available V2/V5), collect deviations
devs={f:[] for f in ORDER}
per_unit=[]  # (rec,lead,cls, {fid:dev})
for rec,leads in recs.items():
    ref={}
    for fid in ORDER:
        vs=[val_of[(rec,rl)][fid] for rl in REF_LEADS if (rec,rl) in val_of and fid in val_of[(rec,rl)]]
        if vs: ref[fid]=sum(vs)/len(vs)
    for lead in leads:
        dd={}
        for fid,v in val_of[(rec,lead)].items():
            if fid in ref:
                dv=abs(v-ref[fid]); dd[fid]=dv; devs[fid].append(dv)
        per_unit.append((rec,lead,cls[rec],dd))

# per-fiducial tolerance = 98th percentile of deviations
TOL={f:round(pct(devs[f],98)) for f in ORDER if devs[f]}

# pass 2: flags
by_lead_tot=defaultdict(int); by_lead_flag=defaultdict(lambda:defaultdict(int))
by_cls_tot=defaultdict(int);  by_cls_flag=defaultdict(lambda:defaultdict(int))
nflag_any=0
with open(OUT,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["record_id","lead","disease_class"]+["dev_"+x for x in ORDER]+["flag_"+x for x in ORDER]+["any_flag"])
    for rec,lead,c,dd in per_unit:
        row=[rec,lead,c]; flags={}
        for fid in ORDER:
            row.append("" if fid not in dd else round(dd[fid],1))
        anyf=0
        for fid in ORDER:
            fl=1 if (fid in dd and fid in TOL and dd[fid]>TOL[fid]) else 0
            flags[fid]=fl
            if fl: anyf=1
        for fid in ORDER: row.append(flags[fid])
        row.append(anyf)
        w.writerow(row)
        by_lead_tot[lead]+=1; by_cls_tot[c]+=1
        for fid in ORDER:
            if flags[fid]: by_lead_flag[fid][lead]+=1; by_cls_flag[fid][c]+=1
        if anyf: nflag_any+=1

N=len(per_unit)
lines=[]
lines.append("Per-fiducial cross-lead consistency QC   reference = V2/V5 mean, anchor = QRS onset")
lines.append(f"(record,lead) units evaluated : {N}")
lines.append("")
lines.append(f"{'fiducial':8s} {'tol_ms':>7s} {'median':>7s} {'p95':>6s} {'p98':>6s} {'flag%':>7s}   {'category':>9s}")
for fid in ORDER:
    a=devs[fid]
    if not a: continue
    fr=sum(by_lead_flag[fid].values())/N*100
    med=statistics.median(a); p95=pct(a,95); p98=pct(a,98)
    cat="boundary" if fid in ("Pon","Poff","J","Ton","Toff") else "peak"
    lines.append(f"{fid:8s} {TOL[fid]:7.0f} {med:7.1f} {p95:6.1f} {p98:6.1f} {fr:6.2f}%   {cat:>9s}")
lines.append("")
lines.append("flag% by fiducial by lead (worst leads):")
for fid in ORDER:
    if not devs[fid]: continue
    rates=sorted(((by_lead_flag[fid][l]/by_lead_tot[l]*100 if by_lead_tot[l] else 0, l) for l in LEADS), reverse=True)
    top=", ".join(f"{l} {r:.1f}%" for r,l in rates[:4])
    lines.append(f"  {fid:5s}: {top}")
lines.append("")
lines.append("R-peak flag% by disease class:")
for c,t in sorted(by_cls_tot.items(), key=lambda kv:-by_cls_flag['R'][kv[0]]/max(kv[1],1)):
    lines.append(f"  {c:8s}: {by_cls_flag['R'][c]/t*100:5.2f}%  ({by_cls_flag['R'][c]}/{t})")
lines.append("")
lines.append(f"any-fiducial flag: {nflag_any} ({nflag_any/N*100:.2f}%)")
txt="\n".join(lines)
open(SUM,"w").write(txt+"\n")
print(txt)
