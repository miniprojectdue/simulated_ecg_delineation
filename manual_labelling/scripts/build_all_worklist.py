import os, csv
csv.field_size_limit(10**7)
ROOT   = os.path.expanduser("~/mnt/Delineation")
DC     = os.path.join(ROOT, "dataset_curation", "data")
MASTER = os.path.join(DC, "assembled", "master_labels.csv")
SIGIDX = os.path.join(DC, "review", "signals_index.csv")
PRIOR  = os.path.join(DC, "review", "require_manual_label.csv")
CLFILE = os.path.join(ROOT, "ecgdeli_labelling", "data", "qc", "crosslead_fiducial_flags.csv")
OUT    = os.path.join(ROOT, "manual_labelling", "data", "all_units_worklist.csv")
FK = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
      "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
KEEP = ["record_id","disease_class","lead","beat_id","fs_hz","n_samples","beat_start_sample",
        "beat_end_sample","p_present","qrs_present","t_present","qc_status","qc_flags"] + FK
QIDX = KEEP.index("qc_status")
RANK = {"critical":0,"minor":1,"clean":2}
def worst(ss): return min(ss, key=lambda s: RANK.get(s,3))

path={}
for d in csv.DictReader(open(SIGIDX, newline="")): path[d["record_id"]]=d.get("path_raw","")
# pipeline tier per (record,lead)
tier={}
for d in csv.DictReader(open(PRIOR, newline="")):
    tier[(d["record_id"],d["lead"])] = d["priority_tier"]
def tier_status(k):
    t=tier.get(k,"")
    return "critical" if t=="1_critical" else ("minor" if t=="2_crosslead" else "clean")
# cross-lead fiducial flag per (record,lead) — QC secondary signal
CLPRIMARY=("R","S","J","Ton","Tpk","Toff")
clflag=set()
for d in csv.DictReader(open(CLFILE, newline="")):
    if any(d["flag_"+k]=="1" for k in CLPRIMARY): clflag.add((d["record_id"],d["lead"]))

saved=[]
def flush(buf,key):
    n=len(buf); t=(min(max(round(n/2)-1,1),n-2) if n>=3 else 0)
    saved.append((buf[t], worst([b[QIDX] for b in buf]), tier_status(key)))
print("streaming...", flush=True)
with open(MASTER, newline="") as f:
    r=csv.reader(f); hdr=next(r); ix={n:i for i,n in enumerate(hdr)}
    ri,li=ix["record_id"],ix["lead"]; keepix=[ix[c] for c in KEEP]
    cur=None; buf=[]; nrows=0
    for row in r:
        nrows+=1; k=(row[ri],row[li])
        if k!=cur:
            if buf: flush(buf,cur)
            cur=k; buf=[]
        buf.append([row[j] for j in keepix])
    if buf: flush(buf,cur)
print("rows",nrows,"units",len(saved), flush=True)

OUTCOLS=["record_id","disease_class","lead","beat_id","fs_hz","n_samples"]+FK+ \
        ["p_present","qrs_present","t_present","beat_start_sample","beat_end_sample",
         "flags","also_delineator","priority","path_raw","qc_status","rep_qc_status","unit_worst_status",
         "crosslead_flag","priority_crosslead"]
pr={"critical":"1","minor":"2","clean":"3"}
CLS={k:i for i,k in enumerate(["sinus","avblock","iab","lae","fam","rbbb","lbbb","mi"])}
LEADS={k:i for i,k in enumerate(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])}
out=[]
for vals,ws,ts in saved:
    d={c:v for c,v in zip(KEEP,vals)}
    o={c:d.get(c,"") for c in ["record_id","disease_class","lead","beat_id","fs_hz","n_samples"]+FK+
       ["p_present","qrs_present","t_present","beat_start_sample","beat_end_sample"]}
    o["flags"]=(d.get("qc_flags","") or "").replace(",",";")
    cl = 1 if (d["record_id"],d["lead"]) in clflag else 0
    o["also_delineator"]=str(cl); o["crosslead_flag"]=str(cl)
    o["priority"]=pr.get(ts,"3"); o["path_raw"]=path.get(d["record_id"],"")
    o["qc_status"]=ts; o["rep_qc_status"]=d.get("qc_status",""); o["unit_worst_status"]=ws
    o["priority_crosslead"]="1" if ts=="critical" else ("2" if (ws=="critical" and cl) else "3")
    o["_s"]=(CLS.get(d.get("disease_class",""),99),RANK.get(ts,9),d["record_id"],LEADS.get(d.get("lead",""),99))
    out.append(o)
out.sort(key=lambda x:x["_s"])
with open(OUT,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=OUTCOLS,extrasaction="ignore"); w.writeheader()
    for o in out: w.writerow(o)
from collections import Counter
print("WROTE",len(out),"qc_status(tier):",dict(Counter(o["qc_status"] for o in out)), flush=True)
