#!/usr/bin/env python3
"""
score_vs_table6.py  -  Score any ECGdeli fiducials file against the paper's Table 6.

Extracts the six healthy-cohort timing biomarkers (and, if signals are given, the
five amplitudes) per lead from a fiducials CSV, compares the per-lead means to the
Table 6 'sim' column, and prints a single distance-to-paper score. Run it on the
output of each ECGdeli preprocessing config (see ecgdeli_config_sweep.m) and keep
the config with the smallest score.

Usage
    python3 score_vs_table6.py FIDUCIALS.csv [--class sinus] [--signals /path/to/MedalCareXL Data]
"""
import csv, sys, argparse, os
import numpy as np
from collections import defaultdict

# Table 6 'sim' (healthy) means per lead
T6_TIMING = {  # [Pdur,QRSdur,Tdur,PQint,QTint,RRint]
 "I":[124.06,131.31,178.12,128.07,310.54,758.15],"II":[128.09,126.10,182.33,127.18,317.08,758.02],
 "III":[164.52,126.80,183.16,171.88,306.94,757.99],"aVR":[127.42,128.81,179.38,126.19,318.73,757.97],
 "aVL":[154.50,128.62,182.76,169.37,299.19,758.08],"aVF":[141.00,125.02,184.05,142.60,310.43,758.06],
 "V1":[140.78,129.05,180.72,160.57,303.89,758.06],"V2":[155.93,136.12,176.15,181.65,287.71,757.90],
 "V3":[154.23,132.80,179.01,174.20,285.50,758.01],"V4":[140.60,127.32,180.05,148.55,290.09,758.03],
 "V5":[128.69,123.89,177.39,126.73,310.74,758.12],"V6":[123.44,126.55,174.63,118.63,320.51,758.06]}
TFEAT=["Pdur","QRSdur","Tdur","PQint","QTint","RRint"]
LEADS=list(T6_TIMING)

ap=argparse.ArgumentParser()
ap.add_argument("fiducials"); ap.add_argument("--klass",default="sinus")
ap.add_argument("--signals",default=None,help="dataset root (enables amplitude scoring)")
ap.add_argument("--manifest",default="ecgdeli_labelling/data/input/medalcare_manifest.csv")
a=ap.parse_args()

def gi(v): return int(v) if v not in ("","None") else None
data=defaultdict(list)
hdr=None
for i,r in enumerate(csv.DictReader(open(a.fiducials))):
    if r.get("disease_class")!=a.klass: continue
    g=lambda k:gi(r.get(k+"_sample",""))
    data[(r["record_id"],r["lead"])].append((int(r["beat_id"]),g("p_onset"),g("p_offset"),
        g("qrs_onset"),g("qrs_offset"),g("t_onset"),g("t_offset"),g("r_peak")))
ms=2.0
ours={ld:{f:[] for f in TFEAT} for ld in LEADS}
for (rid,ld),bl in data.items():
    if ld not in LEADS: continue
    bl.sort(); Rs=[b[7] for b in bl if b[7] is not None]
    for i in range(1,len(Rs)): ours[ld]["RRint"].append((Rs[i]-Rs[i-1])*ms)
    for (_,pon,poff,qon,qoff,ton,toff,rp) in bl:
        if pon and poff: ours[ld]["Pdur"].append((poff-pon)*ms)
        if qon and qoff: ours[ld]["QRSdur"].append((qoff-qon)*ms)
        if ton and toff: ours[ld]["Tdur"].append((toff-ton)*ms)
        if pon and qon: ours[ld]["PQint"].append((qon-pon)*ms)
        if qon and toff: ours[ld]["QTint"].append((toff-qon)*ms)

print(f"scoring {a.fiducials}  (class={a.klass})\n")
print(f"{'feature':8s} {'mean|Δ| (ms)':>14s}")
total=0; nfeat=0
for j,f in enumerate(TFEAT):
    ds=[]
    for ld in LEADS:
        if ours[ld][f]: ds.append(abs(np.mean(ours[ld][f])-T6_TIMING[ld][j]))
    if ds:
        md=np.mean(ds); print(f"{f:8s} {md:12.1f}"); total+=md; nfeat+=1
print(f"\nTIMING distance-to-paper (mean of the six mean|Δ|): {total/nfeat:.1f} ms")
print("(lower is closer; QT dominates when T-end is misaligned)")
