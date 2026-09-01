#!/usr/bin/env python3
"""
build_edited_global.py  -  CORRECTED-PIPELINE variant of build_reconciled_global.py.

Same job as before: turn the per-lead labels in the MANUALLY-CORRECTED master_labels.csv
into ONE global delineation target per signal. What changed (2026-07-25) is HOW the 12
leads are reconciled, because the old rule produced targets that were not usable as an
ML segmentation target.

    build_reconciled_global.py  -> reads ecgdeli primary CSV  (original, from-scratch)
    build_edited_global.py      -> reads master_labels.csv     (after your corrections)

--------------------------------------------------------------------------------------
WHY THE RECONCILIATION CHANGED
--------------------------------------------------------------------------------------
The old rule was onset = min across leads, offset = max
across leads. That may be right in principle (a wave begins when it first appears in ANY
lead) but it is a min/max over 12 NOISY per-lead estimates, so it does not estimate the
union, it estimates the union plus the tail of the delineator's error. Measured over all
16,848 records on the representative beat, 

    rule                        median P    median QRS   P ends after     QRS ends after
                                duration    duration     QRS starts       T starts
    per-lead ECGdeli (ref)        120 ms      126 ms          -                 -
    min/max across leads          252 ms      170 ms        88.2 %            39.7 %
    25th/75th pct + ordering      172 ms      144 ms         0.0 %             0.0 %

An 88 % rate of P-offset landing after QRS-onset is not a physiological finding, it is
the estimator blowing outward, and a target where the segments overlap cannot be learned
by a segmentation model. ECGdeli's own synchronised output (ecgdeli_sync_*) has the same
problem (median P duration 216 ms), which is why it was not used as the target either.

The overlap is NOT purely an artefact of aggregating: ECGdeli's own per-lead labels put
p_offset at or past qrs_onset inside a SINGLE lead in 29.9% of the 194,680 clean units
(median within-lead PQ gap 12 ms, 5th percentile -16 ms). The union does not create the
problem, it amplifies it from 30% to 88%, so no aggregation rule can fix it on its own
and the ordering constraint below has to be explicit. The ST side is clean by comparison
(median within-lead gap 70 ms, never negative), which is why QRS/T overlap is rarer.

WHERE THE WITHIN-LEAD OVERLAP COMES FROM (traced to the delineator, 2026-07-25)
It is a documented design limit of ECGdeli. Two things in
ECG_TOOL/ECGdeli combine:
  1. P_Detection.m ends by clamping the P-offset ONLY against FPT column 5, the Q PEAK:
         if p_pos(i,7) > local_FPT(i,5), p_pos(i,7) = local_FPT(i,5) - 30ms
     Column 4 is the QRS ONSET and is never consulted, so a P-offset is free to sit
     anywhere up to the Q peak, i.e. inside the QRS. In this dataset the Q peak sits a
     near-constant 26 ms after the QRS onset (p25 = p75 = 26 ms), which is exactly the
     amount of intrusion observed. Measured over the 2,546,990 clean-unit beats:
         p_offset <= q_peak      99.80 %   <- the guarantee ECGdeli actually enforces
         p_offset <= qrs_onset   71.02 %   <- the guarantee a segmentation target needs
  2. Check_Position_ECG_Waves.m, ECGdeli's own ordering validator, only tests columns
     2, 6 and 11 (P peak < R peak < T peak). No boundary is ever order-checked.
ECGdeli also never surfaces the problem in its own feature table: ecgdeli_pq_ms is the PQ
INTERVAL (qrs_onset - p_onset; mean |diff| vs our recomputation = 0.0000 ms), not the PQ
segment, so it stays positive (99.99%) no matter how far the P-offset runs on.

The proximate cause is an over-wide P wave, not a mis-timed QRS. Per-lead P duration is
median 134 ms (p75 170, p95 210), against a normal adult P wave of 80-110 ms; 72.0% of
beats exceed the 120 ms clinical threshold. The between-lead scatter is P-offset-dominated
too: median across-lead range 88 ms for p_offset vs 38 ms for qrs_onset, and no lead shows
a systematic QRS-onset bias (per-lead median offset 0 ms in all 12). If the P-offset were
pulled back so that no P wave exceeded 110 ms, within-lead overlap would fall from 32.1%
to 6.6%.

WHY HEALTHY SINUS LOOKS WORST
The overshoot is roughly constant, so the overlap rate is set by how much PQ segment a
class has to absorb it, and healthy sinus has the least of any class in the dataset:

    class     median PR   median P dur   median PQ segment   beats overlapping
    avblock     286 ms       128 ms           96 ms                1.4 %
    mi          246 ms       154 ms           24 ms               18.2 %
    lbbb        206 ms       126 ms            6 ms               40.7 %
    iab         202 ms       130 ms            2 ms               47.6 %
    lae         210 ms       140 ms            2 ms               49.3 %
    fam         192 ms       124 ms            2 ms               49.9 %
    sinus       188 ms       124 ms           -2 ms               57.2 %
    rbbb        190 ms       124 ms           -2 ms               57.7 %

Every other class either prolongs AV conduction (avblock), widens the P wave (lae, iab) or
sits on a remodelled substrate (mi), so all of them buy headroom that normal sinus rhythm
does not have. Sinus is not labelled worse; it has no margin. This is worth one paragraph
in the write-up: the manual corrections concentrated on sinus records are correcting a
systematic delineator bias, not random noise, which is also why the ordering constraint has
to stay in the pipeline even after correction.

Rejecting outlier leads and keeping a true min/max was tried and is not better: filtering
to |v - median| <= k*MAD before the union throws away ~13 of the 72 per-record boundary
values and still leaves 69% overlap (k=2) or 75% (k=3); trimming the two most extreme
leads per side lands at 65%. The plain 25th/75th percentile beats all of them and is
simpler to state, so that is what is used.

--------------------------------------------------------------------------------------
THE RULE NOW
--------------------------------------------------------------------------------------
    R-peak     = MASTER, from reference channel lead II          [definitive beat anchor]
    onset      = QUANT-th percentile across leads                [robust 'earliest']
    offset     = (1-QUANT)-th percentile across leads            [robust 'latest']
    P/Q/S/T pk = median across leads                             [robust central estimate]
    then ordering is enforced: p_on <= p_off <= qrs_on <= qrs_off <= t_on <= t_off,
    the QRS window is widened if needed to contain the master R-peak, and each peak is
    clamped inside its own wave's window.

REPAIR = 'qrs_wins' (default). QRS onset/offset are the highest-SNR landmarks in the
record and the ones the model most needs to be right, so the repair never moves them: a
P-offset that runs past QRS-onset is clipped back to it, and a T-onset before QRS-offset
is pushed out to it. The alternative, REPAIR='midpoint', splits each crossing down the
middle, which drags QRS-onset 4 ms later on average across the whole dataset. Both leave
zero overlaps; the difference is which landmark absorbs the correction:

    p25 + repair      median P / QRS / T / QT      mean displacement of qrs_on
    qrs_wins           168 / 147 / 206 / 392 ms              0.0 ms
    midpoint           172 / 144 / 206 / 388 ms              4.0 ms

QUANT = 0.25 by default. It is a bias-corrected union, not a central estimate: at 0.25
the global window is still genuinely wider than any single lead (P 172 vs 120 ms, QRS
144 vs 126 ms, T 206 vs 186 ms), it just no longer chases the worst lead. Sensitivity, if
you need it for the write-up (median P duration / % of records with an implausible P
duration / % with overlapping segments, before ordering is enforced):

    QUANT   0.00(min/max)  0.05    0.10    0.20    0.25    0.33    0.50(median)
    P dur      252 ms     233 ms  214 ms  188 ms  176 ms  159 ms    138 ms
    P implaus   79.5 %     71.9 %  59.1 %  40.6 %  33.0 %  21.6 %     3.7 %
    overlap     88.2 %     84.9 %  78.1 %  65.4 %  59.5 %  47.6 %    26.0 %

Only leads listed in the manual worklist (UNITS, default final_data_units.csv) take part
in the reconciliation, so leads that were QC-excluded and that you will therefore never
see or correct can no longer set a global boundary on their own. 13,086 of 16,848 records
have all 12 leads in the worklist; 67 have none. Those 67 are DROPPED from the output by
default (their fiducials could only come from leads QC rejected and that nobody will
review) and listed in dropped_no_clean_units.csv next to the output; --keep-fallback
puts them back, reconciled over all 12 leads and marked recon_source='all_leads_fallback'.

--------------------------------------------------------------------------------------
Since each MedalCare-XL record is one simulated beat repeated ~13x, we output ONE ROW PER
SIGNAL using a representative (median) beat.

Other differences vs build_reconciled_global.py (plumbing only)
    * input   = master_labels.csv (carries your merged corrections)
    * is_crit = per-beat  qc_status == 'critical'  (from master, no separate QC list)
    * path_raw = looked up from signals_index.csv (master has no path column)
    * output  = reconciled_global_fiducials_corrected.csv  (never overwrites the original)

Run
    python3 build_edited_global.py
    python3 build_edited_global.py <MASTER_CSV> <OUT_CSV> [SIGNALS_INDEX_CSV] [UNITS_CSV]
Options (anywhere on the command line)
    --quantile=0.25     onset percentile (offsets use 1-q). 0.0 = the old min/max rule.
    --all-leads         ignore the worklist, reconcile over all 12 leads (old behaviour)
    --keep-fallback     keep the records that have no clean lead (dropped by default)
    --repair=qrs_wins   ordering repair policy: qrs_wins (default) | midpoint | none
    --no-order          alias for --repair=none (diagnostics only, leaves overlaps in)
Out
    reconciled_global_fiducials_corrected.csv   (one row per signal, global fiducials + QC)
"""
import pandas as pd, numpy as np, csv, os, sys

# ---- option flags (pulled out before positional args) ----
argv = sys.argv[1:]
QUANT      = 0.25
USE_UNITS  = True
REPAIR     = "qrs_wins"
DROP_FALLBACK = True
_pos = []
for a in argv:
    if a.startswith("--quantile="): QUANT = float(a.split("=", 1)[1])
    elif a.startswith("--repair="): REPAIR = a.split("=", 1)[1]
    elif a == "--all-leads":       USE_UNITS = False
    elif a == "--keep-fallback":   DROP_FALLBACK = False
    elif a == "--no-order":        REPAIR = "none"
    elif a.startswith("--"):       sys.exit("unknown option %s" % a)
    else: _pos.append(a)
if not (0.0 <= QUANT <= 0.5):
    sys.exit("--quantile must be in [0, 0.5] (it is the ONSET percentile; offsets use 1-q)")
if REPAIR not in ("qrs_wins", "midpoint", "none"):
    sys.exit("--repair must be qrs_wins, midpoint or none")
ENFORCE_ORDER = REPAIR != "none"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT, "config", "paths.yaml")):
    ROOT = os.path.dirname(ROOT)

# --- inputs / output (defaults follow the repo layout, override via argv) ---
MASTER = os.path.join(ROOT, "dataset_curation", "data", "assembled", "master_labels.csv")
SIGIDX = os.path.join(ROOT, "dataset_curation", "data", "review", "signals_index.csv")
OUT    = os.path.join(ROOT, "dataset_curation", "data", "global", "reconciled_global_fiducials_corrected.csv")
UNITS  = os.path.join(ROOT, "manual_labelling", "data", "final_data_units.csv")
if len(_pos) > 0: MASTER = _pos[0]
if len(_pos) > 1: OUT    = _pos[1]
if len(_pos) > 2:
    SIGIDX = _pos[2]
else:
    # if only MASTER was overridden, try to find signals_index next to it
    cand = os.path.join(os.path.dirname(os.path.dirname(MASTER)), "review", "signals_index.csv")
    if os.path.isfile(cand): SIGIDX = cand
if len(_pos) > 3: UNITS = _pos[3]

ONS  = ["p_onset_sample", "qrs_onset_sample", "t_onset_sample"]
OFFS = ["p_offset_sample", "qrs_offset_sample", "t_offset_sample"]
PKS  = ["p_peak_sample", "q_peak_sample", "r_peak_sample", "s_peak_sample", "t_peak_sample"]
PRES = ["p_present", "qrs_present", "t_present"]
GLOBAL_FIDS = ["p_onset_sample","p_peak_sample","p_offset_sample","qrs_onset_sample","q_peak_sample",
               "r_peak_sample","s_peak_sample","qrs_offset_sample","t_onset_sample","t_peak_sample","t_offset_sample"]
# left-to-right ordering the global target must satisfy
BOUNDS = ["p_onset_sample","p_offset_sample","qrs_onset_sample","qrs_offset_sample","t_onset_sample","t_offset_sample"]

# ---- 1. load per-lead labels from the CORRECTED master + flag critical beats ----
use = ["record_id","lead","beat_id","disease_class","mi_subclass","split","fs_hz","n_samples",
       "beat_start_sample","beat_end_sample","qc_status"] + ONS+OFFS+PKS + PRES
f = pd.read_csv(MASTER, usecols=use, na_values=["","None"],
                dtype={"lead": "category", "qc_status": "category",
                       "disease_class": "category", "split": "category",
                       "mi_subclass": "str"})
# critical flag now comes straight from master's per-beat qc_status (no separate QC list)
f["is_crit"] = (f["qc_status"].astype(str) == "critical").astype(int)

# ---- 2. genuine-problem units (record,lead) with >=50% of beats critical ----
u = f.groupby(["record_id","lead"], observed=True).agg(nb=("beat_id","size"), nc=("is_crit","sum")).reset_index()
u["genuine"] = u["nc"]/u["nb"] >= 0.5
gen = u[u["genuine"]]
problem_leads = gen.groupby("record_id")["lead"].apply(lambda s: ";".join(sorted(s))).rename("problem_leads")
rec_genuine  = set(gen["record_id"])
rec_anycrit  = set(f.loc[f["is_crit"] > 0, "record_id"])

# ---- 2b. restrict the reconciliation to the leads you actually review ----
# The worklist (final_data_units.csv) is the clean tier: the (record, lead) units that
# survive QC and that the MATLAB corrector shows you. A lead that is not in it is a lead
# you will never inspect, so it must not be allowed to set a global boundary by itself.
# Records with no worklist lead at all keep all 12 leads and are flagged in recon_source.
fc = f
n_all = len(f)
if USE_UNITS and os.path.isfile(UNITS):
    wl = pd.read_csv(UNITS, usecols=["record_id","lead"], dtype=str)
    wl_set = set(wl["record_id"] + "\x1f" + wl["lead"])
    key = f["record_id"].astype(str) + "\x1f" + f["lead"].astype(str)
    mask = key.isin(wl_set)
    kept_recs = set(f.loc[mask, "record_id"].unique())
    missing = sorted(set(f["record_id"].unique()) - kept_recs)
    # keep every beat of a worklist lead, plus every beat of records that have none
    if missing:
        mask = mask | f["record_id"].isin(missing)
    fc = f[mask]
    print("worklist %s: %d units, reconciling over %d of %d master rows (%d records fall back to all leads)"
          % (os.path.relpath(UNITS, ROOT), len(wl_set), len(fc), n_all, len(missing)))
    fallback_recs = set(missing)
else:
    if USE_UNITS:
        print("WARNING: worklist not found at %s - reconciling over all 12 leads" % UNITS)
    fallback_recs = set()

# ---- 3. reconcile every beat across the participating leads -> global fiducials ----
# Beat identity is anchored to a MASTER R-peak from a single reference channel (lead II).
# Onsets/offsets are robust outer percentiles (see the module docstring for why min/max
# was dropped), secondary P/Q/S/T peaks the median across leads.
REF_LEAD = "II"
gb = fc.groupby(["record_id","beat_id"], observed=True)
parts = [gb[ONS].quantile(QUANT),
         gb[OFFS].quantile(1.0 - QUANT),
         gb[PKS].median(),
         gb[PRES].max(),
         gb["beat_start_sample"].min().rename("beat_start_sample"),
         gb["beat_end_sample"].max().rename("beat_end_sample"),
         gb.size().rename("n_recon_leads")]
recon = pd.concat(parts, axis=1).reset_index()
# override the global R-peak with the master reference-channel R-peak (lead II if it is
# in the worklist for this record, otherwise the across-lead median already computed)
masterR = (fc.loc[fc["lead"].astype(str) == REF_LEAD, ["record_id","beat_id","r_peak_sample"]]
             .rename(columns={"r_peak_sample":"r_master"}))
recon = recon.merge(masterR, on=["record_id","beat_id"], how="left")
recon["r_ref_used"] = recon["r_master"].notna().astype(int)
recon["r_peak_sample"] = recon["r_master"].fillna(recon["r_peak_sample"])
recon = recon.drop(columns=["r_master"])

# ---- 3b. make the target a valid segmentation ----
# A percentile over noisy per-lead estimates can still cross (P-offset after QRS-onset).
# Collapse any crossing to its midpoint, left to right, then widen the QRS window if the
# master R-peak fell outside it, then clamp each peak inside its own wave.
n_fixed = 0
if ENFORCE_ORDER:
    before = recon[BOUNDS].to_numpy(dtype=float).copy()
    if REPAIR == "midpoint":
        arr = recon[BOUNDS].to_numpy(dtype=float)
        for i in range(arr.shape[1]-1):
            cross = arr[:, i] > arr[:, i+1]          # NaN comparisons are False -> untouched
            mid = (arr[:, i] + arr[:, i+1]) / 2.0
            arr[cross, i] = mid[cross]; arr[cross, i+1] = mid[cross]
        for i in range(arr.shape[1]-1):              # fmax ignores NaN, np.maximum would spread it
            arr[:, i+1] = np.fmax(arr[:, i+1], arr[:, i])
        recon[BOUNDS] = arr
    # the QRS window must contain the master R-peak, and under qrs_wins it is the anchor
    # nothing else is allowed to move, so it is fixed first and the neighbours yield to it
    r = recon["r_peak_sample"].to_numpy(dtype=float)
    recon["qrs_onset_sample"]  = np.fmin(recon["qrs_onset_sample"].to_numpy(dtype=float), r)
    recon["qrs_offset_sample"] = np.fmax(recon["qrs_offset_sample"].to_numpy(dtype=float), r)
    recon["qrs_offset_sample"] = np.fmax(recon["qrs_offset_sample"], recon["qrs_onset_sample"])
    recon["p_offset_sample"] = np.fmin(recon["p_offset_sample"], recon["qrs_onset_sample"])
    recon["p_onset_sample"]  = np.fmin(recon["p_onset_sample"],  recon["p_offset_sample"])
    recon["t_onset_sample"]  = np.fmax(recon["t_onset_sample"],  recon["qrs_offset_sample"])
    recon["t_offset_sample"] = np.fmax(recon["t_offset_sample"], recon["t_onset_sample"])
    n_fixed = int((~np.isclose(before, recon[BOUNDS].to_numpy(dtype=float), equal_nan=True)).any(axis=1).sum())
    # peaks live inside their own wave
    for pk, on, off in [("p_peak_sample","p_onset_sample","p_offset_sample"),
                        ("q_peak_sample","qrs_onset_sample","qrs_offset_sample"),
                        ("s_peak_sample","qrs_onset_sample","qrs_offset_sample"),
                        ("t_peak_sample","t_onset_sample","t_offset_sample")]:
        recon[pk] = np.fmin(np.fmax(recon[pk], recon[on]), recon[off])

for c in GLOBAL_FIDS+["beat_start_sample","beat_end_sample"]:
    recon[c] = recon[c].round()

# ---- 4. one row per signal the representative (median beat_id) beat ----
med = f.groupby("record_id")["beat_id"].median().rename("medb").reset_index()
recon = recon.merge(med, on="record_id")
recon["d"] = (recon["beat_id"] - recon["medb"]).abs()
rep = recon.sort_values("d").groupby("record_id", as_index=False).first().drop(columns=["d","medb"])
rep = rep.rename(columns={"beat_id":"rep_beat_id"})

# ---- 5. attach metadata, QC status, priority, path ----
meta = f.drop_duplicates("record_id").set_index("record_id")[["disease_class","mi_subclass","split","fs_hz","n_samples"]]
rep = rep.merge(meta, on="record_id").merge(problem_leads, on="record_id", how="left")
rep["problem_leads"] = rep["problem_leads"].fillna("")
rep["n_problem_leads"] = rep["problem_leads"].apply(lambda s: 0 if s=="" else s.count(";")+1)
rep["genuine"] = rep["record_id"].isin(rec_genuine).astype(int)
def status(r):
    if r.record_id in rec_genuine: return "critical"
    if r.record_id in rec_anycrit: return "minor"
    return "clean"
rep["qc_status"] = rep.apply(status, axis=1)
rep["recon_source"] = np.where(rep["record_id"].isin(fallback_recs), "all_leads_fallback",
                               "worklist_clean" if USE_UNITS else "all_leads")
rep["recon_quantile"] = QUANT

# ---- 5b. drop records that have no clean lead at all ----
# Their global fiducials could only be built from leads QC rejected, so the target would be
# a reconciliation of labels nobody will ever review. Dropped by default; --keep-fallback
# puts them back. The IDs are written next to OUT so the exclusion is auditable.
n_dropped = 0
if DROP_FALLBACK and fallback_recs:
    drop_mask = rep["record_id"].isin(fallback_recs)
    n_dropped = int(drop_mask.sum())
    if n_dropped:
        dpath = os.path.join(os.path.dirname(OUT), "dropped_no_clean_units.csv")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        (rep.loc[drop_mask, ["record_id","disease_class","split","n_recon_leads"]]
            .assign(reason="no lead in " + os.path.basename(UNITS))
            .sort_values("record_id").to_csv(dpath, index=False))
        rep = rep[~drop_mask].copy()
# path_raw from signals_index.csv (master has no path column)
sig = pd.read_csv(SIGIDX, usecols=["record_id","path_raw"], dtype=str, na_filter=False)
path = dict(zip(sig["record_id"], sig["path_raw"]))
rep["path_raw"] = rep["record_id"].map(path)
cw = {"lbbb":3,"mi":3,"rbbb":3}
rep["review_priority"] = rep.apply(lambda r: (cw.get(r.disease_class,1) if r.qc_status=="critical" else 0)
                                             + r.n_problem_leads, axis=1)

COLS = (["record_id","disease_class","mi_subclass","split","fs_hz","n_samples","path_raw",
         "rep_beat_id","beat_start_sample","beat_end_sample"] + GLOBAL_FIDS + PRES +
        ["qc_status","genuine","n_problem_leads","problem_leads","review_priority",
         "n_recon_leads","recon_source","recon_quantile","r_ref_used"])
rep = rep[COLS].sort_values(["qc_status","review_priority"], ascending=[True,False])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
rep.to_csv(OUT, index=False)

# ---- report ----
n = len(rep); vc = rep["qc_status"].value_counts()
ms = 1000.0/float(rep["fs_hz"].median() or 500)
pdur   = (rep["p_offset_sample"]  - rep["p_onset_sample"])  * ms
qrsdur = (rep["qrs_offset_sample"]- rep["qrs_onset_sample"])* ms
tdur   = (rep["t_offset_sample"]  - rep["t_onset_sample"])  * ms
qt     = (rep["t_offset_sample"]  - rep["qrs_onset_sample"])* ms
print(f"source: {MASTER}")
print(f"wrote {OUT}")
print(f"rule: onset=p{QUANT:.2f} / offset=p{1-QUANT:.2f} across "
      f"{'worklist (clean) leads' if USE_UNITS else 'all leads'}"
      f"{', ordering repair=' + REPAIR if ENFORCE_ORDER else ', ordering NOT enforced'}")
print(f"signals (rows): {n}")
print(f"  leads per record used: median {rep['n_recon_leads'].median():.0f}, "
      f"min {rep['n_recon_leads'].min():.0f}; fallback records: {(rep.recon_source=='all_leads_fallback').sum()}")
if n_dropped:
    print(f"  dropped {n_dropped} records with no clean lead "
          f"-> {os.path.relpath(os.path.join(os.path.dirname(OUT),'dropped_no_clean_units.csv'), ROOT)}")
elif not DROP_FALLBACK:
    print("  --keep-fallback: records with no clean lead were KEPT")
if ENFORCE_ORDER:
    print(f"  beats whose boundaries needed an ordering repair: {n_fixed}")
print(f"  median durations (ms): P {pdur.median():.0f}  QRS {qrsdur.median():.0f}  "
      f"T {tdur.median():.0f}  QT {qt.median():.0f}")
print(f"  implausible: QRS outside 40-200ms {(~qrsdur.between(40,200)).mean()*100:.2f}%, "
      f"QT outside 250-700ms {(~qt.between(250,700)).mean()*100:.2f}%")
print(f"  segment overlaps remaining: "
      f"P/QRS {(rep.p_offset_sample>rep.qrs_onset_sample).sum()}, "
      f"QRS/T {(rep.qrs_offset_sample>rep.t_onset_sample).sum()}")
for s in ["critical","minor","clean"]:
    print(f"  {s:8s}: {vc.get(s,0)}  ({100*vc.get(s,0)/n:.1f}%)")
print("critical (genuine) by class:")
print(rep[rep.qc_status=='critical'].disease_class.value_counts().to_string())
