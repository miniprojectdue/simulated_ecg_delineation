import csv, statistics, json, os
from collections import Counter, defaultdict
csv.field_size_limit(10**7)

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


ROOT   = _repo_root()
MASTER = os.path.join(ROOT, "dataset_curation/data/assembled/master_labels.csv")
FINAL  = os.path.join(ROOT, "manual_labelling/data/final_data_units.csv")
OUTDIR = "/sessions/rcw-01eznxqjqyb87mcdkyb8z92s/scratch"

reviewable = set()
for r in csv.DictReader(open(FINAL, newline="")):
    reviewable.add((r["record_id"], r["lead"]))
rev_by_rec = defaultdict(set)
for (rr, ll) in reviewable:
    rev_by_rec[rr].add(ll)
EMPTY = frozenset()
print("reviewable units:", len(reviewable), " records:", len(rev_by_rec), flush=True)

ONS  = ["p_onset_sample", "qrs_onset_sample", "t_onset_sample"]
OFFS = ["p_offset_sample", "qrs_offset_sample", "t_offset_sample"]
MS   = 2.0   # 500 Hz

f = open(MASTER, newline="")
rd = csv.reader(f)
hdr = next(rd)
I = {c: i for i, c in enumerate(hdr)}
i_rec, i_lead, i_beat = I["record_id"], I["lead"], I["beat_id"]
i_cls, i_qc = I["disease_class"], I["qc_status"]
COLI = {c: I[c] for c in ONS + OFFS}

def num(v):
    v = v.strip()
    if v in ("", "None", "NaN", "nan"):
        return None
    try:
        return float(v)
    except ValueError:
        return None

n_rec = 0
rec_with_issue = 0
per_fid = Counter()                 # fiducial -> n records whose boundary is set solely by a non-reviewable lead
per_fid_any = Counter()             # fiducial -> n records where the boundary was resolvable at all
unit_drives = defaultdict(list)     # (record,lead) -> [(fiducial, margin_ms)]
unit_meta = {}
leads_missing_hist = Counter()

def flush(block):
    global n_rec, rec_with_issue
    if not block:
        return
    n_rec += 1
    if n_rec % 2000 == 0:
        print(f'  ...{n_rec} records scanned', flush=True)
    rec = block[0][i_rec]
    beats = [int(float(r[i_beat])) for r in block]
    med = statistics.median(beats)
    rep = min(set(beats), key=lambda b: (abs(b - med), b))
    reprows = [r for r in block if int(float(r[i_beat])) == rep]
    present = {r[i_lead] for r in block}
    leads_missing_hist[len(present - rev_by_rec.get(rec, EMPTY))] += 1
    hit = False
    for col, mode in [(c, "min") for c in ONS] + [(c, "max") for c in OFFS]:
        vals = []
        for r in reprows:
            v = num(r[COLI[col]])
            if v is not None:
                vals.append((v, r[i_lead], r))
        if len(vals) < 2:
            continue
        per_fid_any[col] += 1
        vals.sort(key=lambda t: t[0], reverse=(mode == "max"))
        best = vals[0][0]
        attain = [t for t in vals if t[0] == best]
        if len(attain) != 1:
            continue                                    # tie -> correcting one lead changes nothing
        lead = attain[0][1]
        if (rec, lead) in reviewable:
            continue                                    # you will see this lead anyway
        runner = next((t[0] for t in vals if t[0] != best), None)
        margin = abs(runner - best) * MS if runner is not None else 0.0
        per_fid[col] += 1
        unit_drives[(rec, lead)].append((col, margin))
        unit_meta[(rec, lead)] = (attain[0][2][i_cls], attain[0][2][i_qc])
        hit = True
    if hit:
        rec_with_issue += 1

block = []
key = None
for row in rd:
    k = row[i_rec]
    if k != key:
        flush(block); block = []; key = k
    block.append(row)
flush(block)
f.close()

units = []
for (rec, lead), drv in unit_drives.items():
    cls, qc = unit_meta[(rec, lead)]
    units.append({
        "record_id": rec, "lead": lead, "disease_class": cls, "qc_status": qc,
        "n_boundaries_driven": len(drv),
        "fiducials": ";".join(sorted(c for c, _ in drv)),
        "max_margin_ms": round(max(m for _, m in drv), 1),
    })
units.sort(key=lambda u: (-u["max_margin_ms"], -u["n_boundaries_driven"]))

with open(os.path.join(OUTDIR, "boundary_driving_units.csv"), "w", newline="") as g:
    w = csv.DictWriter(g, fieldnames=list(units[0].keys()) if units else
                       ["record_id","lead","disease_class","qc_status",
                        "n_boundaries_driven","fiducials","max_margin_ms"])
    w.writeheader(); w.writerows(units)

summary = {
    "records_in_master": n_rec,
    "records_with_a_boundary_set_by_an_unreviewable_lead": rec_with_issue,
    "boundary_driving_units_total": len(units),
    "per_fiducial_records": dict(per_fid),
    "per_fiducial_resolvable": dict(per_fid_any),
    "units_by_margin": {
        ">=  0 ms (any)": len(units),
        ">=  4 ms":  sum(1 for u in units if u["max_margin_ms"] >= 4),
        ">= 10 ms":  sum(1 for u in units if u["max_margin_ms"] >= 10),
        ">= 20 ms":  sum(1 for u in units if u["max_margin_ms"] >= 20),
        ">= 40 ms":  sum(1 for u in units if u["max_margin_ms"] >= 40),
    },
    "units_by_qc_status": dict(Counter(u["qc_status"] for u in units)),
    "units_by_class": dict(Counter(u["disease_class"] for u in units)),
    "units_by_lead": dict(Counter(u["lead"] for u in units).most_common()),
}
with open(os.path.join(OUTDIR, "boundary_audit_summary.json"), "w") as g:
    json.dump(summary, g, indent=2)
print(json.dumps(summary, indent=2), flush=True)
print("DONE", flush=True)
