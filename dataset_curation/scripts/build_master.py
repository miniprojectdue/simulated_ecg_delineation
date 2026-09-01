#!/usr/bin/env python3
"""
build_master.py  -  Assemble the two training-ready files for the delineation model.

Since testing is done on the separate external (MonoAlg3D) dataset, ALL MedalCare-XL data is
used for training the original test/examples splits are folded into 'train', and 'validation'
is kept as 'val' (a by-record hold-out for model selection). Beats are kept per-beat at their
real sample positions (NOT median-collapsed) — a segmentation model needs fiducial positions
aligned to the actual waveform.

Outputs (written next to this script)
  master_labels.csv   - the full per-(record, lead, beat) fiducial labels, split -> {train, val}
  signals_index.csv   - one row per signal record_id -> raw/filtered/noise waveform paths + shape

The two files pair on record_id for a labels row, load the waveform from signals_index
(path_raw), window it with beat_start_sample/beat_end_sample, and read the fiducial columns.

"""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while ROOT != os.path.dirname(ROOT) and not os.path.isfile(os.path.join(ROOT,"config","paths.yaml")):
    ROOT = os.path.dirname(ROOT)
SRC  = os.path.join(ROOT, "ecgdeli_labelling","data","primary","medalcare_fiducials_ecgdeli.csv")
MAN  = os.path.join(ROOT, "ecgdeli_labelling","data","input","medalcare_manifest.csv")

def remap(split):                       # test/examples/train -> train, validation -> val
    return "val" if split == "validation" else "train"

def mi_group(s):                        # LCX_1.0_ant -> LCX_1.0, six normalised MI groups
    if not s: return ""
    return s[:-4] if s.endswith("_ant") else s[:-5] if s.endswith("_post") else s

# manifest lookup so the MI subclass (blank in the fiducials file) is carried on every label row
mi_by_rec = {rr["record_id"]: (rr.get("mi_subclass","") or "")
             for rr in csv.DictReader(open(MAN))}

# 1) master_labels.csv  (stream line-by-line so we never load the ~0.9 GB file into memory)
with open(SRC, newline="") as fin, open(os.path.join(ROOT, "dataset_curation","data","assembled","master_labels.csv"), "w", newline="") as fout:
    r = csv.reader(fin); w = csv.writer(fout)
    header = next(r); w.writerow(header + ["mi_group"])
    si = header.index("split"); ri = header.index("record_id"); mci = header.index("mi_subclass")
    n_train = n_val = 0
    for row in r:
        row[si] = remap(row[si])
        det = mi_by_rec.get(row[ri], "")
        if det: row[mci] = det                    # fill detailed MI subclass from the manifest
        (n_val := n_val + 1) if row[si] == "val" else (n_train := n_train + 1)
        w.writerow(row + [mi_group(det)])          # append normalised six-category group
print(f"master_labels.csv  -> train {n_train:,} rows | val {n_val:,} rows")

# 2) signals_index.csv  (one row per signal, from the manifest)
cols = ["record_id","disease_class","mi_subclass","mi_group","split","fs_hz","n_samples","n_leads",
        "path_raw","path_filtered","path_noise"]
man = list(csv.DictReader(open(MAN)))
with open(os.path.join(ROOT, "dataset_curation","data","review","signals_index.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for rr in man:
        rr["split"] = remap(rr["split"]); rr["mi_group"] = mi_group(rr.get("mi_subclass","") or "")
        w.writerow({k: rr.get(k, "") for k in cols})
tr = sum(1 for rr in man if remap(rr["split"]) == "train")  # note rr['split'] already remapped above
print(f"signals_index.csv  -> {len(man):,} signals")
