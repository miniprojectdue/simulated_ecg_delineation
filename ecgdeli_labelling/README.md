# ecgdeli_labelling — STEP 1 delineate + quality-control + cross-check

Generates the fiducial (P/QRS/T on/peak/off) labels for the MedalCare-XL simulated 12-lead ECGs
with **ECGdeli** — the same toolbox the dataset authors used for their Table 6 feature validation —
then quality-controls them with a rule-based screen and a cross-lead consistency check.

MedalCare-XL ships **signals only** (no per-beat fiducials), so these labels are training
**pseudo-labels, not ground truth** because the paper's own features were also produced by ECGdeli,
the paper comparison (in `../statistics`) is a **consistency** check, not an accuracy check.

> This step is **not rerun routinely**. `data/primary/medalcare_fiducials_ecgdeli.csv` is the fixed
> source that every downstream module (`../dataset_curation`, `../statistics`) recomputes from.

## Folder layout

```
scripts/
  run_ecgdeli_medalcare.m     MATLAB driver: delineate every record -> primary fiducials CSV
  ecgdeli_config_sweep.m      MATLAB: T-end / preprocessing sweep (diagnostic)
  qc_review_list.py           rule-based QC: tier every beat critical / minor / clean
data/
  input/      medalcare_manifest.csv                 (record_id, disease class, raw/filtered/noise paths)
  primary/    medalcare_fiducials_ecgdeli.csv        (one row per record x lead x beat — THE source table)
  qc/         medalcare_qc_review_list.csv, medalcare_qc_record_summary.csv
logs/         ecgdeli_failures.log
```

## Run order

| # | Script | Reads | Writes |
|---|--------|-------|--------|
| 1 | `run_ecgdeli_medalcare.m` | `data/input/medalcare_manifest.csv` + raw signals | `data/primary/medalcare_fiducials_ecgdeli.csv` |
| 2 | `qc_review_list.py`       | `data/primary/…fiducials…csv` | `data/qc/medalcare_qc_review_list.csv`, `…_record_summary.csv` |

Population validation vs Table 6 / the preprint lives in `../statistics`, assembly of the training
set and the manual-review worklists lives in `../dataset_curation`.

## Requirements

- MATLAB + the ECGdeli toolbox (https://github.com/KIT-IBT/ECGdeli), cloned to `../ECG_TOOL/ECGdeli`
- Python 3 with `pandas numpy`

Paths auto-resolve every Python script walks up to the repo root (the folder holding `config/paths.yaml`)
`run_ecgdeli_medalcare.m` resolves the repo root two levels up and only needs the ECGdeli clone in place.

## QC tiers (`qc_review_list.py`)

Thresholds are in `../config/pipeline_settings.yaml` (QRS 40–200 ms, QT 250–700 ms, PR 80–400 ms
boundary-inversion tolerance 3 samples, gross 20 samples). A `(record, lead)` unit is a **genuine**
problem when ≥50 % of its beats are flagged. Counts on the current corpus critical 104,171 beats /
minor 477,300 / clean 2,064,733.

## Cross-lead consistency

The secondary QC signal is the cross-lead fiducial consistency screen, which flags a (record, lead)
unit whose QRS or T landmarks fall outside their calibrated tolerance against the V2/V5 within-record
reference. It lives in `../manual_labelling/scripts/build_crosslead_priority.py`.

## Conventions / format facts

- MedalCare CSVs are **leads × samples** (12 × 5000, no header), ECGdeli needs samples × leads, so the
  driver transposes on load. 500 Hz, amplitudes in mV, 2 ms/sample.
- Three signal versions per ECG `_raw` (noise-free, used here), `_filtered`, `_noise`.
- Fiducials are **0-based** sample indices, `*_ms = sample × 1000 / 500`.
- ECGdeli returns P/QRS/T onset-peak-offset **and** Q/R/S peaks natively. FPT column map
  `1 P-on  2 P-pk  3 P-off  4 QRS-on  5 Q  6 R  7 S  8 QRS-off  10 T-on  11 T-pk  12 T-off`.
