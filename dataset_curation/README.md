# dataset_curation — STEP 2 assemble the training set + QC + review worklist

Turns the fixed `../ecgdeli_labelling/data/primary/medalcare_fiducials_ecgdeli.csv` into the
model-ready training data for the global (12-lead → one delineation) ECG delineation model.
All MedalCare-XL data is used for **training**, the model is **tested on the separate internal
(MonoAlg3D) dataset**, so there is no MedalCare test split.

## Folder layout

```
scripts/   build_master.py, add_qc_status.py, add_crosscheck_qc.py,
           build_manual_worklist.py, build_reconciled_global.py, extract_signal.py
data/
  assembled/   master_labels.csv                       (the training labels)
  global/      reconciled_global_fiducials.csv         (one delineation per signal, leads reconciled)
  review/      signals_index.csv, require_manual_label.csv, require_manual_label_priority.csv
  audit_samples/  one extracted example record (signal + labels + lead-II plot)
```

The manual correction of the flagged units (worklists, protocol, browser tool, merge-back) lives in its
own module, **`../manual_labelling/`**, which reads `require_manual_label_priority.csv` + `master_labels.csv`
from here and writes the corrected fiducials back into `master_labels.csv`.

## Files

| File | Rows | What it is |
|---|---|---|
| `data/assembled/master_labels.csv` | 2,646,204 | Full fiducial labels one row per (record, lead, beat), **70 columns**, with `split` ∈ {train, val}, MI subclass (`mi_subclass` detailed + `mi_group` six-category), and per-beat QC (`qc_status`, `qc_flags`, `xmethod_flag`, `xmethod_issues`, `needs_review`). |
| `data/review/signals_index.csv` | 16,848 | One row per signal `record_id` → raw/filtered/noise waveform paths, `fs_hz`, `n_samples`, `n_leads`, class, split. |
| `data/global/reconciled_global_fiducials.csv` | 16,848 | One delineation per signal (earliest onset / latest offset / median peak across leads). |
| `data/review/require_manual_label.csv` | 51,033 | The (record, lead) units that need manual labelling, tiered by priority. |
| `build_master.py` | — | Regenerates the labels + index from the primary fiducials CSV + manifest. |
| `add_qc_status.py` | — | Tier-1 per-beat rule-based QC columns. |
| `build_manual_worklist.py` | — | Builds `require_manual_label*.csv` from `master_labels.csv`. |
| `build_reconciled_global.py` | — | Builds the global reconciled delineation table. |
| `extract_signal.py` | — | Pull one signal + its labels (+ optional plot) by `record_id`. |

## Per-beat QC (`qc_status`, `qc_flags`)

Each beat is labelled **critical / minor / clean** (reproducing `qc_review_list.py`)

- **critical** (104,171 beats, 3.9%) — a hard problem QRS-internal order violated, a boundary inversion > 20 samples, QRSdur outside 40–200 ms, QT outside 250–700 ms, PR outside 80–400 ms, or R missing. `qc_flags` lists the reasons (e.g. `QRSdur=204ms;gross_boundary:32smp`).
- **minor** (477,300, 18.0%) — a moderate boundary wobble (3–20 samples), no critical reason.
- **clean** (2,064,733, 78.0%) — nothing beyond a trivial (≤ 3-sample) wobble.

These let you filter/downweight/route beats for training (e.g. train on clean+minor, send critical
to manual review, or use them for a robustness split). The 104,171 critical beats span 2,300
signals — mostly MI/LBBB/RBBB. Run `add_qc_status.py` to regenerate the columns.

## Cross-method QC (`xmethod_flag`, `xmethod_issues`, `needs_review`) — RETIRED

Tier 1 above is ECGdeli checking *itself* (structural + physiological plausibility). Tier 2 was once a
second-opinion cross-method check. It has been retired and its generator removed. The `xmethod_*`
columns survive only as dead columns in `master_labels.csv` and feed nothing downstream. The live
secondary signal is the cross-lead consistency screen in `../manual_labelling`.

- **`xmethod_flag`** — 1 if an independent tool disagrees with ECGdeli on a P/Q/R/S/T landmark for that beat (608,176 beats), else 0.
- **`xmethod_issues`** — the specific disagreements (e.g. `t_peak:-208ms;t_offset:-66ms`).
- **`needs_review`** — 1 if `qc_status=='critical'` OR `xmethod_flag==1` (682,155 beats) — the comprehensive review pool.

Why this matters — the two tiers cross-tabulate as

| | agree (xmethod=0) | disagree (xmethod=1) |
|---|---|---|
| **clean** | 1,609,540 (trustworthy — two methods concur) | 455,193 (**plausible-but-wrong** passed the rules but an independent tool disagrees) |
| **minor** | 354,509 | 122,791 |
| **critical** | 73,979 | 30,192 (**highest priority** — both signals fire) |

The tier was retired on time grounds and its generator removed, so the columns are frozen and
cannot be regenerated. They are kept only so that older tables still parse. Prioritisation of the
manual-labelling effort now comes from the rule-based screen plus the cross-lead consistency check.

## Manual-labelling worklist (`require_manual_label.csv`)

The units to manually label, at the **(record, lead)** level — a unit is included only when a
**majority (≥50%)** of its beats are flagged (by the rule-based *or* the cross-method signal), so
lone transient glitches on the otherwise-clean repeated beat are excluded. **51,033 units across
14,594 signals**, sorted worst-first and tiered

| `priority_tier` | Units | Meaning |
|---|---|---|
| `1_critical` | 4,251 | Majority of beats are rule-based **critical** — definite delineation errors. **Label these first.** |
| `2_both` | 9,541 | Flagged by **both** the rule-based and independent cross-method signals. |
| `3_xmethod` | 37,241 | Majority flagged **only** by the cross-method check — weaker (often low-amplitude P/T ambiguity). |

Columns `record_id, disease_class, lead, path_raw, n_beats, n_critical, n_xmethod, n_needs,
frac_critical, frac_xmethod, frac_needs, priority_tier, example_beat, example_qc_flags,
example_xmethod_issues`. Load `path_raw` (or `extract_signal.py <record_id>`), review the flagged
`lead`, and correct the fiducials from `master_labels.csv`. Regenerate with `build_manual_worklist.py`.

## Split convention

- Original `test` and `examples` splits are folded into **`train`** (everything is used for training).
- Original `validation` is kept as **`val`** — a **by-record** hold-out for model selection / early stopping (splitting by record, not by beat, prevents near-duplicate beats leaking between train and val).
- Result `train` = 14,414 signals (2,260,488 label rows), `val` = 2,434 signals (385,716 rows).

## How the two files pair

They join on **`record_id`**. For a labels row
1. Look up `record_id` in `signals_index.csv` → get `path_raw` (the 12 × 5000 waveform, mV, 500 Hz).
2. Load that waveform, the beat lives at `beat_start_sample … beat_end_sample`.
3. The fiducial positions are the `*_sample` columns (P/QRS/T onset–peak–offset), and the intervals (QT etc.) are the `*_ms` / `ecgdeli_*` columns.

For a global model you use all 12 leads of the signal as input and predict one delineation
`master_labels.csv` carries the per-lead fiducials so you can build the target either per lead
or reconciled across leads.

## Important labels are per beat, not median-collapsed

Beats are kept at their **real sample positions**. A segmentation model maps waveform → per-sample
regions, so the target must be fiducial positions aligned to the actual signal — a median over
beats is a scalar summary with no waveform to attach to (that median is only for population
statistics, see the Statistics folder). If you want to reduce the ~13×-per-signal redundancy,
select one **real interior beat** per record (a real waveform window + its real fiducials) rather
than synthesising a median beat, or feed the whole 10 s strip as one sequence.
