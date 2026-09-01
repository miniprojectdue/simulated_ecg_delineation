# Machine Learning Enabled Extraction of Electrocardiogram Biomarkers for Drug-Induced Cardiotoxicity Screening


This folder is a **code-and-metrics** snapshot of the working project. It carries every script,
every configuration file and every evaluation metric behind Chapter 4, in the same directory
structure the project was developed in, so the pipeline can be read and re-run end to end.

It deliberately carries **no bulk data and no model weights**. The raw MedalCare-XL signal
corpus is ~27 GB, the intermediate label tables reach ~2 GB and the trained checkpoints ~1.3 GB,
none of that belongs in a Git repository(size is too big than allowed on a free account). Every one of those artefacts is instead named in
[`DATA_MAP.md`](DATA_MAP.md), which states, for each stage, the exact script that produces it,
the file it writes, and its size. Running the pipeline in the documented order regenerates them.

---

## Layout

```
config/                     paths.yaml + pipeline_settings.yaml, the single source of truth
ECG_TOOL/ECGdeli/           the ECGdeli MATLAB toolbox (third party, KIT-IBT/ECGdeli)

ecgdeli_labelling/          STEP 1  pseudo-label generation and quality control
dataset_curation/           STEP 2  assemble the label table, QC columns, review worklists
manual_labelling/           STEP 2b manual correction tool and merge-back
Gold/                       STEP 2c reviewed labels, offset fitting, de-biasing propagation
statistics/                 STEP 3  population validation against the MedalCare-XL publication
external_monoalg3d/         STEP 4  the MonoAlg3D external set and the rule-based baseline
ml_modelling/               STEP 5  the delineation network: configs, scripts, results
thesis_figures/             the Chapter 2-4 figure scripts
```

Each module keeps its own `README.md` describing that stage in detail.

### Where the module names map onto the dissertation

| Folder | Dissertation section |
| --- | --- |
| `ecgdeli_labelling/` | 3.2 Fiducial Point Labelling Framework, 3.2.2 Quality Control,  3.2.3 Cross-lead consistency |
| `dataset_curation/` | 3.2 label schema, 3.3.2 correction and propagation |
| `manual_labelling/`, `Gold/` | 3.3 Semi-supervised Manual Labelling Approach, App. A.4 review interface |
| `statistics/` | 4.2.3 Consistency with published MedalCare-XL features, App. A.3 |
| `external_monoalg3d/` | 2.4.4 rule-based baseline, 3.3.3 external test preparation, App. A.1, A.6, A.7 |
| `ml_modelling/` | 3.4 task formulation, 3.5 architecture, 3.6 training, 3.7 post-processing, 3.8-3.10 evaluation,  all of Ch. 4 |

---

## Running the pipeline

Paths below are relative to this folder, which is the repository root: scripts locate the root by
walking up until they find `config/paths.yaml`.

```bash
python3 -m venv project_version && source project_version/bin/activate

pip install -r requirements.txt
pip install torch                       # a hard requirement, not pinned in requirements.txt
```

You must supply two inputs that are not in this repository:

1. **MedalCare-XL** (~27 GB) from the published release, unpacked to
   `WP2_largeDataset_Noise/` and `WP2_largeDataset_ParameterFiles/`.
2. **MATLAB** with the bundled `ECG_TOOL/ECGdeli/` toolbox on the path, for step 1 and for the
   manual-review interfaces.

Then, in order:

```bash
# STEP 1  pseudo-labels (MATLAB) + QC
matlab -batch "run('ecgdeli_labelling/scripts/run_ecgdeli_medalcare.m')"
python3 ecgdeli_labelling/scripts/qc_review_list.py
python3 ecgdeli_labelling/scripts/crosslead_fiducial_qc.py

# STEP 2  assemble, QC columns, re-derive T onset, build the review worklists
python3 dataset_curation/scripts/build_master.py
python3 dataset_curation/scripts/add_qc_status.py
python3 dataset_curation/scripts/build_reconciled_global.py
python3 dataset_curation/scripts/build_clean_unit_labels.py --write
python3 dataset_curation/scripts/rederive_t_onset.py --write
python3 manual_labelling/scripts/build_all_worklist.py

# STEP 2c  reviewed labels, offsets, de-biasing  (after the MATLAB review passes)
python3 Gold/scripts/build_reviewed_labels.py --write
python3 Gold/scripts/fit_gold_offsets.py
python3 Gold/scripts/validate_gold_offsets.py
python3 Gold/scripts/apply_gold_offsets.py

# STEP 3  population validation
python3 statistics/scripts/build_per_signal_stats.py
python3 statistics/scripts/score_vs_table6.py

# STEP 5  the model
python3 ml_modelling/scripts/cut_training_corpus.py --write
python3 ml_modelling/scripts/selfcheck.py                      # synthetic end-to-end check
python3 ml_modelling/scripts/cache_signals.py \
    --units ml_modelling/data/pretrain_units.csv \
    --units ml_modelling/data/finetune_units.csv --workers 8
python3 ml_modelling/scripts/train.py --config ml_modelling/configs/treatments/pretrain_toffset_fix_tailweight.yaml
python3 ml_modelling/scripts/train.py --config ml_modelling/configs/treatments/finetune_toffset_fix_tailweight.yaml
python3 ml_modelling/scripts/evaluate.py --checkpoint <ckpt> --units <units.csv> --tag <name>
```

`ml_modelling/test_commands_toffsetfix_2.md` and `test_commands_tonsetfix_3.md` hold the exact
command lines used for every run reported in Chapter 4, including the `--set` overrides.

---

## The reported model

Chapter 4 reports one Stage-1 checkpoint and one Stage-2 checkpoint. Both come from the
`treatments/` configs, not from the bare `pretrain.yaml` / `finetune.yaml`, which are the base
files those inherit from.

| | Config | Checkpoint (not in this repo) |
| --- | --- | --- |
| Stage 1, pretrain | `configs/treatments/pretrain_toffset_fix_tailweight.yaml` | `checkpoints/pretrain_toffset_fix_tailweight/best.pt` (89 MB) |
| Stage 2, reported | `configs/treatments/finetune_toffset_fix_tailweight.yaml` | `checkpoints/finetune_toffset_fix_tailweight/best_geometry.pt` (88 MB) |

The Stage-2 run uses `train.freeze: encoder_bn`, the policy reported as best in Table 4.14
(boundary MAE 9.00 ms, mIoU 0.8642, seed 1337).

### Checkpoints are not in this repository

Fifteen checkpoints stand behind Chapter 4 -- the two above, three freeze policies, two extra
seeds, and eight few-shot adapters -- at roughly 88 MB each, about **1.3 GB in total**. Storing
binaries of that size in Git bloats the history permanently, since Git keeps every version and
`.pt` files do not delta-compress. They live outside the repository, backed up separately.

`DATA_MAP.md` lists every checkpoint, the config and command that produced it, and the Chapter 4
table it backs. Each is reproducible from the configs and the unit tables described there.

---

## What is in `ml_modelling/results/`

One directory per run, holding `metrics.json` (every number machine-readably, broken down by
disease class and by lead), `report.txt` (the formatted table), and `per_unit.csv` (one row per
unit with the signed error of each landmark in milliseconds -- the basis for the error
distributions and the per-morphology tables). Training runs additionally carry
`config_resolved.json`, the full merged configuration including any command-line override, plus
`train_log.txt` and `history.json`.

Only the runs behind a reported number are included. The mapping from run directory to
dissertation table is in `DATA_MAP.md`. Exploratory arms that no reported number depends on --
the non-tailweight `toffset_fix` treatments, the `nshot`/`nshot3`/`nshot4` adaptation ladders,
the `adapt_ischemia1`-`4` probes, the quantile-matching (`_qm`) external variants and the
`sigma_posthoc` soft-target analysis -- were left out of this snapshot; they remain in the
working project directory.

---

## Notes on provenance

- **The rule-based comparator.** `ml_modelling/scripts/baseline_v3.py` is a Python port of
  `external_monoalg3d/delineate_ecg_v3.m`, itself from Smith & Holmes (2024). The MATLAB
  original assumes 1000 Hz where one sample is one millisecond. the port keeps the constants in
  milliseconds and converts at the actual sampling rate, so it reduces exactly to the original at
  1000 Hz and preserves the algorithm in time rather than in samples at 500 Hz.
- **ECGdeli** (`ECG_TOOL/ECGdeli/`) is third-party, from KIT-IBT/ECGdeli, under its own licence
  (included). The 3.4 MB bundled example signal was dropped from this snapshot.
- **Recovered scripts.** `baseline_v3.py`, `conditioning_tests.py`, `selfcheck.py`,
  `verify_augment.py`, `verify_training_labels.py`, `preflight.py` and `ladder.py` were missing
  from the working tree at snapshot time and were restored from the project's Git history, where
  they are the versions that produced the reported results.
- **The external test set** (MonoAlg3D, 100 recordings from UK Biobank-derived anatomies) is not
  redistributed here. `external_monoalg3d/` holds the code that prepares and reviews it.
