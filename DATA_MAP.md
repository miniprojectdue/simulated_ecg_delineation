# Data map

Every artefact the pipeline produces, the script that writes it, and where it lives. Nothing in
this table is stored in this repository -- it is all either bulk data or model weights. Running
the scripts in the order below regenerates all of it.

Sizes are from the working project directory at snapshot time.

---

## Inputs you must supply

| Input | Size | Source |
| --- | --- | --- |
| `WP2_largeDataset_Noise/` | 27 GB | MedalCare-XL published release. Raw 12x5000 waveforms, 16,848 usable records. |
| `WP2_largeDataset_ParameterFiles/` | 198 MB | MedalCare-XL published release. Simulation parameter files. |
| MonoAlg3D external corpus | ~11 MB | Not redistributed. 100 recordings, 1000 Hz, UK Biobank-derived anatomies (Smith 2025; Holmes 2025). |

---

## STEP 1 -- `ecgdeli_labelling/` : pseudo-label generation and QC

| Script | Writes | Size |
| --- | --- | --- |
| `run_ecgdeli_medalcare.m` | `data/input/medalcare_manifest.csv` | 8.6 MB |
| `run_ecgdeli_medalcare.m` | `data/primary/medalcare_fiducials_ecgdeli.csv` -- 2,646,204 beat-lead rows, the master fiducial table | **891 MB** |
| `qc_review_list.py` | `data/qc/medalcare_qc_review_list.csv` -- per-row clean/minor/critical tier | 11.5 MB |
| `qc_review_list.py` | `data/qc/medalcare_qc_record_summary.csv` | 1.2 MB |
| `crosslead_fiducial_qc.py` | `data/qc/crosslead_fiducial_flags.csv` -- 202,176 record-lead units vs the V2/V5 reference | 21.7 MB |
| `crosslead_fiducial_qc.py` | `data/qc/crosslead_fiducial_summary.txt` | *included in repo* |
| `ecgdeli_config_sweep.m` | ECGdeli parameter sweep, exploratory | -- |

Backs: **Table 4.1** (QC tiers), **Figure 4.1** (cross-lead deviations), **Table A.2** (thresholds).


---

## STEP 2 -- `dataset_curation/` : assemble the label table

| Script | Reads | Writes | Size |
| --- | --- | --- | --- |
| `build_master.py` | manifest + master fiducials | `data/assembled/master_labels.csv` | **960 MB** |
| `build_master.py` | | `data/review/signals_index.csv` -- record_id to waveform path | 4.3 MB |
| `add_qc_status.py`, `add_crosscheck_qc.py` | master_labels + QC lists | QC columns merged in place | -- |
| `build_reconciled_global.py` | master_labels | `data/global/reconciled_global_fiducials.csv` -- record-level reconciled labels | 1.8 MB |
| `build_manual_worklist.py` | reconciled global | `data/review/require_manual_label.csv` | 1.1 MB |
| `build_clean_unit_labels.py --write` | worklist + measured fiducials | `data/assembled/clean_units_labels.csv` -- 194,645 clean units, one per record-lead at its representative beat | 63 MB |
| `apply_qrs_polarity.py`, `rename_qrs_polarity.py` | | Q/R/S deflection naming by polarity relative to the QRS-onset baseline | -- |
| `extract_signal.py` | | single-record waveform extraction helper | -- |


---

## STEP 2b -- `manual_labelling/` : the correction pass

| Script / tool | Writes | Size |
| --- | --- | --- |
| `scripts/build_all_worklist.py` | `data/all_units_worklist.csv` | 43 MB |
| `scripts/build_corrector_worklist.py` | reviewer worklists | -- |
| `scripts/build_crosslead_priority.py` | flagged units prioritised for review | -- |
| `tool/medalcare_label_ecg.m` | `corrections/*_corrections.csv` -- the reviewer's placements | -- |
| `scripts/merge_manual_corrections.py` | corrections merged back into the label table | -- |
| `scripts/apply_deletions.py` | applies reviewer deletions | -- |
| `audit/boundary_audit.py` | `audit/boundary_audit_summary.json` | *included in repo* |
| -- | `data/final_data_units.csv` | 23 MB |
| -- | `data/qrs_morphology.csv` | 17 MB |

Backs: **Appendix A.4** (the review interface), **Section 3.3.1**.

---

## STEP 2c -- `Gold/` : reviewed labels, offset fitting, de-biasing

| Script | Writes | Size |
| --- | --- | --- |
| `build_gold_worklist.py` / `_clean.py` | the calibration and test review worklists | -- |
| `tool/gold_label_ecg.m` | `corrections/gold_worklist_{calibration,test}_corrections.csv` | -- |
| `build_reviewed_labels.py --write` | `data/gold_reviewed_labels.csv` -- the only human-signed-off labels; the fine-tuning set | 1.4 MB |
| `fit_gold_offsets.py` | `data/gold_offsets.csv` + `results/fit_gold_offsets.txt` -- per-landmark, per-class median offsets from the calibration half | *txt included* |
| `validate_gold_offsets.py` | `data/gold_offsets_validated.csv` -- global vs per-class scope decided on the held-out half | -- |
| `crossvalidate_gold_offsets.py` | `results/crossvalidate_gold_offsets.txt` | *included* |
| `apply_gold_offsets.py` | `data/reconciled_global_debiased.csv` -- de-biased copy; never overwrites the canonical file | 1.8 MB |
| `apply_gold_offsets.py` | `data/gold_excluded_record_ids.csv` -- the 800 gold records, removed from every training split | -- |
| `measure_all.py` | `measured_units.jsonl` -- signal-derived fiducial estimates | -- |

Backs: **Equation 3.9** (the correction estimate), **Table 3.4** (data allocation).

Both halves of the gold set leave the training splits: the test half for the obvious reason, the
calibration half because its reviewed labels determined the offsets applied to every other record.

---

## STEP 3 -- `statistics/` : population validation

| Script | Writes | Size |
| --- | --- | --- |
| `build_per_signal_stats.py` | `data/per_signal_median.csv` | 18 MB |
| `score_vs_table6.py` | comparison against MedalCare-XL Table 6 | -- |
| `build_gaussian_overlays.py`, `build_comparison_figure.py` | `figures/` | 1.2 MB |
| `build_amplitudes_*.py`, `build_fig6_*.py` | amplitude and timing figures | -- |
| `build_apd_vs_qt.py` | `data/apd_by_run.csv` | 2 MB |
| `reproduce_paper_stats.{py,m}` | the published-statistics reproduction | -- |

Backs: **Table 4.3**, **Figures 4.2-4.3**, **Tables A.3-A.4** (per-lead reproduction).

---

## STEP 4 -- `external_monoalg3d/` : the external set and the rule-based baseline

| Script | Purpose | Writes |
| --- | --- | --- |
| `delineate_ecg_v3.m` | the rule-based comparator, Smith & Holmes (2024) | fiducials directly |
| `run_delineation_v3.m` | batch driver for the above | -- |
| `apply_boundary_rule.m` | the 5%-of-wave-height spatial-magnitude boundary rule | -- |
| `manual_label_ecg.m` | the multi-lead review interface for the external set | `labels/smith2026_manual_corrections.csv` |
| `build_reference_worklist.m` | the review worklist | `smith2026_worklist.csv` |
| `export_test_set_500hz.m` | 1000 -> 500 Hz decimation, 65-tap windowed-sinc anti-alias, centred convolution | `test_export/` |
| `build_external_units.py --write` | the model-ready external table, all 12 leads reviewed | `smith2026_test_units_qrsanchor.csv` (1,200 units) |
| `make_labelfree_external_units.py` | the label-free evaluation variant | -- |
| `repair_labels.py`, `repeat_review.m` | review repair passes | -- |

Backs: **Section 3.3.3**, **Appendix A.6** (reference-label construction), **A.7** (resampling).

Landmark timings are kept in milliseconds at the original 1000 Hz for error calculation, so
resampling adds no quantisation error.

---

## STEP 5 -- `ml_modelling/` : the delineation network

### Corpus construction

| Script | Writes | Size |
| --- | --- | --- |
| `cut_training_corpus.py --write` | `data/pretrain_units.csv` -- 185,022 units, 15,976 records, 158,446 train / 26,576 val | 62 MB |
| `cut_training_corpus.py --write` | `data/finetune_units.csv` -- 429 reviewed units, 401 records, 275 train / 45 val / 109 test | 132 KB |
| `cut_training_corpus.py --write` | `data/held_out_record_ids.csv` -- 804 record ids no training split may contain | 28 KB |
| `verify_training_labels.py` | consistency check over the tables above | -- |
| `cache_signals.py` | `data/signal_cache/` -- one `.npy` per record, resumable | **3.7 GB** |
| `preflight.py`, `verify_augment.py` | prove the structural transforms fire before a long run | -- |
| `selfcheck.py` | synthetic end-to-end check, seconds on CPU | -- |

### Training and evaluation

| Script | Role |
| --- | --- |
| `train.py` | one driver for both stages. writes `config_resolved.json` beside every log |
| `evaluate.py` | score a checkpoint on any units table -> `metrics.json`, `report.txt`, `per_unit.csv` |
| `model.py` | the 1-D U-Net, FiLM lead conditioning, bottleneck attention, skip gates, P-observability head |
| `dataset.py` | crop, target construction, robust scaling, magnitude and validity channels |
| `losses.py` | masked class- and boundary-weighted cross entropy + soft Dice |
| `augmentations.py` | the four structural transforms |
| `postprocess.py` | posteriors -> regions -> landmarks, with the Q/R/S naming rule |
| `biomarkers.py` | QRS duration, QT, T duration, T-peak-to-T-end |
| `metrics.py` | segmentation and fiducial metrics, and the report table |
| `baseline_v3.py` | the rule-based comparator, ported to Python |
| `run_baseline.py`, `run_baseline_indist.py`, `compare_baseline.py` | baseline runs and the paired comparison |
| `ablation_conditioning.py`, `conditioning_tests.py` | the lead-query perturbation sweep |
| `ensemble.py`, `evaluate_ensemble.py` | posterior averaging over checkpoints |
| `paired.py` | paired, record-clustered bootstrap over two per-unit tables |
| `build_quantile_match.py`, `sigma_posthoc.py`, `ladder.py` | analyses not reported in the final text |

### Adaptation probe

| Script | Role |
| --- | --- |
| `adaptation_probe/make_adaptation_split.py` | splits the 100 external records into two disjoint halves |
| `adaptation_probe/nshot/make_nshot_splits.py` | per class (20 records): 10 fixed test, 2 val, 8 adaptation pool; N in {1,2,5,8} draws the first N |
| `adaptation_probe/nshot/run_nshot.sh` | trains and scores each N against the identical 600-unit test set |
| `adaptation_probe/cv_harness/` | cross-validation harness |

---

## Checkpoints

All under `ml_modelling/checkpoints/<run>/`, ~88 MB per `.pt`, **~1.3 GB total. 

| Checkpoint | Produced by | Backs |
| --- | --- | --- |
| `pretrain_toffset_fix_tailweight/best.pt` | `configs/treatments/pretrain_toffset_fix_tailweight.yaml` | Stage 1; Table 4.14 row 1 |
| `finetune_toffset_fix_tailweight/best_geometry.pt` | `configs/treatments/finetune_toffset_fix_tailweight.yaml` | **the reported model**; Tables 4.2-4.13 |
| `ablate_freeze_encoder_bn/best_geometry.pt` | as above, `--set train.freeze=encoder_bn` | Table 4.14 (identical to the reported run) |
| `ablate_freeze_none/best_geometry.pt` | `--set train.freeze=none` | Table 4.14 |
| `ablate_freeze_encoder/best_geometry.pt` | `--set train.freeze=encoder` | Table 4.14 |
| `ablate_seed_2024/best_geometry.pt` | `--set run.seed=2024` | Table 4.14 |
| `ablate_seed_7/best_geometry.pt` | `--set run.seed=7` | Table 4.14 |
| `nshot2_adapt_N{1,2,5,8}/best.pt` | label-supervised few-shot ladder | Table 4.13, "Few-shot, labels" |
| `nshot5_adapt_N{1,2,5,8}/best.pt` | distilled few-shot ladder | Table 4.13, "Few-shot, distilled" |

---

## Results directory -> dissertation table

Everything below **is** in this repository, under `ml_modelling/results/`.

| Run directory | Units | Backs |
| --- | --- | --- |
| `indist_toffsetfix2_tailweight` | 109 | **Tables 4.2, 4.4-4.9** -- the in-distribution headline (6-boundary MAE 9.00 ms, 5-peak 4.60 ms, mIoU 0.8642) |
| `zeroshot_ext_qrs_off2` | 1,200 | **Tables 4.2, 4.4, 4.5, 4.7, 4.8, 4.10** -- external zero-shot (4-boundary MAE 18.60 ms) |
| `baseline_indist` | 109 | **Table 4.11** -- rule-based baseline, unmodified and P-blanked |
| `baseline_external`, `baseline_external_perlead`, `baseline_external_qrs`, `baseline_ext_qrs_full` | 1,200 | external baseline runs |
| `baseline_heldout50` | 600 | **Table 4.12** -- baseline rows |
| `zeroshot_heldout50_2` | 600 | **Table 4.12** -- zero-shot rows (4-boundary 17.61 ms) |
| `nshot2_N0/N1/N2/N5/N8` | 600 | **Table 4.12** -- "Few-shot, labels" |
| `nshot5_N1/N2/N5/N8` | 600 | **Table 4.12** -- "Few-shot, distilled" (17.61 -> 16.66 -> 15.38 -> 14.35 ms) |
| `ablate_stage1_indist` / `ablate_stage1_external` | 109 / 1,200 | **Table 4.14** row 1 (9.84 ms in-dist; 16.61 ms external) |
| `ablate_freeze_encoder_indist` | 109 | **Table 4.14** -- encoder frozen (9.46 ms) |
| `ablate_freeze_none_indist` | 109 | **Table 4.14** -- nothing frozen (9.14 ms) |
| `ablate_freeze_encoder_bn_indist` | 109 | **Table 4.14** -- encoder+BN frozen, the reported policy (9.00 ms) |
| `ablate_seed_2024_indist` | 109 | **Table 4.14** -- seed 2024 (8.68 ms) |
| `ablate_seed_7_indist` | 109 | **Table 4.14** -- seed 7 (8.92 ms) |
| `ablate_ensemble_indist` / `ablate_ensemble_external` | 109 / 1,200 | **Table 4.14** -- posterior ensemble of the three seeds (8.92 ms; 18.68 ms external) |
| `ablate_conditioning_indist` / `ablate_conditioning_external` | 109 / 1,200 | **Section 4.6** -- lead-query perturbation (8.99 vs 9.60 ms) |
| `pretrain_toffset_fix_tailweight`, `finetune_toffset_fix_tailweight`, `ablate_*` (training dirs) | -- | `config_resolved.json`, `train_log.txt`, `history.json`, validation curves |
