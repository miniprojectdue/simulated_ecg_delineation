# manual_labelling — correcting the critical pseudo-labels

Fix the genuine ECGdeli delineation errors in the **tier-1 critical** units (4,251 record–lead units
across 2,300 recordings) so the training set is not misled. Corrected fiducials replace the ECGdeli
values and are tagged `label_source = manual_corrected`. This is training-set cleaning, not a
gold-standard test set.

## Layout

```
MANUAL_LABELLING_PROTOCOL.md              how to place each fiducial + decision rules (READ FIRST)
tool/
  medalcare_fiducial_corrector.html       browser tool: view + drag/nudge fiducials, auto-saves corrections
scripts/
  build_corrector_worklist.py             build the ordered plan + per-batch tool worklists
  merge_manual_corrections.py             fold saved corrections back into master_labels (propagate per unit)
data/
  critical_labelling_plan.md              ordering, batching, effort estimate
  require_manual_label_critical_ordered.csv   the 4,251 units in labelling order
  corrector_batches/critical_batch_NN.csv     tool-ready worklists (one representative beat per unit)
  corrections/                            put the tool's saved corrections CSVs here
```

## Workflow

1. Read `MANUAL_LABELLING_PROTOCOL.md`, then `data/critical_labelling_plan.md`.
2. (Re)generate worklists if needed `python3 scripts/build_corrector_worklist.py`
3. Open `tool/medalcare_fiducial_corrector.html` in **Chrome/Edge**
   - load a `data/corrector_batches/critical_batch_NN.csv` (input 1),
   - point **Dataset folder** at `WP2_largeDataset_Noise/` (input 2),
   - **link an auto-save CSV** into `data/corrections/` once.
4. Correct each unit's representative beat per the protocol, press `r` (reviewed), `n` (next).
5. Merge back `python3 scripts/merge_manual_corrections.py data/corrections/<file>.csv`
   — updates `dataset_curation/data/assembled/master_labels.csv` (keeps a `.bak`), propagating each
   corrected beat to all beats of its unit by R-peak alignment (T segment RR-scaled), and stamping the
   rows `manual_corrected`.

## Dependencies (kept in `dataset_curation`, since they are its outputs)

- `dataset_curation/data/review/require_manual_label_priority.csv` — the tiered review list (input).
- `dataset_curation/data/review/signals_index.csv` — record → raw-signal path.
- `dataset_curation/data/assembled/master_labels.csv` — read for beats, written by the merge step.
