# MedalCare-XL fiducial corrector — MATLAB tool

`medalcare_label_ecg.m` is a MATLAB manual corrector for the ECGdeli
pseudo-labels. It reads a **worklist CSV** (one representative beat per
`(record, lead)` unit) and writes corrections in the schema that
`merge_manual_corrections.py` consumes, so the merge-back works unchanged. It
shows the disease class and a class/flag-specific reminder for every unit and
adds a present/absent toggle for whole waves.

It works with two kinds of worklist

- **`all_units_worklist.csv`** — *the current one*. Every `(record, lead)` unit
  in the dataset (202,176 units across all 8 classes and all three QC tiers),
  each carrying a `qc_status` of **critical / minor / clean**. Use this to
  correct the flagged units and to spot-check the classifier on the rest.
- **`critical_batch_NN.csv`** — the older tier-1-critical batches. Still open
  fine, the QC filter simply greys out (those files have no `qc_status`).

## What you are correcting

Each worklist row is one `(record, lead)` unit reduced to **one representative
interior beat** (the positional-middle beat) with its 11 ECGdeli fiducials. You
correct that one beat, `merge_manual_corrections.py` propagates the fix to every
beat of the unit by R-peak alignment plus RR-scaling of the T segment. The 11
landmarks, in canonical order

```
P_on  P_pk  P_off | QRS_on  Q  R  S  QRS_off | T_on  T_pk  T_off
```

## Setup

Put `medalcare_label_ecg.m` on the MATLAB path (it lives in
`manual_labelling/tool/`). No toolboxes required, needs a desktop figure
(R2019b+). The raw waveforms (`WP2_largeDataset_Noise/`) must be reachable from
the repository root.

## Run

```matlab
medalcare_label_ecg('manual_labelling/data/all_units_worklist.csv')   % the all-units worklist
medalcare_label_ecg                                    % or pick a worklist CSV via a dialog
medalcare_label_ecg(WORKLIST_CSV, REPO_ROOT)           % if the repo root isn't auto-found
medalcare_label_ecg(WORKLIST_CSV, REPO_ROOT, OUT_CSV)  % choose the corrections file
```

`REPO_ROOT` is the folder that contains `WP2_largeDataset_Noise/` (auto-detected
by walking up from the worklist file, or from `config/paths.yaml`). By default
corrections are saved to
`manual_labelling/corrections/<worklist>_corrections.csv` — for the
all-units worklist that is `all_units_worklist_corrections.csv`.

## Filtering the list

Two dropdowns at the top-left combine

- **class** — `all` or one disease class (sinus, avblock, iab, lae, fam, rbbb,
  lbbb, mi).
- **qc** — `all` / `critical` / `minor` / `clean` (the pipeline tier the unit
  was classified as critical = 4,251 units, minor = 9,541, clean = 188,384).

Pick, say, `mi` + `critical` to work the real errors, or `sinus` + `clean` to
spot-check. The banner shows **record · lead · CLASS · beat · unit N/of**, where
N/of is the position within the current filtered set.

The left list shows up to 20,000 units at once (a full class+qc combination
fits), for the very large views (`all`/`all`, or `all`+`clean`) it shows the
first 20,000 with a "+N more" line — **Prev/Next still walk through every
matching unit**, the cap only limits how many rows are drawn in the list.

## Correcting a landmark

- **Drag** its coloured line (snaps to 1 sample = 2 ms). Each of the 11
  landmarks has its own distinct colour, with a matching boxed label, labels are
  staggered in height so clustered landmarks (the P group, the QRS group) stay
  readable.
- Or **click it** (on the line or in the right-hand fiducial list) to select,
  then **← / →** to nudge (**Shift** = 5 samples), or **click on the trace** to
  jump the selected landmark there. Clicking the trace also **restores** a
  landmark you had marked absent.

Marking a wave absent (common in low-amplitude limb leads and FAM)

- **P absent** / **T absent** clears (or restores) that whole wave and sets its
  presence flag. **Clear sel** marks just the selected landmark absent (e.g. a
  missing Q or S). Absent landmarks are written as empty samples, which is how
  the merge step and QC treat "no wave here."

Other controls **Reset** restores the ECGdeli values for the current unit, the
right panel shows PR / QRS / QT / T-duration (with a `!` when outside the
plausibility range) and a warning if the landmarks fall out of order. The line
under the trace shows the unit's flag, a suggested fix for it, and a
class-specific reminder.

Keys `n`/`p` next/prev unit, `r` mark reviewed & next, `s` save now,
`Delete`/`Backspace` clear the selected landmark.

## Saving vs. Reviewed (important)

**Every edit auto-saves immediately.** Dragging, nudging, P/T-absent, Clear sel
and Reset each write to the corrections CSV in the background, it also saves when
you switch unit and when you close the window. The green status line reads
"All changes saved", you never need to press **SAVE** per unit (that button just
forces a write now and prints the path to the console). Reopening the same
worklist reloads your corrections so you can resume.

Saving and reviewing are different jobs

- **Auto-save** preserves your edits so nothing is lost — written with
  `reviewed = 0`.
- **Reviewed** (the button, or `r`) marks a unit finished with `reviewed = 1`
  and advances to the next.

`merge_manual_corrections.py` **applies only rows with `reviewed = 1`**. So a
unit you edited but never marked reviewed is saved (for resuming) but *skipped*
at merge time. Workflow per unit fix the fiducials → press **Reviewed**. A beat
that already looks correct still gets pressing **Reviewed** to record
"checked, correct" and include it in the merge with the existing values.

The output CSV columns are exactly what the merge step reads `record_id,
disease_class, lead, beat_id, fs_hz, n_samples,` the 11 `*_sample` values
(empty = absent), `p_present, qrs_present, t_present, flags, also_delineator,
priority, label_source = manual_corrected, reviewed, edited_at`.

## Deleting a bad record

Some recordings are too corrupt to keep. The red **Delete rec** button (far
right of the button row) excludes the WHOLE record, all 12 leads, since the
global model uses all 12. It asks for confirmation, then records the record in
`deleted_records.csv` next to the corrections file and removes its units from
the list right away. Deleted records stay hidden when you reopen the tool and
are skipped by both filters. Raw signal files are not touched by the tool.

Deletion is an exclusion list, so it stays reversible until you run
`apply_deletions.py` (below). Remove a line from `deleted_records.csv` to bring a
record back. `apply_deletions.py` then strips the excluded records out of
`master_labels.csv`, `signals_index.csv`, `all_units_worklist.csv`, and the
corrected global file, writing a `.bak` of each and keeping every raw file.

## Downstream pipeline (after correcting)

From the repo root

```bash
# 1) fold the reviewed corrections into master_labels.csv (propagates to all beats)
python3 manual_labelling/scripts/merge_manual_corrections.py \
    manual_labelling/corrections/all_units_worklist_corrections.csv

# 2) (optional) refresh the qc_status metadata to reflect the corrections
python3 dataset_curation/scripts/add_qc_status.py
python3 dataset_curation/scripts/add_crosscheck_qc.py

# 3) remove any records you deleted with the Delete rec button (raw files kept)
python3 manual_labelling/scripts/apply_deletions.py

# 4) rebuild the GLOBAL delineation target from the corrected master
python3 dataset_curation/scripts/build_edited_global.py
#    -> dataset_curation/data/global/reconciled_global_fiducials_corrected.csv
```

`merge` updates `master_labels.csv` in place (with a `.bak`). Use
**`build_edited_global.py`** — not `build_reconciled_global.py` — for the
corrected global target the original reads the raw ECGdeli CSV and would ignore
your corrections. Likewise **do not rerun `build_master.py` after merging**, it
rebuilds `master_labels.csv` from the raw ECGdeli CSV and would wipe the
corrections. Those two scripts are for a clean-slate rebuild only.

## Notes

- **Time (x-axis, ms).** 500 Hz → 1 sample = 2 ms, the axis is `sample × 2`. The
  view is zoomed to the current beat (earliest→latest fiducial, plus ~120 ms
  padding each side). The right-hand list shows both the sample index and the ms.
- **Voltage (y-axis, mV).** The lead's raw waveform amplitude in millivolts, read
  straight from the signal file, the axis auto-scales to the visible trace.
- Raw files are `12 × 5000` (leads × samples), lead order
  `I II III aVR aVL aVF V1..V6`, the tool transposes automatically if a file is
  stored samples × leads.
- Signals are cached per record, so stepping through several leads of the same
  recording only reads the file once.
- The tool corrects the **representative middle beat only** — the merge step
  propagates it to every beat. Do not try to hand-place all ~13 beats.
- A single bad unit can never freeze the UI the render is guarded, so a
  problem unit shows an error in the banner and you can move on.
