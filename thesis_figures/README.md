# ImageCodes — generation code for the Chapter 3 and 4 figures

Every figure included in Chapters 3 and 4 of `main.tex` (and the results appendix), the script
that generates it, and the data it reads. Run everything from the repository root
(`~/Desktop/Delineation`), for example:

    python3 Dissertation/ImageCodes/mkfig_landmarks.py

Each script has an `EDIT HERE` block (or plain constants) at the top for the things you are
most likely to change: legend labels, colours, the exemplar record/lead, and the output paths.
`figstyle.py` holds the shared palette and matplotlib settings; every script imports it, so a
colour changed there changes everywhere.

## Chapter 3 (Methods)

| Figure | Script | Reads |
|---|---|---|
| `fig_ml_unit_window.png` | `mkfig_ml_unit_window.py` | `ml_modelling/data/finetune_units.csv`, `ml_modelling/data/signal_cache/` |
| `fig_ml_postprocess.png` | `mkfig_ml_postprocess.py` | same, plus `ml_modelling/checkpoints/finetune_toffset_fix_tailweight/best_geometry.pt` |

Both pick their exemplar unit automatically (first sinus unit on lead II with a P wave whose
record is in the signal cache) — set `RECORD` and `LEAD` at the top to show a specific unit.
The postprocess script runs the checkpoint on CPU through the same dataset pipeline as
`evaluate.py`, so it needs torch and takes a few seconds.

## Chapter 4 (Experimental Results)

| Figure | Script | Reads |
|---|---|---|
| `fig_crosslead_fiducial.png` | `mkfig_crosslead_fiducial.py` | `ecgdeli_labelling/data/qc/crosslead_fiducial_flags.csv` |
| `fig_repro_timing_leadII.png`, `fig_repro_amp_leadII.png`, `fig_perbeat_timing_leadII.png`, `fig_perbeat_amp_leadII.png` | `reproduce_paper_stats.py` (copy of `statistics/scripts/reproduce_paper_stats.py`) | the statistics pipeline inputs; writes the PNGs next to itself, copy them into `Dissertation/images/` |
| `fig_landmark_surfaces.png` | `mkfig_landmarks.py` | `ml_modelling/results/indist_toffsetfix2_tailweight/per_unit.csv`, `ml_modelling/results/zeroshot_ext_qrs_off2/per_unit.csv` |

`mkfig_crosslead_fiducial.py` recomputes the medians, 98th-percentile tolerances and the
per-lead flag rates from the QC output, so the figure cannot drift from the QC numbers
(current: overall 5.9%, worst lead I at 13.5%). The `reproduce_paper_stats.py` copy is kept
verbatim from `statistics/scripts/` — if you edit it, edit the original too or they will
diverge.

## Results appendix

| Figure | Script | Reads |
|---|---|---|
| `fig_fewshot_examples.png` | `mkfig_fewshot.py` | `ml_modelling/results/{zeroshot_heldout50_2, fewshot_ext_adapted2, fewshot_ext_adapted5}/per_unit.csv`, `ml_modelling/data/signal_cache/` |
| `fig_perbeat_*` | covered by `reproduce_paper_stats.py` above | |
| `fig_gold_tool.png` | none — this is a screenshot of the labelling tool, there is no generating code | |


`mkfig_ml_postprocess.py`. Outputs go to `Dissertation/images/` as both PNG (300–400 dpi) and
PDF unless noted above.
