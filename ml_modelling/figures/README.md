# Chapter 4 figures

Run from the repository root. Each script writes one vector PDF into
`Dissertation/images/` and prints the numbers it plotted so they can be checked against the
tables in the text.

    python3 ml_modelling/figures/mkfig_landmarks.py    # Figure, landmark error on both surfaces
    python3 ml_modelling/figures/mkfig_forest.py       # Figure, the baseline adjustment ladder
    python3 ml_modelling/figures/mkfig_heatmap.py      # Figure, the lead-pair degradation matrix
    python3 ml_modelling/figures/mkfig_ablation.py     # Figure, ablations against the seed floor
    python3 ml_modelling/figures/mkfig_examples.py     # Figure, three delineated beats

`figstyle.py` holds the shared palette and the matplotlib settings. Every colour used is muted by
design, the type is set in a serif to sit with the document, and no figure carries a gridline,
border or tick that is not doing work.

The forest and ablation scripts recompute their intervals from the per-unit tables rather than
reading a stored summary, so a figure cannot drift away from the table beside it. Both print what
they plotted for exactly that reason.
