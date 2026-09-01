# Gold

A gold-standard fiducial review set for the MedalCare-XL delineation dataset, plus the
patched MATLAB tool used to build it.

Nothing outside this folder is changed by anything in here. The original tool at
`manual_labelling/tool/medalcare_label_ecg.m` is untouched, and no script in `Gold/scripts/`
writes to the canonical corpus.

## Why this folder exists

The corpus carries 194,680 clean record-lead units. Correcting them by hand is not
feasible, and two measurements taken on the first 186 reviewed units show that it would
not be the right use of the time even if it were.

The first measurement is that the reviewer's edits barely moved the training target. The
global label for a record is a p25 / p75 reconciliation across roughly twelve leads, so
editing 1.54 of those leads shifted the reconciled boundary by a median of 0.0 ms. Two
hours of careful work changed almost nothing downstream.

The second measurement is that every one of those 186 units was marked clean by the
cross-lead QC and every one of them still needed an edit. Cross-lead consistency cannot
detect an error that all twelve leads share, and the same heuristic delineator ran on
every lead, so a bias in the delineator appears as agreement rather than as disagreement.
Consistency and accuracy are different properties and only the first one was ever
measured.


## The protocol

800 records, 100 per disease class, split into 400 calibration and 400 test with 50 per
class in each half. One row per record rather than one row per lead, carrying the
reconciled global label, so a record is reviewed once instead of twelve times.

The calibration half is what the correction is estimated from. The per-fiducial offset
between the reconciled label and the reviewed label is measured on those 400 records,
per disease class where the class has enough signal to support it, and that offset is
what gets applied to the rest of the corpus.

The test half is never fitted on. It is the only evidence that the correction estimated
from the calibration half generalises instead of memorises. It must also be excluded
from any training split, since a model trained on records whose labels were used to
validate the label correction would report an optimistic error.

## Files

`scripts/build_gold_worklist.py` builds both worklists. It reads the reconciled global
labels produced by `dataset_curation/scripts/build_edited_global.py` rather than
recomputing the reconciliation, so the label under review is exactly the label the model
will be trained on. Records already reviewed under the old per-lead protocol are held out,
which removed 134 of 16,781.

`data/gold_worklist_calibration.csv` and `data/gold_worklist_test.csv` hold 400 records
each and share no record. Both carry the exact column set the MATLAB tool requires, so
they open without any conversion step.

`tool/gold_label_ecg.m` is the reviewer. `corrections/` is where it writes.
`scripts/gold_common.py`, `scripts/fit_gold_offsets.py`, `scripts/validate_gold_offsets.py`
and `scripts/apply_gold_offsets.py` turn the reviewed records into a de-biased dataset, and
are described further down.

Beyond the tool's required columns each row carries a few review aids. `gold_split` marks
which half the record belongs to. `qc_status` is the old cross-lead verdict, kept for
comparison rather than for triage. `n_recon_leads` is how many leads survived QC and fed
the reconciliation. `ref_lead_mad_ms` is how far the displayed lead's own boundaries sit
from the reconciled label once both are aligned on their R peak, with a median of 8.0 ms
and a p90 of 14.7 ms across the 800. `p_dur_ms`, `pq_seg_ms` and `qt_ms` are the three
intervals most likely to expose a boundary error, and they are repeated in the `flags`
string so they are visible in the tool's unit list without opening the record.

## The tool

`tool/gold_label_ecg.m` is a copy of `manual_labelling/tool/medalcare_label_ecg.m` with
two functional changes. All twelve leads of the current record are drawn behind the
reference lead in grey, each normalised to the reference lead's amplitude over the visible
window so that a low-amplitude lead stays legible next to a large one. Press `o` to toggle
the overlay. Every other control behaves as it does in the original.

The overlay is the whole point of the copy. A wave boundary belongs to the heartbeat and
not to any single lead, so the true onset is the earliest moment at which any lead leaves
baseline and the true offset is the latest moment at which any lead returns to it. A
reviewer looking at one lead cannot see that, and will reproduce the same lead-local bias
the delineator has. With all twelve visible the earliest departure and the latest return
are directly readable.

The normalisation is recomputed on every zoom and pan, so zooming into the P wave rescales
all twelve leads to that window and makes the earliest P onset readable even when the P is
small next to the QRS.

### The spatial magnitude strip

The overlay shows twelve traces and leaves the reviewer to combine them by eye. The second
change does that combination arithmetically and draws the result as one orange curve in a
band below the trace. Press `m` to toggle it.

The curve is the root sum of squares of eight leads at every sample, each with its own
baseline removed. It answers the question the overlay can only hint at, which is where the
heart actually starts and stops rather than where any one lead happens to notice.


Squaring removes the polarity problem outright, so an inverted lead contributes exactly as
much as an upright one and aVR stops being a special case. Summing over eight axes removes
the perpendicularity problem, since a vector perpendicular to one lead is not perpendicular
to all eight. What is left rests on its noise floor and lifts the moment any part of the
heart depolarises.

Eight leads and not twelve, since III, aVR, aVL and aVF are exact linear combinations of I
and II. III is II minus I, aVR is minus half of I plus II, aVL is I minus half of II, and
aVF is II minus half of I. Those four identities were checked against the raw files and
hold to one part in ten thousand, which is the precision the signals are stored at.
Inculding them would count the frontal plane three times over and tilt the curve toward
frontal activity that the precordial leads see only once. The eight used are I, II and V1 through V6.


### How to read the strip

The dotted orange line is that record's own tenth percentile magnitude, drawn as a rough
noise floor. Treat it as context and not as a rule. Measured across the eight records it
ranges from 0.016 per cent of the peak magnitude to 3.7 per cent, a factor of 237, so a
fixed multiple of it means something quite different from one record to the next. The MI
records are the noisy end of that range by a wide margin, with a dynamic range near 27 to 1
where a clean sinus record reaches 6000 to 1.

Read the shape instead. The curve has a flat stretch between waves and a visible lift where
a wave begins, and the boundary is where the lift starts rather than where the curve
crosses any particular height. Zoom in, since the strip is normalised over the visible
window and a P wave twenty times smaller than the QRS fills the band once you zoom to it.

The fiducial lines span the full axis height on purpose, so each one crosses the orange
curve and shows at a glance whether the landmark sits at the lift or somewhere after it.

Whatever rule you settle on, apply the same one to all 800 records. A rule applied
consistently makes the residual a property of the delineator, which is exactly what the
offsets are meant to estimate. A rule that drifts across the review makes the residual a
property of the reviewer, and no amount of records recovers from that.

One thing the strip does not do is tell you which way the delineator is wrong. A
threshold-crossing experiment on the eight records gives answers that swing by hundreds of
milliseconds depending on the threshold chosen, and on the P wave in particular any
threshold set as a fraction of the peak is a large fraction of the P itself and places the
crossing late by construction. The direction and size of the bias is what the 400
calibration records are for. The strip is there to make each individual judgement better
informed, not to pre-empt the measurement.

### The spatial velocity strip

A second curve is available on the `v` key, drawn in teal in its own band below the orange
one. It is the root sum of squares of the first derivative of the same eight leads, so it
measures how fast the heart vector is moving rather than how far it has already moved. A
derivative carries no baseline term at all, which leaves the curve blind to any residual
offset and resting on zero wherever the signal is still.

That difference matters most at the QRS. Measured across the same eight records the velocity
curve spends about 9 ms between the two heights the magnitude curve takes 19 ms to cross, so
its foot is roughly half as long and the eye has a sharper corner to fix the QRS onset and
offset against.

It is not a replacement. The same eight records were scored again at all three boundaries
with the crossing height taken from each wave's own peak rather than from the peak of the
beat, and on that fairer test the velocity curve is the worse of the two at the slow
boundaries. Its P foot lands about 84 ms inside the labelled P onset against 46 ms for the
magnitude curve, and at the T offset its spread across records is 33 ms against 10 ms, since
a T wave ends by flattening and a derivative has nothing left to report once the wave stops
moving. Use the teal curve for the QRS boundaries and keep reading the P and T boundaries
off the orange one. It is off by default, so the view stays identical to the one the earlier
records were reviewed under until you ask for it.

To run it, start MATLAB at the repository root and use

    addpath('Gold/tool')
    gold_label_ecg('Gold/data/gold_worklist_calibration.csv')

Corrections auto-save to `Gold/corrections/gold_worklist_calibration_corrections.csv` in
the same format `merge_manual_corrections.py` reads. Do the calibration half first and
leave the test half until the correction has been estimated, so that the test half is
reviewed without knowing what the correction predicts.


## The scripts, and the order to run them

Review the calibration half, then

    python3 Gold/scripts/fit_gold_offsets.py

Review the test half, then

    python3 Gold/scripts/validate_gold_offsets.py
    python3 Gold/scripts/apply_gold_offsets.py --recon <reconciled_global.csv>

`fit_gold_offsets.py` takes the difference between the reviewed label and the reconciled
label on the 400 calibration records and summarises it as one number per landmark. It
produces two candidates for every landmark, one global and one per disease class, and it
adopts neither. A median is used rather than a mean, since a handful of records will have
a landmark dragged onto the wrong feature and one such record moves a mean by several
milliseconds. Residuals beyond 200 ms are dropped and counted separately, since at that
distance the two labels are on different features rather than disagreeing about one.

`validate_gold_offsets.py` is the only script that reads the test half, and it should run
once. Fitting, looking at the test half, adjusting the fit and looking again turns the
test half into a second calibration set and the reported improvement stops being evidence.
