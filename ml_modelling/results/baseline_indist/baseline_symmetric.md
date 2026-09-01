# Network against delineate_ecg_v3, both directions

The baseline has now been scored on both test surfaces. Each is the other's
mirror. On the external set the baseline is on the corpus it was authored for and
the network is out of distribution. On the in-distribution set the roles reverse.
Statistics are clustered by recording throughout.

## Headline

| Surface | network | delineate_ecg_v3 | paired difference | verdict |
|---|---|---|---|---|
| MedalCare-XL held out, 109 units, 100 records | **8.27** | 46.08 | −37.81 [−42.63, −33.63] | **network**, by 5.6× |
| MonoAlg3D external, 1,944 units, 162 records | 27.95 | **17.35** | +10.60 [+5.62, +16.16] | baseline, by 1.6× |

Boundary aggregate in milliseconds. The in-distribution figure is over the four
boundaries the baseline can produce, so the two systems are compared on equal
terms. The baseline column in-distribution is its **best case**, explained below.

The network's margin on its own terrain is more than three times the baseline's
margin on its.

## In-distribution, per landmark

| Landmark | network MAE | network sens | v3 MAE | v3 sens |
|---|---|---|---|---|
| p_onset | 14.41 | 0.824 | — | 0.000 |
| p_peak | 9.80 | 0.852 | — | 0.000 |
| p_offset | 13.83 | 0.815 | — | 0.000 |
| qrs_onset | **5.23** | 0.954 | 24.31 | 0.211 |
| q_peak | 1.46 | 0.787 | — | 0.000 |
| r_peak | 2.59 | 0.938 | — | 0.000 |
| s_peak | 1.50 | 0.863 | — | 0.000 |
| qrs_offset | **7.28** | 0.963 | 34.39 | 0.734 |
| t_onset | **11.56** | 0.862 | 99.54 | 0.018 |
| t_peak | **6.48** | 0.936 | 43.16 | 0.826 |
| t_offset | **10.53** | 0.927 | 29.83 | 0.853 |

The network wins every landmark both systems can produce, and the baseline
produces nothing at all on six of the eleven, having no P stage and not naming
the Q, R and S deflections.

The T onset is the starkest. The baseline places it within 25 ms on 1.8 per cent
of units against the network's 86.2 per cent.

## Two choices made on the baseline's behalf, both generous

**Beat segmentation.** The tool has no beat detector, and its own driver cuts
fixed 1,000 ms cycles from the record start, which works on MonoAlg3D because
each record holds one beat beginning at activation. A MedalCare record is ten
seconds of about thirteen beats, and the mid-RR window used by the training
corpus places the QRS onset 250 to 320 ms in, past the tool's 200 ms search
limit. Each segment was therefore started at a seeded random offset of 100 to
180 ms ahead of the reviewer's QRS onset. The tool is told roughly where the beat
is, which is help it never receives on the external set, and the offset is
randomised so the QRS onset does not become a constant.

**The P wave.** The tool estimates its noise floor over the first 80 ms of the
segment and has no P stage. On a MedalCare beat the PQ segment is close to zero,
so any window giving it the lead-in its thresholds assume also contains atrial
activation. Both conditions were run.

| Condition | v3 boundary MAE | network | difference |
|---|---|---|---|
| natural, P wave present | 102.20 | 8.27 | −93.94 [−99.85, −88.32] |
| P blanked to the isoelectric median | 46.08 | 8.27 | −37.81 [−42.63, −33.63] |

Left alone with a P wave the tool collapses. Its QRS onset bias is −118 ms,
because it locks onto atrial activation and calls it the QRS, and its sensitivity
falls to 0.083. Blanking the P segment more than halves its error and is what the
table above reports, so the headline uses the most favourable reading available
to it.

## What the pair establishes

Each system is better on the corpus it was built for, which is the expected and
honest result, but the two margins are not symmetric. The network is 5.6 times
better on MedalCare-XL than the baseline is, while the baseline is 1.6 times
better on MonoAlg3D than the network is. The network also covers eleven landmarks
against five, and degrades rather than collapses when the domain changes, where
the baseline collapses outright when handed a wave class it was never written
for.

Three qualifications belong with it. The external corpus cannot discriminate on
the two QRS boundaries, where a constant predictor beats both systems. The
external T-offset comparison depends on a documented 17.6 ms convention
difference between the two labelling protocols, and the network's advantage on
matched morphology survives either convention. And the baseline was given beat
segmentation and a blanked P wave in distribution, neither of which it would have
in use.

## Files

    baseline_indist/metrics.json           both conditions, full landmark breakdown
    baseline_indist/per_unit_natural.csv
    baseline_indist/per_unit_p_blanked.csv
    run_baseline_indist.py                 the run, with both design choices in the docstring
