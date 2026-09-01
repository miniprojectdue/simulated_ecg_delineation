# Rule-based baseline comparison, external test set

`delineate_ecg_v3` scored against the fine-tuned network on the same 1,944 units,
the same reference and the same metrics code. Statistics are clustered by
recording, since the boundary reference is one criterion-defined set per
recording shared by all twelve leads.

## How it was run

The MATLAB source was ported to Python constant for constant. Every threshold,
search window, refinement and fallback is as written, with the millisecond
constants converted to samples at the actual rate, so at 1000 Hz the port reduces
exactly to the original. The tool ran on the same 500 Hz signals the network
read, since no 1000 Hz source survives in a readable form. Its five landmarks are
the QRS onset and offset and the T onset, peak and offset, which is its
documented scope.

## The headline, and why it does not mean what it looks like

Over the three scoreable boundaries the baseline is ahead, 17.35 ms against
32.02 ms, a paired difference of 14.67 ms with a 95 per cent interval of 9.60 to
20.33 that excludes zero.

That result is almost entirely an artefact of the corpus, and a control shows it.

| Landmark | reference sd | constant predictor | delineate_ecg_v3 | network arm B |
|---|---|---|---|---|
| QRS onset | 1.4 ms | **0.88** | 4.51 | 21.66 |
| QRS offset | 11.9 ms | **9.30** | 12.96 | 33.98 |
| T offset | 71.0 ms | 66.60 | **34.57** | **35.24** |

The constant predictor ignores the signal and returns the corpus median. On the
QRS onset it beats both systems, and on the QRS offset it beats both systems. The
MonoAlg3D simulations begin at ventricular activation, so every record starts its
QRS within a 10 ms window and the reference standard deviation is 1.4 ms. That
landmark carries almost no information on this corpus, and any method that emits
a plausible constant scores well on it. The same holds, less severely, for the
QRS offset.

Both systems clear the control on the T offset and only on the T offset, by
32.04 ms and 31.36 ms respectively. That is the one landmark on this test set
where the comparison is informative.

## The informative comparison

| Split | delineate_ecg_v3 | network arm B | paired difference | verdict |
|---|---|---|---|---|
| all 162 recordings | 34.57 | 35.24 | +0.65 [−4.71, +6.35] | not significant |
| matched, 117 recordings | 43.07 | 31.52 | **−11.60 [−17.60, −5.54]** | favours the network |
| ischemia, 45 recordings | 12.45 | 44.93 | **+32.48 [+27.28, +38.47]** | favours the baseline |

On the morphologies that have a counterpart in the training corpus, healthy sinus
and anterior and inferior infarction, the network places the T offset 11.6 ms
closer to the reviewer than the purpose-built tool does. On the 45 ischemia
recordings, a morphology absent from MedalCare-XL entirely and one the baseline
was calibrated for, the baseline is far ahead. Pooled, the two are
indistinguishable.

That is a coherent and defensible result. Each system is better on the terrain it
was built for, and the aggregate hides both effects.

## Landmarks to exclude or caveat

**T peak, 0.94 ms against 37.72 ms, is not a delineation result.** The reviewer
placed peaks by an extremum on the lead trace and the baseline computes the same
extremum, so the two agree by construction. The labelling protocol says as much,
recording that a deterministic rule reproduces hand-placed peaks 88 to 91 per
cent of the time exactly. Report it as a property of the reference, not as a win.

**T onset is unscoreable by the protocol's own statement.** The reviewer used a
5 per cent threshold on the spatial magnitude, landing about 10 ms after the QRS
offset, and the baseline requires a 16 per cent departure from the ST plateau
sustained for 12 ms, landing far later. The 74.9 ms bias is the distance between
two conventions. The protocol says to state that T onset is ill-defined for this
data, and it should be stated.

## Circularity check

The review tool seeded its markers from `delineate_ecg_v3`, so exact agreement
was measured to establish whether the reviewer simply accepted the seed.

| Landmark | exact | within 1 ms | within 2 ms |
|---|---|---|---|
| QRS onset | 6.7 % | 23.8 % | 27.7 % |
| QRS offset | 0.5 % | 1.7 % | 4.6 % |
| T onset | 0.2 % | 0.4 % | 0.8 % |
| T offset | 1.0 % | 3.2 % | 5.3 % |
| T peak | 49.2 % | 98.8 % | 99.4 % |

The reviewer overrode the seed on essentially every boundary, which is what the
protocol instructs and which clears the boundary comparison. The T peak figure is
the extremum agreement described above.

## Biomarkers, all 162 recordings

| Biomarker | delineate_ecg_v3 | network arm B | verdict |
|---|---|---|---|
| QRS duration | 13.12 | 19.59 | favours the baseline |
| QT interval | 33.09 | 43.97 | favours the baseline |
| T peak to end | 34.72 | 55.57 | favours the baseline |

On the matched 117 the QT interval is a tie, 41.56 against 41.89. QRS duration
inherits the degenerate QRS onset, so it carries the same caveat as its
endpoints.

## What this does and does not establish

It does not establish that a learned delineator beats a hand-tuned one. On this
test set, with these landmarks, it does not.

It does establish three things worth reporting. The external corpus cannot
discriminate on the QRS boundaries at all, which is a finding about the test set
and one no previous evaluation of this baseline has stated. On the one landmark
that does discriminate, the network is better than the purpose-built tool on
matched morphology and worse on morphology it has never seen, which is exactly
the behaviour a learned model should show. And the baseline cannot attempt the P
wave or the Q, R and S deflections at all, so on six of the eleven landmarks
there is no comparison to make.

## The experiment this now demands

The symmetric run has not been done. The baseline has never been scored on
MedalCare-XL, where the network is in distribution, where a P wave exists and
where eight pathology classes are present. If the network wins there as clearly
as the baseline wins on ischemia here, the honest conclusion is that each tool
owns its own domain and only one of them transfers, which is a stronger and more
interesting claim than the one the dissertation currently makes.

## Files

    baseline_external/metrics.json     full landmark, lead and class breakdown
    baseline_external/per_unit.csv     1,944 rows, predictions, truth and errors
    baseline_external/comparison.json  paired clustered comparison, three splits
    baseline_external/control.txt      the constant-predictor control
    baseline_v3.py                     the port
    run_baseline.py                    the scoring run
    compare.py                         the head to head
