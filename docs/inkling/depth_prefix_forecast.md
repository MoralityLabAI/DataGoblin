# Predicting a third Inkling pass from two training transitions

The crossed RLT schedule reduced the effect of a small prelude
perturbation by 34% against direct feedback. Its third-pass activation
error rose by 1.4% and was higher on all seven calibration roots. Matched
damping gave a similar reduction in perturbation gain with a smaller
increase in mean activation error.

## Run setup

This CPU experiment used the verified Inkling Q4 pass-boundary traces and
the frozen 256-feature, top-16 SAE. It used 448 prompt tokens from seven
training roots and 448 from seven disjoint calibration roots. The SAE
had previously seen third-pass activations from the training roots.

The six affine predictors fit only the prelude-to-first-pass and
first-to-second-pass transitions. Each used an intercept, visit divided
by three, and a ridge penalty of 0.1. Fit schedules used a planned horizon
of two or three. Every runtime schedule rolled forward for three passes
from the held-out prelude using its own predicted state. A shared noise
draw with a prelude RMS of 0.0777 in SAE code units measured perturbation
gain. The fitting job used CPU under a 512 MiB commit cap.

## Third-pass measurements

| Fit horizon | Feedback | Root-balanced activation MSE | Perturbation gain | Roots below direct MSE |
| --- | --- | ---: | ---: | ---: |
| 2 | Direct | 11.059 | 8.726 | — |
| 2 | RLT | 11.215 | 5.773 | 0/7 |
| 2 | Matched damping | 11.122 | 6.008 | 2/7 |
| 3 | Direct | 11.059 | 8.726 | — |
| 3 | RLT | 17.510 | 16.165 | 0/7 |
| 3 | Matched damping | 17.289 | 14.807 | 0/7 |

For the crossed fit-two/runtime-three comparison, the paired difference
in activation MSE against direct was +0.155 for RLT and +0.063 for
matched damping. The 2.5th and 97.5th percentiles from resampling the
seven root pairs were [0.077, 0.233] and [−0.012, 0.138], respectively.
These intervals summarize variation across the calibration roots.

## Exact gains of the fitted heads

Each step has an affine state map with linear part
`(1 - eta) I + eta A`, where `A` is the fitted endpoint's state matrix.
Multiplying those matrices gives the three-pass Jacobian. Its largest
singular value is the worst-case L2 gain. Its Frobenius norm divided by
the square root of the code width gives the isotropic RMS gain.

| Feedback, fit two and runtime three | Worst-case L2 gain | Isotropic RMS gain | Feature 121 L2 gain |
| --- | ---: | ---: | ---: |
| Direct | 127.24 | 8.83 | 0.284 |
| RLT | 82.58 | 5.81 | 0.342 |
| Matched damping | 87.05 | 6.05 | 0.351 |

Worst-case directions amplify even where the selected SAE feature
attenuates. These gains come from the affine predictors in SAE code
space. The feature-121 live intervention still needs its matched controls.

The experiment evaluated activation prediction and perturbation response.
Live answer accuracy and causal interventions have their own evaluation
stages. The local `native-rlt-depth-forecast-01` capture contains fitted
heads, pass-indexed norms, the comparison by fit horizon, and digests of
the source trace and SAE outside Git. Its `linear_gain.json` audit records
exact per-pass gains and binds them to the fitted-head digest.

## Six-pass transfer

The frozen heads were evaluated on final-prefill-token trajectories from
the [six-pass holdout](six_pass_holdout.md). Its 14 roots are disjoint from
both the head-fitting and calibration roots. Each head started at the
measured held-out prelude and rolled forward on its own predictions for six
visits. The visit clock remained visit divided by three. Each feedback rule
used its six-visit coefficient schedule.

| Fit horizon | Feedback | Pass-six activation MSE | Isotropic RMS gain |
| --- | --- | ---: | ---: |
| 2 or 3 | Direct | 120.79 | 24.12 |
| 2 | RLT | 76.56 | 8.12 |
| 2 | Matched damping | 68.10 | 8.19 |
| 3 | RLT | 58.93 | 18.12 |
| 3 | Matched damping | 58.27 | 19.29 |
| — | Copy the measured prelude | 56.56 | — |

At pass six, RLT fitted with horizon three reduced activation MSE by
51% versus the direct head. The measured-prelude baseline had the lowest
error. The frozen SAE's reconstruction of the measured pass-six state
had MSE 5.42. The local `native-rlt-depth-transfer-06` report holds
per-pass metrics, per-root errors, and the source digests. This comparison
estimates offline state transfer. Answer-level effects remain part of
the separate live routing trial.

## Stationary endpoints with a bounded state map

A follow-up fit replaced the visit clock with a shared stationary endpoint.
It used the same first two training transitions, the frozen SAE, and a
three-visit fit schedule. Each head was fitted by weighted affine ridge
regression, projected to a spectral cap, and refitted for its intercept.
The cap was selected on three-step activation error in leave-one-root-out
validation using only the seven training roots. The direct head selected
zero, giving a constant endpoint. RLT and matched damping selected
one, giving a non-expansive state map in SAE code space.

| Predictor | Pass-three calibration MSE | Pass-six holdout MSE |
| --- | ---: | ---: |
| Bounded direct | 7.61 | 10.56 |
| Bounded RLT | 8.12 | 11.23 |
| Bounded matched damping | 8.31 | 11.95 |
| Training pass-two mean activation | 8.24 | 10.29 |
| Frozen SAE center | 7.49 | 10.76 |

At pass six the stationary bounded RLT head reduced MSE from 58.93
to 11.23 relative to the earlier affine RLT head fitted with horizon
three. The training pass-two mean activation still had the lowest
pass-six error. The model family changed both its clock and its spectral
constraint. The evaluation reused the seven calibration roots and
14 six-pass roots from the earlier reports. The local
`native-rlt-stable-endpoints-01` artifact contains the frozen fits,
root-fold errors, per-pass gain bounds, and baseline scores.
