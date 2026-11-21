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

The experiment evaluated activation prediction and perturbation response.
Live answer accuracy and causal interventions have their own evaluation
stages. The local `native-rlt-depth-forecast-01` capture contains fitted
heads, pass-indexed norms, the comparison by fit horizon, and digests of
the source trace and SAE outside Git.
