# DISTILL-18 native MoLT compactification: hosted result

Tested commit 8f20fdc0dbc7d79ef2b36181cce1e00adee01e39. [Dedicated run 46098983652](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46098983652) passed 12 focused tests, Ruff and the frozen campaign. [Artifact 15244119432](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46098983652/artifacts/15244119432), raw rows digest 0823f0af2d152601143fe8be519ceb6cae12e37ec0263b934e3a66c699351ae0. First run failed SIM108 after tests; the corrected commit only simplified the conditional assignment.

128 development roots (64 train, 64 calibration), two variants, single and eight-repeat workloads, four arms: 9,216 native episodes, all verified and optimal. Full action/observation traces identical to reference; utility, receipt binding, verifier calls unchanged. No provider calls or model training.

Repeated-workload aggregate (2,048 submissions per arm):

| Arm | DP visits | Submission process CPU (ms) | Cache hits |
|---|---:|---:|---:|
| Reference | 13024 | 17.910906 | 0 |
| Compiled | 8928 | 28.848412 | 0 |
| Cache2 | 1116 | 29.911951 | 1792 |
| Cache8 | 1116 | 24.947787 | 1792 |

Cache hit rate 87.5%, DP visits down 91.4% relative to reference, yet measured CPU slower in all repeated-workload roots. No hits in single-pair workloads. Both caches reach at most two entries, so the eight-entry timing difference is not evidence of additional capacity benefit. Compilation is also slower on aggregate in single-pair workloads. These are descriptive process CPU measurements including normalization and key work; DP visits alone omit that work. Repeated exact trajectories are stress fixtures, not independent roots or real recurrence prevalence.

Keep the original default. Preserve exact numerical selection equivalence and independent receipt authority as an optional development component. Next test a packed integer DP and validate-once public problem preparation across native-sized and wider synthetic public arrays. No training admission or promotion.
