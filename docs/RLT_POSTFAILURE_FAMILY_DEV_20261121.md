# RLT-PF03: a family-aware stopping prior against the 192D ridge head

Date: 2026-11-21. Status: **fresh development experiment, independently
replayed CPU NativeRanker trajectories**. No native Lean/MCP or Inkling
teacher inference occurred.

The [GitHub Actions campaign #44918460975](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44918460975)
passed its source-only family-prior tests, Ruff, true fixed-depth
trajectory collection, complete exact replay and both preregistered
value-control fits. Its artifact datagoblin-postfailure-family-control-20261126
retains the full suite, model snapshot, 448 original trajectories,
1,792 independent candidate checks, ridge and family-calibration
receipts and their evaluation readouts.

## Experimental custody

- Seed 20261126, 16 roots per family in each of five split buckets.
  Source collection selected 112 TRAIN, 112 CALIBRATION, 112 ordinary
  and 112 OOD roots. The sealed_poison portion was unopened.
- The source model is the same direct-feedback NativeRanker
  configuration used in RLT-PF02: model seed 7, trained at depth 4,
  inference at four full visits.
- This suite has 560 unique task digests. Its intersection with
  either previous scaled suite (seeds 20261124 and 20261125) is empty.
- All 448 source trajectories were independently replayed with their
  original frozen model/checker; there were zero censored trajectories.
- The 192D ridge uses public candidate attributes, observed failure
  code and remaining visit budget. The family prior sees only the
  registered task family after the first verified failure.
- Training uses TRAIN root labels only. Threshold selection uses CAL.
  Both policies were registered with zero missed CAL rescues.
  Ordinary and OOD readouts are development evidence rather than a
  scientific sealed confirmation.

## Results on identical new roots

| Policy | Ordinary solves/112 | Ordinary calls | OOD solves/112 | OOD calls | Combined solves/224 | Combined calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One check | 24 | 112 | 34 | 112 | 58 | 224 |
| First valid by depth four | 29 | 369 | 36 | 342 | 65 | 711 |
| Ridge continuation | 29 | 282 | 36 | 288 | **65** | 570 |
| Family continuation | 29 | 276 | 35 | 248 | 64 | **524** |
| Ridge-rate random | 29 | 309 | 36 | 288 | 65 | 597 |
| Family-rate random | 27 | 288 | 35 | 224 | 62 | 512 |

The family prior saved 46 verifier calls versus the 192D ridge,
losing one confirmed OOD rescue. It saved 187 calls relative to
first-valid and lost one solve. The ridge recovered all observed
later rescues on this panel, saving 141 calls versus first-valid.
Random stopping matched the ridge's solve count but spent 27 more
calls in this particular seed.

On TRAIN, there were only two future rescues among 87 initial
failures: one arithmetic-program and one constraint-assignment
root. The family prior uses a fixed Beta(1,9) pseudocount and learned
the following estimated probabilities of later rescue:

| Family | Failed TRAIN roots | Rescues | Estimated rescue probability |
| --- | ---: | ---: | ---: |
| Arithmetic program | 13 | 1 | 0.087 |
| Constraint assignment | 13 | 1 | 0.087 |
| Evidence retrieval | 16 | 0 | 0.038 |
| Experiment plan | 16 | 0 | 0.038 |
| Finite-state plan | 0 | 0 | 0.100 (prior only) |
| Graph route | 14 | 0 | 0.042 |
| Table transform | 15 | 0 | 0.040 |

The family CAL threshold was 0.04; the ridge CAL threshold
was 0.004396466410206273. Both preserved the 32 CAL
first-valid solves. Family selection spent 264 calls and ridge
selection 279, against 357 for full first-valid CAL.
The family model lost a single OOD repair in evaluation despite
its zero-missed CAL criterion.

## Interpretation and next design gate

The observed data show that a low-capacity feature can recover
much of the value-of-check policy selected by a 192D ridge.
The first 84-root pilot failed to find any additional solves.
Across the larger datasets, later rescue events remain rare,
with significant source-family and seed dependence. Any claimed
mechanistic advantage of latent RLT state would require an
additional gain above this family-only baseline at a registered
verifier-cost budget.

Useful next tests include a hierarchical prior conditioned on
already observed failure codes and uncertainty, a calibration
margin for source distribution changes, and online execution
where a deferred check actually avoids a model call. All
proposed families, thresholds and comparisons should be frozen
on new TRAIN/CAL worlds before opening the next evaluation roots.

The new LeanKernelChecker and MCPStateChecker modules define
operator-owned verification interfaces; their tests currently use
synthetic mocked runtimes, while the genuine DataGoblin seven-family
source/rollout studies in this series have real independently
replayed environment verification.
