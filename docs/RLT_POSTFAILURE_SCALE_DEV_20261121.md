# RLT-PF02: scaled cross-family continuation-value development

Date 2026-11-21. Status: two **verified development campaigns**,
not a sealed or powered scientific confirmation.

Both CPU campaigns used the same predeclared NativeRanker direct
schedule (seed 7), four full candidate visits and seven task families.
The independent task generator suite seeds were 20261124 and
20261125, with 16 roots per family in each of TRAIN, CALIBRATION,
ordinary_test, OOD_test and unused sealed_poison. Each campaign
selected 448 roots: 112 in each of the four accessible splits.
The task suites each contain 560 unique task digests and no lineage
was shared across the two seeds.

The completed GitHub Actions campaign,
[44917960659](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44917960659),
exactly replayed 448 native trajectories per seed (896 total)
and independently checked 1,792 candidates per seed (3,584 total).
Both source runs completed without censored episodes. The source
and model manifests, fitted shrinkage-ridge controllers and
selection receipts were retained separately in two job artifacts.

TRAIN sees only first-failure public candidate/checker-code features.
The next-step rescue outcome is a training label. CAL chooses the
score threshold subject to either 0 or 1 missed possible CAL repairs.
The ordinary/OOD test roots are scored after the threshold is frozen.
They are descriptive development tests and are not a fresh sealed
confirmation. The budgets are verifier-call counts from offline
fixed-depth counterfactual replay, not measured online runtime savings.

## Prespecified strict CAL gate: zero missed possible CAL rescues

| Source seed | Fixed one solved/checks | Learned solved/checks | Random rate-matched solved/checks | First verified by depth four solved/checks |
| --- | --- | --- | --- | --- |
| 20261124, 224 test roots | 57 / 224 | 67 / 653 | 65 / 626 | 67 / 707 |
| 20261125, 224 test roots | 55 / 224 | 60 / 431 | 58 / 398 | 61 / 720 |
| **Both, 448 test roots** | **112 / 448** | **127 / 1084** | **123 / 1024** | **128 / 1427** |

The learned controller saved 343 checker calls (24.04%) versus
always checking through the first valid candidate/budget. It lost
one available rescue on one OOD/ordinary test root across the
two seeds. Relative to fixed one-check, it gained 15 solves at
636 extra checker calls. Relative to rate-matched random, it
gained four solves for 60 extra checks. These are small raw
differences without an uncertainty interval or external validity
claim. The strict CAL gate missed zero CAL rescues but failed to
guarantee zero later missed rescues on the second seed.

Training failure counts were 88 per seed, with only 3 and 8
known future rescues respectively. The learned selection thresholds
were 0.003470723 (seed 20261124) and 0.105043867 (20261125),
with CAL continuation fractions 0.8214 and 0.3068 among
initially failed tasks. This source difference is a strong
warning about policy stability.

## Alternative prespecified CAL gate: up to one missed CAL rescue

| Source seed | Learned solved / 224 | Learned checks | First-valid solves | First-valid checks |
| --- | ---: | ---: | ---: | ---: |
| 20261124 | 67 | 620 | 67 | 707 |
| 20261125 | 56 | 297 | 61 | 720 |
| **Both** | **123** | **917** | **128** | **1427** |

The looser CAL criterion saved 35.74% of verifier calls versus
first-valid under the pooled arithmetic, while foregoing
five confirmed opportunities to rescue a failed first attempt.

## Family patterns, strict gate, 448 test roots

| Family (64 roots each) | First-check solves | First-valid solves | Learned solves | Learned checks |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic program | 12 | 15 | 14 | 157 |
| Constraint assignment | 17 | 18 | 18 | 164 |
| Evidence retrieval | 5 | 5 | 5 | 148 |
| Experiment plan | 0 | 0 | 0 | 121 |
| Finite-state plan | 64 | 64 | 64 | 64 |
| Graph route | 6 | 8 | 8 | 216 |
| Table transform | 8 | 18 | 18 | 214 |

The table-transform family contributed ten of the 16 available
rescues beyond first-check success. Experiment plan had no
success or rescue under this CPU source model, and finite-state
plan was already solved at its first visit in every observed
evaluation root. These are development observations, so a
policy chosen based on these outcomes must be tested on
new, root-disjoint data before its efficacy is evaluated.

## Next registered control

Compare the 192D ridge controller with a much simpler TRAIN-fitted
family-conditional rescue-rate prior. Use a separate CAL threshold,
the same verifier-call budget/first-valid controls, and fresh
suite seed 20261126 with 16 independent roots per family and split.
Include a seeded matched-continuation random gate, and keep
zero CAL missed rescues as the initial predeclared criterion.
The family-rate model must receive no ordinary/OOD outcome
data during fit or threshold choice.

The new external Lean/MCP checker interfaces remain source-only:
a kernel or private MCP effect judge must reverify every candidate
before any historical/fresh training labels can be admitted.
