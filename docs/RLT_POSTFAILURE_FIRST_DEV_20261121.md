# RLT-PF01: post-failure continuation value, seven-family development campaign

Date: 2026-11-21. Status: **negative development finding** in a source-replayed
CPU NativeRanker control, with no new LLM teacher or native Lean/MCP execution.

The first actual end-to-end [GitHub Actions run #44917750394](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44917750394)
completed on a real private-repo runner. The source artifact
datagoblin-seven-family-postfailure-dev retains the entire suite,
frozen NativeRanker checkpoint, all candidate trajectories, independent
exact replay and the selected controller. The source and output
manifests are SHA-bound.

## Frozen procedure

Generate a seven-family task suite with seed 20261121 and
per_family=3, with independent TRAIN, CALIBRATION, ordinary_test,
ood_test and unused sealed_poison partitions. The chosen 84 roots
are 21 in each of TRAIN, CAL, ordinary and OOD. The sealed split
was never entered into source collection or the postfailure controller.

Fit the source NativeRanker on TRAIN and collect all four visits
for the direct feedback schedule (training model seed=7). Disable
early verifier exits during the source collection, retaining a
full fixed-depth counterfactual for each task. Exact replay
recomputed 84 trajectories and 336 candidate/verifier checks.
There were zero censored episodes.

The post-failure controller sees the public source task,
the first failed candidate, the already returned checker code
and the remaining check budget. It trains a 192D shrinkage-ridge
value predictor on the 21 TRAIN roots and selects its threshold
on CAL with a zero-missed-future-rescue allowance. The selected
threshold was 0.6019778753020285.

Controls are fixed one-check, first verified by depth four,
learned continuation, and seeded random continuation matched
to the CAL continuation rate among first-step failures.

## Outcome: missed every later rescue

| Policy | Ordinary solves/21 | Ordinary verifier calls | OOD solves/21 | OOD verifier calls | Overall solves/42 | Overall calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| One check | 7 | 21 | 4 | 21 | 11 | 42 |
| Learned continue | 7 | 24 | 4 | 33 | 11 | 57 |
| Random continuation | 8 | 29 | 4 | 27 | 12 | 56 |
| First valid by depth four | 9 | 60 | 5 | 72 | 14 | 132 |

The full-depth model exposed two later verified repairs on ordinary
roots and one on OOD roots. The learned controller missed all three,
while consuming 15 more checker calls than the single-check floor.
The random rate-matched continuation captured one repair on ordinary
roots, and achieved a higher solve count at one fewer check.

This result fails the useful post-failure value-of-compute criterion
in this cohort. The result is consistent with very sparse positive
training examples and a high-dimensional controller trained on
only 21 source roots. The experiment does not establish the reason
for failure and cannot support a claim of generalizable RLT stopping.
Any model runtime saving is an offline counterfactual approximation;
only independently rechecked verifier-call counts are reported.

## Next registered experiment

Increase source-root counts to 16 per family in every split, use
separate fresh suite seeds 20261124 and 20261125, and retain the
same model seed, direct schedule, depth four, 192D ridge and
predeclared CAL missed-rescue allowances of 0 and 1 as two
policy arms. Evaluate each on untouched ordinary and OOD
development roots. Report per-family rescue counts, accuracy,
verifier-call cost, CAL selection and seed-to-seed heterogeneity.

Keep the exact one-check floor and rate-matched random control
as primary comparisons. A useful improvement must exceed
the baseline's verified answer count at a defensible check cost.
A stronger value predictor or per-family route should first earn
its keep on this enlarged development design.

New Lean kernel and MCP state-effect contracts are in
[external verifier notes](rlt_external_lean_mcp_verifiers.md).
Their source-only mock tests passed; native Lean/MCP experiments
remain outstanding.
