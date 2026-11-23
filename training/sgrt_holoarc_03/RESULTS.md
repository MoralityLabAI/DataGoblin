# DG-SGRT-03: sequential history pilot result

**Run:** 2026-11-23, CPU-only public synthetic development.
**GitHub Actions:** https://github.com/MoralityLabAI/DataGoblin/actions/runs/45105770458
**PR:** https://github.com/MoralityLabAI/DataGoblin/pull/55
**Result:** INSUFFICIENT LEARNED HEADROOM. Continue HOLD on Aleph architecture promotion.

This experiment followed the negative three-seed DG-SGRT-02 candidate-coverage
study. It introduces chronological evidence between decisions, matched
candidate sets, and flat/noncommuting orthogonal transport. The two episodes
in each root present identical fictional evidence events in reverse order;
the exact interpreter guarantees their final states differ.

## Protocol

- Training: 72 unique lineage roots, 144 two-event episodes across three
  composition families; fourth family withheld. Every model trained for
  384 paired-root Adam steps with seed 23 and an identical 937-parameter
  tensor layout, initialization, loss, and example order.
- Evaluation: 48 independent roots, 96 episodes, including 12 roots from
  completely withheld event-pair compositions.
- Fixed candidate multiset: the same 16 world-state forecasts at each
  decision. One model prediction per observation, with independent
  simulator checking after each episode. Four checks per paired root,
  192 for every arm.
- Primary paired-root measure: both reversed-order final forecasts
  must be correct for a root to count.
- Supplementary: individual first/last step success, whether an arm
  distinguishes AB and BA, root-paired bootstraps versus direct
  recurrence and transport diagnostic invariants.

## Geometry implementation checks

| Invariant | Observed |
|---|---:|
| Maximum flat transport commutator Frobenius norm | 3.14018e-16 |
| Maximum curved transport commutator Frobenius norm | 5.54987 |
| Maximum flat/curved event trace mismatch | 1.77636e-15 |
| Maximum transport orthogonality error | 3.49010e-15 |

The flat event transports commute while retaining the curved
transports' eventwise eigenangle spectrum. Geometry tests validate
this construction. They do not establish task utility.

## Learned outcomes

| Arm | Both orders correct / 48 | First-step correct / 96 | Final-step correct / 96 |
|---|---:|---:|---:|
| bag (order-invariant negative control) | 0 | 13 | 8 |
| direct learned recurrence | 0 | 19 | 7 |
| flat matched transport | 0 | 20 | 6 |
| curved HoloArc | 0 | 31 | 6 |
| random spectrally matched transport | 0 | 19 | 8 |
| curved with transports swapped between evidence steps | 0 | 23 | 9 |
| environment-owned exact simulator | 48 | 96 | 96 |

All learned arms have zero exact paired-root success, giving a
degenerate [0,0]-percentage-point paired contrast against direct
recurrence. The test cannot rank architectures on the primary endpoint.
The curved head's 31 first-step successes may warrant further
engineering, but 6/96 final-step correct outcomes are at a
near-chance level for 16 legal states.

The order-invariant bag mode produced identical final logits for AB
and BA by construction and correctly failed the paired criterion.
The exact simulator attained the 48/48 ceiling. That validates the
pairing and scorer. It also shows the candidate task's answer is
computable when the intended transitions are applied correctly.

## Interpretation and next gate

The two-step learned state update failed to carry enough
compositional information to evaluate transport chronology. At this
difficulty, a geometric advantage is unidentifiable: successful
holonomy readout, commutator fidelity, or phase shuffling cannot
substitute for task competence.

Before another architectural comparison, run a separate
**headroom-calibration study** with a deliberately easier endpoint:

1. Supervise exact single-event transitions with the same training
   families, recording final-state accuracy before adding a second
   event; check that held-out event-state pairs are still distinct.
2. Evaluate binary named-state queries (e.g. whether the archive
   signal is active), registering query selection and balancing before
   training. Include history-aliased AB/BA pairs whose correct
   binary answers flip.
3. With sufficient base accuracy, restore multi-bit two-event
   predictions. Compare ordinary learned recurrence against flat,
   curved, and random transports at matched parameters, measured
   FLOPs, and paired root custody before attempting an adaptive
   storyworld experiment.

Changes motivated by these public results must be labeled
development, with further held-out seeds and task families for any
confirmatory mechanism claim. The Aleph architecture remains
unchanged. The running local Inky Mini service was untouched.

## Artifacts and replay

The CI job uploads a short-lived artifact named
dg-sgrt-03-public-history-development, containing public_eval.jsonl,
scorer_only.jsonl and report.json. The public stream has 96 rows
and no simulator truth. The scorer-owned stream has 48 paired
root rows with exact targets, per-arm outcomes and receipts.

The complete deterministic source and CLI remain in
src/datagoblin/training_gym/sgrt_history.py.
