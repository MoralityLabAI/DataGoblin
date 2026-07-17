# Model training roadmap

The future Goblin is envisioned as an approximately 0.8B–1.3B specialist research control plane, not the primary learner. Size alone does not create research ability.

1. Collect supervised public research episodes: learner snapshots, Scout findings, falsifiable hypotheses, recipes, Rat critiques, decisions, tickets, outcomes, and concise interpretations.
2. Produce constitutional chosen/rejected pairs with deterministic corruptions and minimal repairs.
3. Train a violation-vector prediction head alongside schema-constrained proposal generation.
4. Add calibrated outcome and compute prediction heads using executed ledger records, including nulls and failures.
5. Retrieve analogous experiments from the ledger without exposing hidden evaluator material.
6. Enforce a structured output grammar and bounded repair loop at inference.
7. Train result interpretation and next-experiment selection against held-out outcomes.
8. Evaluate on held-out environment families, not merely new seeds from known mechanics.
9. Compare against deterministic heuristic search, Bayesian optimization, and larger-model planners under matched budgets.
10. Measure calibration, selective accuracy, abstention quality, false acceptance of violations, information gain, and realized resource error.

Promotion should require independent held-out evidence across environment families and regression tests of prior constitutional examples. Provider clients, weight training, and stronger process isolation remain optional future work.

