# MOLT-10 exact response storage accounting

Companion to [LoopedTransformers #408](https://github.com/MoralityLabAI/LoopedTransformers/pull/408).
Recomputes the exact response-archive, native chart and fixed-head payload sizes,
checks the declared engineering parity gates and complete continuation roster,
and reports Pareto alternatives subject to an incremental payload budget.

Specify teacher and centered-head residency separately. Recompute is available
only with an already-resident complete teacher. A separately retained centered
head and teacher Gram must be charged. The planner uses measured engineering
arm wall time descriptively; those timings include parity audits and all
representations co-resident and are not comparable deployment forecasts.

    python -m pytest -q tests/test_molt_response_storage.py
    python -m datagoblin.training_gym.molt_response_storage --report /tmp/molt10/replay/receipt.json --cap-bytes 2400000 --out /tmp/response-plan.json

No model source identity, hard RAM cap, training equivalence beyond the supplied
receipt, or promotion is independently certified. All sizes exclude Python and
tensor allocator overhead, optimizer state, metadata and transient compute.
