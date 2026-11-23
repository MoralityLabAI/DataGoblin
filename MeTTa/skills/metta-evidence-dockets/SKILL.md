---
name: metta-evidence-dockets
description: Build or verify Storyworld proof obligations, observer-scoped receipts, shared-root evidence dockets, and bounded joint verification cost.
---

# Plan and verify evidence dockets

1. Bind the query and candidate to the public observer, epoch, goals, and
   proof depth. Identify the admissible evidence and observed receipts for
   that scope. Preserve root IDs when two references share one source.
2. Obtain candidate plans for the whole docket. Compare plans by their
   distinct outstanding obligations and the applicable costs. Shared checks
   are charged once under the joint budget. A per-goal minimum can miss a
   cheaper joint plan. Keep proofs and their underlying root sets attached.
3. Use the native docket or receipt gate and an independent reference when
   available. Check cross-observer and stale receipts, an impossible goal,
   and a plan that exactly meets the cost limit. Where the query contract
   returns an obligation set, `()` can be a valid already-satisfied result.
   An empty result stream supplies no usable plan.

In the Storyworld checkout, use `engine/evidence_docket.metta` and
`engine/evidence_docket_costs.metta` for the contracts. Run the tests with
Hyperon available:

```sh
python -m pytest -q tests/test_evidence_proofs.py
```

Return each goal, its proof or failure, the distinct outstanding checks,
scope, cost, and verification status.
