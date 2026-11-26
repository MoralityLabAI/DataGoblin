# MOLT-08: observer bundle accounting

Companion to the [nonlinear observer error study](https://github.com/MoralityLabAI/LoopedTransformers/pull/406). The earlier allocation solver selects resident maps by forecast time savings. This independent accounting pass selects the cheapest representation of a set of records sharing one fixed observable basis.

For each possible maximum stored rank K, independently select the cheapest eligible prefix rank for each record, or store the ambient vector. Add the full serialized map cost (including spectrum and binding digest). Compare against all-ambient storage. This is an exact finite optimization for a single shared map with additive payload costs. Mixed ranks share one prefix basis. It does not price reconstruction time, map construction scratch, Python overhead or per-record identity metadata.

At dimension 16 and rank 5, each scalar-bound record costs 48 B including its 8 B bound; the 812 B map breaks even after 11 records. Each five-output signed-interval record costs 120 B including two endpoints per output, requiring 102 records. Different roots cannot share a map just because their rank agrees. The existing MOLT-08 panel has only three dose records per reached-state group. It should retain ambient vectors where a map does not pay for itself.

```sh
python -m pytest -q tests/test_molt_observer_bundle.py
python -m datagoblin.training_gym.molt_observer_bundle --report /private/molt08.json --method interval16_scalar --out /private/bundle-plan.json
```

The report importer accepts the fixed development roster and uses recorded conditional nonlinear eligibility claims only. It rejects a reported failed sign audit, excludes the linear-only diagnostic, and refuses incomplete or duplicate dose/rank records. It accounts for actual distinct records without inventing reuse by duplicating them. Fixture model/root grouping is not a production binding proof. Input file hash is recorded; nothing here establishes a formal certificate, an enforced RAM cap, or actual model runtime savings. No held evaluation or promotion is authorized.
