# MeTTa-06: PLN-style lineage and EvidenceID shadow ledger

**Status:** original, CPU-only curation research utility. This implements a very small lineage-aggregation control inspired by [trueagi-io/PLN](https://github.com/trueagi-io/PLN), a MeTTa-native, MIT-licensed implementation with EvidenceIDs, Simple Truth Values (strength/confidence) and bounded inference. The original PLN runtime uses PeTTa; our shadow module does not import or execute it, and does not implement PLN's truth-value inference/revision calculus.

## Observations and controls

An input `EvidenceClaim` supplies a claim ID, declared independent lineage ID, observed source revision and a Boolean support/refute observation. The API accepts up to 4,096 distinct claims and 512 lineages. It treats repeated identical IDs as the same observation; multiple IDs from a shared lineage and revision contribute a single vote. If the same ID maps to different content, computation fails closed.

For each lineage, only its highest observed revision participates in the advisory count. Older revision observations are retained in the audit counts. A same-revision conflict triggers review and contributes neither a positive nor a negative independent vote. This is an **epoch-local experimental simplification**, not proof of remote source freshness. A change in the real source must be checked independently.

The shadow uses a nominal strength (fraction of independent included lineages supporting the proposition) and a nominal confidence (number of included lineages divided by that count plus a configurable positive equivalent prior). These illustrative values are **uncalibrated** and are not PLN-native truth values. They assume the caller's declared lineage identity is honest. A SHA-256 receipt binds the chosen lineage/revision/support states without printing raw IDs.

## Run

    python -m MeTTa.pln_lineage --out runs/metta/pln_lineage_shadow.json
    python -m pytest -q tests/unit/test_metta_pln_lineage.py

The fixture includes two copies of a positive observation from the same source and an independent negative observation. The source copies do not increase confidence.

## Authority

Every response contains `authorization=none`, `training_ready=false`, `native_pln_executed=false`, and `confidence_calibrated=false`. Real data admission stays with DataGoblin EtiquettePolicy/ResearchMembrane and the environment's independent verifier.

The next research connection is to compare motif-derived or rollout-derived evidence claims with identical source ancestry. An OOD evaluation should challenge independent-lineage assumptions, updated source revisions, incorrect claims of negative evidence, and deliberately conflicting source assertions. A native PeTTa PLN adapter and a proper calibration dataset are separate validation gates.
