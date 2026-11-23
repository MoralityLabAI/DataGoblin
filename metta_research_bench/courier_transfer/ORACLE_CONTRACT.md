# DG-METTA-12 | Oracle and native courier adapter contract

The native Storyworld/MeTTa runtime is the authority for courier state transitions. A courier state is an immutable ledger containing the current location, independent witnessed-source IDs, held permits, carried parcels, and completed deliveries. Typed actions are `move(node)`, `inspect(witness)`, `permit(id)`, `pickup(id)`, `deliver(id)`, and `STOP`. Permits may require distinct independent source identities. Two aliases backed by one source count once. An episode succeeds only when every required parcel was delivered and the policy emitted a correct STOP.

## Required native adapter

Before any train or evaluation claim, fetch and pin the **full** Storyworld SHA for the locally reported courier kernel `4cfe171` and its exact Hyperon dependencies. Implement:

```python
class NativeCourierAdapter:
    def initial(self, root_id: str, root_seed: int) -> object: ...
    def legal(self, state: object) -> tuple[str, ...]: ...
    def verify_step(self, state: object, action: str) -> tuple[bool, object, str]: ...
    def delivered(self, state: object) -> frozenset[str]: ...
    def required_deliveries(self, state: object) -> frozenset[str]: ...
    def state_sha256(self, state: object) -> str: ...
    def public_features(self, state: object) -> object: ...
    def certify_macro(self, state: object, actions: tuple[str, ...]) -> tuple[bool, object, str]: ...
```

Every result must carry proof/transition receipts and an input-state digest. `verify_step` consumes a policy verifier call, including a rejected proposal. Grading checks are a separate meter. The Looped Transformer sees typed public features and the ordered allowed-action vocabulary; it never sees gold labels or the oracle value table at inference. No state mutation may occur on a failed proof.

## Exact set oracle

For remaining action horizon `h`, define a lexicographic Bellman value:

```
V(s,0) = (delivered_count(s), 0)
V(s,h) = max_lex(
    (delivered_count(s), 0),            # STOP
    { (V(step(s,a),h-1).deliveries, V(step(s,a),h-1).negative_steps - 1)
      for a in legal(s) excluding STOP }
)
A*(s,h) = { a in legal(s) : continuation_value(s,a,h) == V(s,h) }
```

The second coordinate prefers the fewest accepted steps for the best attainable delivery count, preserving multiple shortest successful action sequences. Keep only states where a full completion is attainable for successful-action training. Recheck every oracle label against an independent replay. Canonical A uses `min(sorted(A*))`; hashed B takes SHA-256 of `root_id|seed|state|sorted(A*)` and chooses an index modulo `len(A*)`; C uses the full set loss. The three fitted arms share the identical normalized legal-action logits. An optimal action set may contain more than one member while still representing a single supervised state presentation.

## STOP macro contract

The learned C checkpoint is reused byte-for-byte in D. With a shared ceiling of five **policy verifier calls**, D reserves up to two checks until the policy proposes STOP. A macro is a bounded atomic transaction: both typed state changes are certified against the native kernel, including their joint final state, before either is committed. Failures leave the original world unchanged. A native implementation lacking transaction/rollback support blocks D; a Python simulation may validate plumbing but cannot be counted as a native MeTTa repair. Record actual verifier calls, partial-check failures, macro attempts, certified and committed macros, and paired regressions.

## Reproducibility and release gates

Check split ID/seed disjointness and both oracle value and tie-set correctness. Require independent witness-source handling, permit ordering, graph topology, STOP, and atomicity tests. Pin the generator, MeTTa rule hashes, full native commit, tokenizer, protocol, source and training data hashes, initial-model hashes, checkpoint hashes, optimizer budget and the development report before any frozen run. Abort on changed inputs or attempts to tune on frozen labels. Frozen primary runs once per locked fitting cohort. `reference_wiring` is never conflated with `native` results.
