# ALEPH-STATE-10 | proof transport, dependency custody and search controls

2026-11-25. CPU-only diagnostic following STATE-09. Keep direct Aleph A and both paid slots frozen. No native model training or historical checkpoint continuation.

## Verification status

The sandbox had no Lean/lake/elan executable, and a bounded request to the Lean release URL failed DNS resolution. The executed verifier is an independent explicit-term type checker for an acyclic propositional Horn fragment, not the Lean kernel. No positive Lean training labels may be issued from these receipts. The generator emits 168 explicit Lean theorem terms (72 anchor and 96 held goals), with no sorry, and 72 invalid transport attempts. Optional kernel replay is provided, unexecuted. The experiment does not establish that Lean elaborates the generated terms. No GitHub workflow changes or paid compute.

## Design

Ground atomic propositions, local facts and local rules with arrow types P1 -> ... -> Pk -> Q. Generate three disjoint chains of depth 6, 10 or 14, with two inaccessible decoy rules at each chain conclusion. Each of 24 runs cycles through one of these three structures and varies candidate order by seed 2026112516+i. The 24 runs are search-order replicates, NOT 24 structurally independent theorem environments. All six task families are composition/context changes relative to the warm anchors:

- join: combine two derived anchor facts;
- nested: combine three via an intermediate join;
- weaken: add unrelated declarations and a new fact used in the join;
- repair: mutate a used rule under its original name and add an alternative derivation;
- rule_removed: mutate the used rule without a repair;
- fact_changed: change a used fact's proposition under its original name.

There are 144 obligation instances: 96 derivable, 48 underivable, as determined independently by least-fixed-point Horn closure. No held target is used to build the warm cache. Cache only three anchor proofs per run. Reuse checks every referenced declaration's original signature against the current context, stopping at the first mismatch. If invalid, fall back to the same indexed backward search. The cold control uses memoized goal-directed search with candidate lists indexed by conclusion. The stronger forward control uses a premise-indexed worklist and fires rules when all premises are available. All policies emit explicit proof terms checked by a separate recursive type checker.

A search-work unit is an indexed fact lookup, one backward rule probe, one forward premise-edge probe, or one cached dependency lookup. These are accounting proxies, not equally expensive operations, FLOPs, wall time or Lean kernel calls. Index construction, cache construction traversal, hashing, memo lookups and final verification are excluded from the budget; final proof-check node counts are logged separately. The full cache warm-up costs 2,982 charged units and 792 final proof nodes. The same final proof check is required in every arm. No timing or kernel-call advantage is claimed.

## Results

| Budget | Cold /96 | Reuse /96 | Forward /96 |
|---:|---:|---:|---:|
|16|0|8|0|
|32|0|48|32|
|64|23|89|96|
|128|84|96|96|
|256|96|96|96|

Every policy had zero false accepts at every budget. At 256: cold 11,512 charged units, reuse 6,068, forward 4,848, all with 2,520 final proof nodes across accepted tasks. Reuse eliminates 47.3% of cold deployment search work, or 21.4% including the counted warm-up (9,050 total units). Forward search remains cheaper even without any warm-up cost. At budget 32, reuse leads forward on this synthetic distribution (48 vs 32 successes); at 64 forward solves all 96. No general superiority over symbolic search established. At 256 the reuse policy accepts 192 cached subproofs and rejects 72 invalidated cache candidates; direct independent checks reject all 72 invalid old target constructions (including 24 repair contexts). Underivable goals are established by the finite closure oracle, not by a timeout in proof search.

Total measured CPU process time for the local run, including proof generation and most output serialization, 0.433027 seconds. Python 3.13.5. Zero paid credits.

## Mathematical transport lemma

Let Γ be a finite Horn context assigning types to named local facts/rules, and let π be an explicit proof tree of an atom A. Write supp(π) for the named declarations occurring in π, and sigΓ(r) for the ordered premise list and conclusion of r. If Γ proves π:A and Γ' agrees with Γ on the signature of every declaration in supp(π), then Γ' proves π:A. Prove by structural induction: leaves retain the same atomic fact type; application nodes retain the rule signature, and inductively their arguments retain the required types. No condition on unrelated declarations is needed in this fragment. Consequently a context-wide digest is safe as a conservative cache key but can miss legal weakening; a support digest allows finer reuse. Signature disagreement invalidates this particular transport even if a different proof exists. The repair families explicitly test that distinction.

For arbitrary Lean dependent types, the dependency closure must include types of referenced local declarations, universe parameters, global constants and definitions used by elaboration/definitional equality, environment revision, and any relevant binder identities. The Horn support theorem does not authorize skipping a Lean kernel check. A SHA-256 fingerprint is a cache integrity mechanism, not a proof of entailment. Keep goal, environment and proof hashes and actual verified terms. Never equate a name match or a hash match on truncated context with valid proof transport.

## Replay and integration

    python training/aleph_next/state01/state10.py --out /tmp/state10
    python training/aleph_next/state01/state10.py --out /tmp/state10-lean --lean

The second command needs an installed, pinned Lean toolchain on PATH. It checks the generated positive file, then for each negative test first checks the same-context trivial theorem and separately requires the invalid transport to fail with a type-mismatch diagnostic. A negative diagnostic format mismatch will fail the replay gate; report and audit rather than suppress errors. Record the actual lean --version. The current code emits lean_kernel_verified=false in the proxy receipt even if separate replay subsequently succeeds, so the consumer must explicitly examine STATE10_LEAN_REPLAY.json. Do not ingest generated rows as kernel-verified before replay and a custody audit.

Executed source SHA256 e997bbe6f02cb86e323725fb6efc2606ce59b81a720c8265615ba61a7eb4a357. Git blob c90c2fc4e20a88e90ab0968402b3cfffa60a656c matches the local executed source. Full proxy receipt SHA256 5b69bd7a9391657d433f5d3611fcd9f0df6dc1f95980c52792b49644fdb036b4 (timing dependent). See STATE10_SUMMARY.csv.

DataGoblin already checks positive Lean action labels for kernel_ok, closed, exactly one kernel call and an allowed axiom set in training/aleph_rlt_placement/corpus.py. Preserve those gates. A next experiment should use genuinely held Lean goal families from an independently locked environment, rank cached proof support or tactic suggestions against strong indexed/forward controls where applicable, run actual Lean, charge measured checker time, and separate useful context weakening from stale dependencies. No paid model training authorized.
