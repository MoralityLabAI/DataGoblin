# Threat model

## Protected boundaries

- Evaluation leakage: evaluation seeds, labels, expected answers, and hidden metrics remain harness-owned.
- Builder/evaluator collusion: identities and writable roots are separate; builder tickets omit evaluator source.
- Proposer self-certification: claimed evidence is recorded separately from verified evidence.
- Metric mutation: split and success metrics are frozen before execution.
- Filesystem and symlink escape: command outputs are resolved beneath an assigned root and symlinks are rejected.
- Poisoned upstream artifacts: ingestion is selected, hashed, read-only, and retains upstream claim boundaries.
- Recipe reward hacking: Rat and validator inspect shortcuts, hidden metrics, cherry-picking, confounds, and permission expansion.
- Curriculum collapse: unique-fraction and diversity declarations are checked and reported.
- Solver bugs: independent verifier replay and property tests reduce risk, but do not prove solver correctness.
- False provenance: evidence authority and verifier identity are typed; self-authored environment evidence is rejected.
- Ledger tampering: canonical hashes and immutable IDs make accidental mutation visible, while administrators can still alter local storage.
- Accidental secret ingestion: denylist patterns exclude secrets, private trees, weights, checkpoints, caches, and common generated directories.

## Limitations

Process-level isolation is not a security sandbox. Environment allowlists do not prevent all OS-level behavior, and v0 does not provide containers, VM isolation, syscall filtering, network namespaces, or hostile-code execution guarantees. `CommandBuilderBackend` is therefore opt-in and inappropriate for arbitrary untrusted code. The trusted template builder is the only builder used by the default demo.

The constitution can detect declared and structurally visible violations; it cannot prove truthful declarations, model alignment, evaluator independence, or scientific validity.

## Sealed verdict-label custody

Public sealed-poison rows contain recipes and untrusted proposer hints but no decisions, violation masks, receipts, or
correctness fields. A separate validator invocation writes labels outside the public dataset. Evaluation checks example
and recipe hashes before merging. This prevents accidental prompt exposure but is not external organizational or human
custody; access controls and independent stewardship remain deployment responsibilities.
