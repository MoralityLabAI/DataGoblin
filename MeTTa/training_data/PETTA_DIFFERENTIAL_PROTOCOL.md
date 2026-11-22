# MeTTa-08: PeTTa / Hyperon / independent finite reference

This public diagnostic compares identical ground relation programs across three
implementations on the restricted finite `match &self` subset established in
MeTTa-06. The primary purpose is to discover divergent evaluation semantics,
unexpected failure modes and runtime-specific parser behavior before extending
DataGoblin's MeTTa training candidates.

## Reproducible runtime identity

- PeTTa source: `trueagi-io/PeTTa`, pinned commit
  `ee5cacb6a6ecdd554a1c773aaf47e9a0f8a5af0f`.
- PeTTa launcher: the upstream `sh run.sh file.metta --silent`, using SWI-Prolog
  9.3 or newer in the upstream CI OCI image
  `ghcr.io/patham9/petta-ci:latest`.
- The workflow resolves the pulled OCI image to an **immutable sha256 digest**
  and uses that exact digest for execution. Both the digest and detected
  `swipl --version` are carried into every evidence row.
- Hyperon: Python package `hyperon==0.2.10`.
- Independent reference: `finite_semantic_oracle.py` computes deterministic,
  finite, ground-fact pattern unification without invoking either runtime.

The workflow fetches PeTTa's pinned Git source revision into a separate
working directory. Each PeTTa evaluation runs inside a disposable container
with networking disabled, a read-only source mount, limited memory and
processors, restricted process count, and a short timeout. The precise
source and interpreter identities remain in the artifact.

## Trial and result semantics

The initial trial covers 12 public source roots with four `match &self` queries
each, giving 48 query instances. The four families exercise branch enumeration,
repeated-variable equality, nested pattern projection and empty results.

Hyperon prints one bracketed result list per evaluation; PeTTa's `swrite`
prints a MeTTa expression of results, with `()` for an empty set. The two
formats are parsed separately into structural expression values.

All comparisons use the same alpha-normalized, order-independent result
multiset. Observed failure, output misalignment, parse errors and timeouts
become explicit statuses. The CI gate checks that the comparison ran and
that receipts are complete; semantic disagreements are preserved for
investigation and never silently converted into passing results.

The type-error pilot from MeTTa-06 is intentionally excluded until a
cross-implementation error taxonomy distinguishes PeTTa's compile-time
rejection from Hyperon's runtime `Error` atom. A uniform string comparison
would incorrectly classify these error models.

## Reproduce

```sh
python -m pip install 'hyperon==0.2.10' 'ruff==0.17.6'
python -m unittest discover -s MeTTa/training_data/tests \
  -p 'test_petta_differential.py' -v
# Fetch pinned PeTTa and pull the upstream image, resolving its digest:
python MeTTa/training_data/scripts/petta_differential.py \
  --petta-repo /path/to/pinned/PeTTa \
  --petta-image ghcr.io/patham9/petta-ci@sha256:<actual-digest> \
  --roots 12 --timeout-sec 25
```

All inputs and answers are public synthetic examples. Every observation
remains `training_ready=false` and `sealed_eval_eligible=false`, with
`model_evaluation_executed=false`. This establishes differential reference
behavior only for the registered finite fragment; it leaves the 187
source-conditioned SFT candidates and the sealed-key protocol unchanged.

The next experiment should add an independently pinned interpreter for
rewrite rules and controlled state transitions, with per-feature coverage
gates and mismatch minimization. Model-quality claims require actual
predictions on independently held source-root challenges.
