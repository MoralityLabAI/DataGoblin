# RLT-HARD-02D: genuine kernel-negative List tactic controls

Date: 2026-11-22. Development QA under DataGoblin issue #24.

## Reason for the new gate

Native RLT-HARD-02B (Actions #44989821104) exposed a failed ordinary
negative-control registration. All eight List roots were proved by the
first candidate, including both source cases intended to have exhausted
menus. The old `simpa [List.length_append] using congrArg ...` form
accepted several numerically altered congruences because the simplifier
could still derive the target equality. Native outcome evidence determines
actual menu coverage. Generator witness-omission flags are only hypotheses.

## Frozen root generation

Source seed: `2026112219`. Exact upstream Mathlib v4.12.0 source commit:
`809c3fb3b5c8f5d7dace56e200b426187516535a`.

Eight fresh List append/length source files each supply five distinct
tactic scripts. Their full UTF-8 byte lengths match. Every candidate
first normalizes the goal with:

```lean
simp only [List.length_append, List.length_singleton]
exact congrArg (fun m : Nat => m + K) h
```

Here K is a single decimal digit. The expected positive witness has
K=1, corresponding to the appended singleton's length. Candidate decoys
use K=0,2,3,4,5. Six roots include the witness; two deliberately omit it.
The scripts use explicit Lean proof terms after controlled normalization
to prevent broad simplifier recovery from invalid candidate expressions.
These are registered positive/negative *menu* controls, and the
underlying propositions are mathematically provable in all eight cases.

## Native verification and gate

The runner records the real Lean open goal after the authorized prefix.
Each of the five candidate scripts is compiled through pinned
`lake env lean`, with DataGoblin's `LeanKernelChecker` checking every
successful proof using a fresh process and an explicit allowed-axiom audit.
All candidate receipts are independently replayed; successful proof
sources, compiler error tails, source hashes and verifier receipts
are preserved as a create-only GitHub Actions artifact.

The quality gate requires exactly one kernel-certified candidate in
each of six expected-positive roots and zero accepted candidates in
each of two expected-exhausted roots. A violation fails the native
Actions job while retaining the diagnostic artifact. It must be
repaired before another learned Mathlib stopping claim is promoted.

RLT-HARD-02D is solely a proof-state/candidate-quality engineering gate.
It fits no model, proves no open mathematical theorem, and opens no
sealed evaluation roots. Fresh domain- and theorem-lineage-disjoint
hard tasks and source-blinded MCP retrieval remain requirements of
issue #24.

## Reproduction

Use `.github/workflows/rlt-hard02d-list-gate.yml`, or after acquiring
the exact Mathlib commit and Lean 4.12.0 toolchain:

```bash
python -m datagoblin.training_gym.rlt_gym.hard02d_list_gate \
  --lake "$HOME/.elan/bin/lake" --mathlib /tmp/dg-rlt-mathlib \
  --seed 2026112219 --roots 8 --out runs/rlt-hard02d-list-gate
```

Retained evidence is authoritative for pass/fail interpretation.
