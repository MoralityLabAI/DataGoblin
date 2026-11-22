# RLT-HARD-02E: learned stopping with kernel-verified negative List menus

**Development registration:** November 22, 2026. Source seed `2026112247`.
Related: [DataGoblin issue #24](https://github.com/MoralityLabAI/DataGoblin/issues/24),
[RLT-HARD-02A/D PR #28](https://github.com/MoralityLabAI/DataGoblin/pull/28).

## Scientific question

RLT-HARD-02C found a useful candidate-content feature ablation on generated
Mathlib-dependent Nat divisibility theorems: a content-aware ranker certified
6/8 OOD roots with 17 kernel candidate checks, versus 23 for the paired
legacy ranker. A separate List development cohort had an invalid exhausted
menu construction. RLT-HARD-02D repaired it and verified 6 positive and
2 exhausted roots, with 40 independent candidate replays and explicit
native Lean axioms.

This experiment measures learned **rank and STOP** decisions on freshly
registered List proof tasks whose entire five-action candidate menus are
exhaustively verified for every split. Negative controls are established
from the **actual Lean kernel** rather than inferred from a teacher's
omission flag.

## Pinned data custody

Mathlib source commit:
`809c3fb3b5c8f5d7dace56e200b426187516535a`,
Lean 4.12.0, and upstream `Mathlib/Data/List/Basic.lean`.
All source theorem identities and file bytes are SHA-256 registered and
saved. There are four source-file-disjoint splits with eight roots each:
TRAIN, CAL, ordinary development and OOD development. Repeated-source
lineage is prevented by the literal generated theorem statement and its
per-root source file. The mathematical **family is shared** across splits,
and that constraint remains a known limitation.

The source task states a length equality preserved when the same literal
List suffix is appended to both List Nat arguments. TRAIN/CAL/ordinary
suffix lengths vary from one to five elements. OOD lengths vary from six
to eight. The sole pre-authorized successful candidate is:

```lean
simp only [List.length_append, List.length_cons, List.length_nil]
exact congrArg (fun m : Nat => m + K) h
```

Here K is the actual literal suffix length. Each registered source menu
contains five one-digit `K` candidates of the same byte length. Six
roots per split intentionally include a correct K, while two omit it.
The kernel must verify that there is exactly one accepted candidate in
each positive menu and zero accepted candidates in each intended
exhausted menu. These labels are evaluator-owned and absent from the
public task passed to the ranker.

Each root receives a native `trace_state` check after the authorized
`intro xs ys h` prefix. Every candidate in every split is then
compiled through real `lake env lean` and a subsequent
`#print axioms` source/kernel check. The full QA ledger for held-out
splits is reserved strictly for verification and audit. Neither learned
ranker nor the CAL-frozen STOP policy receives it.

## Preregistered policies and controls

TRAIN alone fits the previously implemented NumPy logistic rankers:
the 208-feature position-invariant signed-hash content ranker and the
legacy position/first-command ranker. CAL alone selects one threshold
for each using the previous fixed grid and utility
`certified - 0.03 × candidate checks`.

Ordinary and OOD run independent native compiler checks, limited to
five candidates per root, for six arms:

* Content-aware learned ranker with learned STOP threshold
* Legacy-grounded learned ranker with separately calibrated STOP
* Fixed full search
* Deterministically shuffled random full search
* Fixed first candidate
* Explicit syntax-based suffix-length rule

The last control is intentionally strong. It reads the public literal
suffix and the numeric K in the candidate proof. On a menu without a
matching K, it may STOP before any native candidate call. Any improvement
over fixed full or random alone is insufficient to claim learned value
if this cheap syntax rule achieves the same or better solve/cost result.

The online policies use only their registered inputs and actual
verifier outcomes. All their candidate receipts are independently
checked against evaluator-owned exhaustive menu receipts, and both
learned arms undergo extra fresh Lean source/kernel recompilation.
Record failed STOP where full search eventually certified the root,
candidate counts, execution time and actual kernel-verified coverage.

## Gates and interpretive boundary

A source-quality failure (a wrongly accepted decoy, missing positive
witness, or unexpected exhausted menu) fails the quality gate and must
be preserved in the native artifacts. A registered negative menu means
the five candidates are exhausted, never that the theorem is
mathematically unprovable.

This is a controlled, synthetic proof-template study on real Mathlib
and Lean 4. It establishes infrastructure, finite search and stopping
behavior. It does not establish free-form tactic synthesis, recurrent
goal transitions, hard theorem proof discovery, cross-family transfer,
or blinded MCP retrieval. None of the held-out sealed confirmations
from the broader research program are used.

The next hard-goal milestone is fresh Mathlib/LeanGym tactic-state
transitions with learned action generation and a hidden target-symbol
MCP retrieval oracle. Issue #24 remains open and PR #28 remains draft.

## How to reproduce

The dedicated `rlt-hard02e-verified-list.yml` GitHub workflow pins
Mathlib/Lean, checks CPU tests and Ruff, performs the bounded native
campaign, then independently audits source, verifier and replay receipts.
Source seed and population are fixed in the workflow. The generated
artifacts remain uncommitted:

```bash
python -m datagoblin.training_gym.rlt_gym.hard02e_list_stop \
  --lake "$HOME/.elan/bin/lake" \
  --mathlib /tmp/dg-rlt-mathlib \
  --source-seed 2026112247 --per-split 8 \
  --out runs/rlt-hard02e-verified-list

python scripts/audit_rlt_hard02e.py runs/rlt-hard02e-verified-list
```
