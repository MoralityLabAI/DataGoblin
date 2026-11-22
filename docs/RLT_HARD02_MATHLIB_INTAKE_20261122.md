# RLT-HARD-02A: pinned Mathlib-native state intake

2026-11-22. **Development pilot for issue #24.**
This stage uses newly generated theorem source files with real Mathlib
definitions, an actual Lean open-goal trace after a registered prefix, and
independent kernel checking of completed scripts. These fresh derived
theorems are **not** an existing Mathlib benchmark or proofs extracted from
Mathlib's historical source theorems.

## Provenance and split contract

* Mathlib tag: \`v4.12.0\`; immutable commit
  \`809c3fb3b5c8f5d7dace56e200b426187516535a\`.
* Toolchain: \`leanprover/lean4:v4.12.0\`.
* Registered source seed: \`2026112202\`; generator SHA recorded.
* Source groups: TRAIN = Set image/subset; CAL = Finset cardinality;
  ordinary development = List length/append; OOD development = Nat
  divisibility. Each generated statement lives in its own source file.
  The upstream source modules are also disjoint across the splits.
* Each of the four splits has eight newly generated theorem roots,
  each with a full five-action menu. Two roots per split intentionally
  omit the candidate witness, producing a potential exhaustion control.
  The independent kernel result decides whether any alternative succeeded.
* The source-root/statement hashes, Mathlib module bytes, Lake manifest,
  pinned Lean executable, toolchain version, script/source names and actual
  compiler receipts are saved to the CI artifact. No material from
  historical Lean-CGRT, previous evaluation or sealed cohorts is used.

## Experimental question and controls

A registered prefix introduces the variables and hypotheses into a genuine
Lean proof state. The next candidate is then applied with the exact pinned
Mathlib import. A separate \`trace_state\` compile records the open goal
before training and includes its source SHA-256. A successful candidate
is independently recompiled under \`#print axioms\` and checked with
DataGoblin's existing \`LeanKernelChecker\`.

The ranker uses the project’s existing
\`rlt_lean_train_rank.features\` representation, fitted with NumPy on
**TRAIN-only real verifier labels**. Its available candidate vocabulary
includes exact, apply, simp, rfl, constructor and source-specific rcases.
An induction candidate or a true state-by-state tactic generator is beyond
this initial fixed-menu version. This representation is expected to
transfer poorly across the intentionally family-disjoint source groups.

The CAL cohort chooses one of the preregistered stopping thresholds
\`[.05, .15, .30, .50, .70, .85, .95]\` by
\`kernel_certified_roots - .03 * attempted_candidates\`. Tie-breaks
favor the larger threshold. No development held-out labels influence
the fitted parameters or threshold.

Evaluation policies execute actual compiler checks at a matched
five-candidate-per-root maximum:

- learned ranking with CAL-frozen continuation threshold
- fixed frozen-menu order
- deterministic randomized menu order
- rfl/simp/constructor-first menu order

Each run records kernel-certified roots, actual candidate counts and
candidate wall seconds. Every learned evaluation attempt is separately
recompiled and its verifier receipt compared. There is no sampled
chain-of-thought or model-generated proof claim.

## Hardness / promotion gates

The pilot must record the number of roots discharged by the
trivial-first control, separately for each split. If the simple control
already solves many roots quickly, the generated cohort fails the
hardness gate. The report must also distinguish genuine kernel success,
negative attempted scripts and menu exhaustion. A negative menu does
not imply unprovability.

**Issue #24 remains open** until a later experiment achieves source-
and theorem-lineage-disjoint Mathlib or LeanGym tasks with a genuinely
challenging 8+ roots/split hardness gate, goal-level transitions chosen
by a trained policy, matched online compute/wall-time budgets,
a meaningful learned-versus-controls gain on fresh registered holdout,
and separately source-blinded MCP lookup when retrieval is enabled.

The current Mathlib-native pilot makes progress on pinning, state
intake, direct use of the existing ground-ranker features and kernel
replay. It should be reported as a failed or partial intake whenever
the gates are unmet. It is not sealed confirmation.

## How to reproduce

\`\`\`bash
git clone https://github.com/leanprover-community/mathlib4.git /tmp/dg-rlt-mathlib
git -C /tmp/dg-rlt-mathlib checkout 809c3fb3b5c8f5d7dace56e200b426187516535a
cd /tmp/dg-rlt-mathlib
lake exe cache get
cd /your/databoglin/worktree
python -m datagoblin.training_gym.rlt_gym.hard02_mathlib \
  --lake "$HOME/.elan/bin/lake" --mathlib /tmp/dg-rlt-mathlib \
  --per-split 8 --source-seed 2026112202 \
  --out runs/rlt-hard02-mathlib
\`\`\`

The workflow \`.github/workflows/rlt-hard02-mathlib.yml\` installs the
toolchain and verifies its exact source commit before evaluation.
All generated source, complete verifier results and certified proofs
are retained as development artifacts. Run-time caches and model weights
stay out of Git.
