# RLT-LEAN-STOP-01: kernel-backed tactic ranker with CAL-frozen continuation

Date: 2026-11-22. Development protocol at pinned source seed 20261124.

## Motivation and custody

The merged PR #12 actual Lean kernel receipts had 32/32 eventual successes
from an intentionally elementary fixed menu, but among TRAIN first-failures
there were no exhausted candidate pools. PR #13 established a blind MCP
development generator and a failed-first/repaired second policy cohort.
This independent Lean pilot uses the existing DataGoblin LeanNativeSession
and LeanKernelChecker and leaves both previous evidence archives unchanged.

TRAIN fits a six? No: a thirty-feature, NumPy-only logistic tactic ranker
on **native kernel-verified per-script outcomes**, using source roots in
TRAIN exclusively. CAL sweeps an explicit threshold grid and freezes
a continuation decision before evaluating any ordinary/OOD source root.
The model sees the public theorem statement and registered candidate
script text. It never receives a root's generated solution flag,
future verification result, or private evaluation row.

Each split has eight freshly generated root statements. Six contain a
registered successful multi-command candidate script, and two have a
registered finite menu with only unsuccessful candidates. The underlying
equations remain true, so "menu exhausted" means no script in the
registered budget was accepted by the actual Lean compiler. It makes no
claim of mathematical unprovability.

Statements contain conjunctions combining distributivity,
associativity and commutativity over Nat. Candidate proofs must first
introduce bound variables, construct a conjunction and then discharge
two Lean goals using explicit rewrites or simplification. Five scripts
are shuffled per root, including wrong-algebra distractors. A
source-root-disjoint nonce and separate numerical constants prevent
root duplication. OOD uses larger numerals with the same proof families.
This is substantially more structured than single-line rfl positive
controls while remaining a controlled synthetic algebra benchmark.

## Policy and baselines

Ranker features are derived from public formula structure and tactic
lemma tokens, plus their pairwise interactions. Logistic model weights
come from native verified TRAIN labels, and are saved only in a CI
artifact alongside a hash.

CAL thresholds: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70,
0.80, 0.90, 0.95. CAL utility = kernel-certified roots minus 0.025
times candidate attempts. Ties prefer larger thresholds. This is an
explicit development-level preference for reducing futile search.
It neither uses ordinary/OOD labels nor opens sealed data.

On test roots the learned policy evaluates the most promising remaining
candidate through Lean, verifies successful proofs with a fresh compiler
process and an explicit #print axioms audit, and after any failure can
stop if its remaining candidate probabilities fall below the threshold.
A successful result is terminal only after kernel acceptance.

Matched development controls use separate actual compiler checks:

* fixed_one: first frozen-menu script only
* fixed_full: fixed menu until kernel success or all five attempted
* random_full: deterministic random permutation until success or five
* learned: TRAIN-fitted, CAL-frozen rank/continue policy

For every learned test attempt, a second independent Lean check
recompiles and compares its complete receipt. The campaign reports
per-arm source attempts and elapsed compiler wall time, plus the
full replay compiler count. The native evaluation results have
an execution cost rather than a simulated offline prefix metric.

## Mandatory review gates

The workflow verifies seed-disjoint roots and TRAIN/CAL custody in unit
tests, pins Lean 4.12.0 via elan, runs the actual compiler, and publishes
source receipts, a model hash and certified Lean files with SHA-256.
Results require a green native CI run and an inspected report.

Success is defined as the count of actual kernel-certified roots at
the specified candidate budget. The most important comparison is
verified success and real native attempt count versus fixed_full.
A failure to beat fixed_full remains a substantive negative result.
A failure in a whole proof family may indicate a bad template rather
than a ranker failure, so individual native proof-family checks must be
inspected before interpreting that label.

This development experiment does not establish learning on real
Mathlib problems, trained Lean language-model proof search, dynamic
tactic generation, or out-of-distribution theorem families. The next
gate is genuinely difficult unseen Mathlib/LeanGym states with
goal-level proof-state transitions, train/cal/eval source custody,
counterfactual cost controls and sealed confirmation.

## Reproduction

Run the workflow .github/workflows/rlt-native-lean-stop.yml or, given
an operator-supplied pinned Lean executable:

    python -m datagoblin.training_gym.rlt_gym.native_lean_stop \
      --lean /absolute/path/to/lean --per-split 8 --source-seed 20261124 \
      --out /new/empty/parent/rlt-native-lean-stop

A source directory may not already exist. The CI run keeps all proofs
in ephemeral workspaces until the archive is written and verified.
