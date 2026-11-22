# RLT-LEAN-STOP-01: kernel-backed tactic ranker with CAL-frozen continuation

Date: 2026-11-22. Development protocol at pinned source seed 20261124.

## Motivation and custody

The merged PR #12 actual Lean kernel receipts had 32/32 eventual successes
from an intentionally elementary fixed menu, but among TRAIN first-failures
there were no exhausted candidate pools. PR #13 established a blind MCP
development generator and a failed-first/repaired second policy cohort.
This independent Lean pilot uses the existing DataGoblin LeanNativeSession
and LeanKernelChecker and leaves both previous evidence archives unchanged.

TRAIN fits a thirty-feature, NumPy-only logistic tactic ranker
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

## First kernel-native development result

The original seed-20261124 native campaign passed
[Actions #44980029581](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44980029581).
It invoked the pinned Lean 4.12.0 compiler 300 times and independently
replayed all 16 learned-policy held-out attempts with identical receipts.
All 26 recorded file digests (including 24 certified proof sources)
matched the downloaded artifact. Accepted proofs carried either no
axioms or the explicitly audited `propext` dependency, which is
permitted by the repository's LeanKernelChecker.

TRAIN had 6/8 menu witnesses and CAL had 6/8. The CAL-frozen threshold
was 0.95. Development evaluation:

| Arm | Ordinary certified / attempts | OOD numeric shift certified / attempts |
| --- | --- | --- |
| Learned | 6/8, 8 checks | 6/8, 8 checks |
| Fixed full | 6/8, 28 checks | 6/8, 23 checks |
| Random full | 6/8, 31 checks | 6/8, 27 checks |
| Fixed one | 1/8, 8 checks | 2/8, 8 checks |

This is a valid first development gain on finite-menus: the trained
ranker identified all available scripted witnesses on the first try,
then stopped on the exhausted cases, using 16 checks instead of the
51 needed by the full fixed-order menu. The model's formula features
are literal, hand-written pattern predicates. This is a **known
potential shortcut**, and it provides no evidence of generalization
to reworded or genuinely difficult Lean theorems.

## Registered second development iteration

After inspecting the first result, the next study adds a structural
perturbation. Source seed `20261125`, per-split 8, and `--alpha-ood`
must be fixed before executing the new OOD tasks. On OOD only, the
public quantifier and bound occurrences rename `a b c` to `u v w`.
The registered tactic scripts still introduce local `a b c` through
`intro`, so the same theorem family and proofs remain in scope.
There is no change to the trusted Lean checker, candidate menu or
reward policy.

The hypothesis is that a ranker depending on literal string
features will fail to find some alpha-equivalent proofs. The new
TRAIN and CAL roots are disjoint from the previous cohort. This
follow-up is motivated by the v1 result, remains development-only,
and will be reported alongside the original evidence.

The second run's preregistered command is:

    python -m datagoblin.training_gym.rlt_gym.native_lean_stop \
      --lean /absolute/path/to/lean --per-split 8 \
      --source-seed 20261125 --alpha-ood \
      --out /new/empty/parent/rlt-native-lean-alpha-ood


## Native second cohort: variable renaming failure

The preregistered seed-20261125 run passed
[Actions #44980268127](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44980268127).
The artifact passed all 26 SHA-256 file checks, retained 24 certified
Lean proof sources and recorded 281 real compiler invocations. TRAIN
and CAL each had six witnessed and two menu-exhausted roots, with a
CAL-frozen stopping threshold of 0.95.

On ordinary development roots, the learned ranker certified 6/8 in
eight candidate attempts. On the **alpha-renamed OOD roots** it
certified just **2/8** in eight attempts. The full fixed-menu search
kernel-certified 6/8 on the same OOD cohort in 22 attempts, and the
random-full baseline also certified 6/8. This is a learned-stopping
failure: four available true witness scripts were missed after
changing the goal's bound variable spelling. It directly tests the
literal-formula feature shortcut identified after the first run.

## Registered third development iteration: alpha-canonical inputs

The third native cohort uses source seed `20261126`, with
`--alpha-ood` and the same frozen CAL utility and candidate menus.
The only policy-family change is a deterministic **public syntax
normalizer**: it reads the three bound Nat variable names and maps
them to the canonical names `a b c` before extracting the 30
feature components. Tactic scripts, the trusted checker, split
allocation, verifier costs, and the threshold sweep are unchanged.

A new unit contract requires two alpha-equivalent goal strings to
have exactly the same public features for every registered tactic.
The model still trains on fresh TRAIN roots and picks its threshold
on fresh CAL roots; ordinary/OOD labels remain development-only.
This is a post-failure engineering repair and a separately registered
study. Success would demonstrate an engineered invariance in the
input representation, with no claim that the logistic model learned
alpha-renaming on its own.

Workflow argument: `--per-split 8 --source-seed 20261126 --alpha-ood`.
The original v1 and v2 green-run artifacts are preserved.

## Native third cohort: canonicalized binder result

The source-seed 20261126 campaign passed
[Actions #44980413832](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44980413832)
with eight offline tests, Ruff, real pinned Lean 4.12.0 compilation,
313 native compiler calls, and 16 independent learned-attempt replays.
All 26 artifact SHA-256 checks passed; 24 distinct proof sources were
retained with their digests.

CAL selected threshold 0.95 again. Ordinary development certified 6/8
with eight learned checks, against 6/8 in 24 fixed-full checks. On the
fresh alpha-renamed OOD cohort, the binder-canonicalized learned policy
certified **6/8 in eight checks**, and the fixed-full policy certified
6/8 in 32 checks. It stopped after one failed attempt on each of the
two finite-menu-exhausted OOD roots.

The sequence now has three independently pinned development archives:
seed 20261124 (numeric OOD success), seed 20261125 (literal-feature
alpha-renaming failure), seed 20261126 (engineered alpha-invariance
restores matched proof coverage). These outcomes cannot be pooled as
a sealed confirmation, since the second and third interventions
follow explicit inspection of earlier results.

## Scientific interpretation and next gate

The observed gain is in **bounded action selection** for four generated
Lean Init theorem families. The data establish an actual trained
classifier and a CAL-frozen stopping rule; each accepted proof is
recompiled by the Lean kernel checker. The representation has
explicitly encoded variable-alpha invariance, so its success measures
the engineered input interface plus learned tactic-family matching.

The next experiment must sample genuinely hard, unfamiliar Lean
goals (for example, fresh Mathlib or LeanGym theorem roots) and
generate possible tactic actions from actual proof states. A
pretrained or trained tactic ranker must choose among native
actions, with matched verifier budget, separate positive and
exhausted branches, independent kernel replay, explicit source
custody and held-out goal-family evaluation. Those harder
theorem-solving and model-transfer claims remain open.
