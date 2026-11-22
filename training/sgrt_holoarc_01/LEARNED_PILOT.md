# DG-SGRT-02: learned low-rank HoloArc comparison

**Status:** CPU-trained public-development study. Add-on to DG-SGRT-01 draft
PR #53. No Aleph 1B weights or current Inky Mini process are changed.

## Objective and method

Test whether a task-conditioned HoloArc transport makes a learned candidate
ranking head more effective on executable programs than an equally trained
non-geometric semantic mapper.

The 4-bit, two-operator program-repair generator is inherited from DG-SGRT-01.
Public tasks carry one erroneous anchor program and four input/output pairs.
A separate evaluator owns the full 16-input target function. Only synthetic
training roots may provide full-domain positive rewrite labels to the trainer.
Learned candidate scores are computed from public task features and
candidate syntax/Fourier features, without executing candidates.

All five trained arms use the same 16-dimensional bilinear head:

    q = task_features @ A
    candidate_latents = candidate_features @ B
    gate = sigmoid(task_features @ w)
    z = q + gate * (H(task) - I) @ q
    score(candidate) = dot(candidate_latents, z) / sqrt(16)

A, B and w are learned by root-wise Adam, using a multi-positive exact-match
softmax objective; each arm trains the same number of parameters and uses
identical initialization, training-root order, learning rate and optimizer
updates. H(task) is a **frozen** public-task-conditioned transform, chosen
by the arm. The path and connection are not learned in this experiment.
Any learned-model result belongs to the low-rank proposal head plus its
frozen geometric prior.

## Registered arms

| Arm | Transport |
|---|---|
| semantic_lowrank | Single task-conditioned direct orthogonal transform, with no circuit |
| flat_lowrank | Closed six-gear trajectory with commuting generators |
| holoarc_gear_lowrank | Closed six-gear trajectory with noncommuting generators |
| holoarc_random_path_lowrank | Smooth random closed Fourier path matched for sampled arc length |
| holoarc_step_shuffle_lowrank | Same gear edge-step multiset in a shuffled order, still closed |

The earlier rigid, heuristic and repeated-edit proxies are included in the
evaluation but are marked as untrained comparators. All trained arms score the
same 18 possible edits and pass the eight highest-ranked distinct candidates
to an independently owned exact demonstration checker. The full-domain
evaluator then scores canonical selected survivors for real program
equivalence; merely selecting a demonstration-consistent candidate is
insufficient.

The gear, random and shuffled paths use the same registered radius multiplier
(3.0) and eight microsteps. Transport fiber has dimension 16, while the
parameter base has dimension 12.

## Public-development split

Training seed 31001 covers roots of arithmetic, bitwise and structural
families; mixed is withheld at every step. Evaluation seed 61001 produces
fresh roots across all four families. The report separates mixed-family
out-of-distribution items from new roots belonging to seen families.

Defaults:
- 128 generated training roots; 96 retained by the family split.
- 384 Adam updates per arm, with learning rate 0.025 and one root per update.
- 64 evaluated roots, of which 16 belong to the withheld mixed family.
- 8 candidate evaluations per trained arm and non-rigid control.
- Seed 23 controls paired initialization and optimizer ordering.

This is exploratory **public** development. The program DSL, seeds and
family generator are transparent. They cannot establish sealed generalization
or confirm a gain on authentic research/creative tasks.

## Run and verify

    python -m pytest tests/unit/test_sgrt_learned.py tests/unit/test_sgrt_holoarc.py -q
    ruff check src/datagoblin/training_gym/sgrt_learned.py tests/unit/test_sgrt_learned.py
    python -m datagoblin.training_gym.sgrt_learned --seed 23 --train-roots 128 --updates 384 --eval-roots 64 --budget 8 --out runs/dg-sgrt-02-public

The separate CI workflow executes the public development run and attaches
its report and individual evaluation scores. Local output directories
must start absent or empty; model weights exist only in process memory, and
their SHA-256 digest appears in report metadata. No network requests,
external teachers or GPU are needed.

## Result interpretation

Primary development metric: independently checked whole-task success at
the same candidate-check budget. Record target-reachable candidate sets,
full-domain exact solves, checks per root and per solve, task-family grouping,
and paired per-root deltas. The order-shuffled edge path is a mechanism
control; a benefit over the semantic mapper and smooth-random path would
justify a new confirmation plan. An apparent advantage over a commuting
flat path alone is insufficient to claim geometric utility.

The baseline's task transform is a single learned-gated direct rotation and
shares the same head parameter count. The geometric arms may have greater
computation cost, which a later end-to-end trained-Aleph experiment must
measure and match. This pilot does not itself equalize wall-clock or
floating-point operations across arms.

A genuine adaptive path-order claim needs an online policy receiving feedback
between visits. This benchmark ranks a fixed pool once, after an excursion.
Retain all negative and null outcomes, including a possibility that task
labels do not support any useful HoloArc benefit.

## Promotion boundary

After development, freeze the rank, gates, a true trainable task-conditioned
connection, regularization, total model FLOPs, independent families, and
experiment seeds. Compare with task-trained semantic and capacity-matched
non-geometric controls from shared optimizer initialization. Keep all
evaluation labels with a separate scorer. Run the first real model experiment
with a frozen host and a single low-rank boundary sidecar, before any Aleph
operator swap.

The extant Inky Mini llama.cpp server remains outside this public CPU study;
an authorized live experiment must reuse it without restart or paid fallback,
with 12k default and 16k hard context constraints.
