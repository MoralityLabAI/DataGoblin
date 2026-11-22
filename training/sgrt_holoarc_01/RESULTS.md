# DG-SGRT-02 and 02R: learned low-rank development results

Date: 2026-11-23. Repository: MoralityLabAI/DataGoblin, PR #53.
Status: completed public synthetic CPU development, with an exploratory
three-seed follow-up. **Architecture promotion: HOLD.**

## What was trained

Five equal-capacity 2,197-parameter low-rank proposal heads trained on
96 public synthetic roots in arithmetic, bitwise and structural families.
The mixed-composition family was held out of fitting. Each head used 384
root-wise Adam updates, identical optimizer initialization/order within
each seed and identical candidate / checker budgets. The task-dependent
holonomy geometry was fixed; only the projections and residual gate
received gradients. Every learned arm presented eight distinct candidates
per 4-bit two-operation program-repair task to the same demonstration
checker; exact success required full 16-input semantic equivalence after
canonical selection. The underlying 1B Aleph model was not modified.

## Public initial pilot

GitHub Actions: https://github.com/MoralityLabAI/DataGoblin/actions/runs/45073156589

Optimizer seed 23; public evaluation seed 61001; 64 fresh roots, including
16 from mixed-family OOD.

| Arm | Whole-task solved / 64 | Mixed family / 16 | Candidate checker calls |
|---|---:|---:|---:|
| semantic_lowrank | 46 | 9 | 512 |
| flat_lowrank | 45 | 12 | 512 |
| holoarc_gear_lowrank | 44 | 7 | 512 |
| holoarc_random_path_lowrank | 44 | 9 | 512 |
| holoarc_step_shuffle_lowrank | 50 | 10 | 512 |

The shuffled-step maximum was a public-development selection. The candidate
sets changed between learned arms, and the original closed-curve ordering
showed no advantage over semantic_lowrank.

## Declared exploratory seed replication

Registry: [REPLICATION.md](REPLICATION.md). Frozen seed pairs before run.
GitHub Actions: https://github.com/MoralityLabAI/DataGoblin/actions/runs/45073354861

| Learned arm | 41 / eval 61011 | 67 / eval 61013 | 89 / eval 61017 | Pooled / 192 |
|---|---:|---:|---:|---:|
| semantic_lowrank | 49 | 49 | 40 | **138** |
| flat_lowrank | 49 | 45 | 40 | 134 |
| holoarc_gear_lowrank | 50 | 45 | 37 | 132 |
| holoarc_random_path_lowrank | 42 | 39 | 39 | 120 |
| holoarc_step_shuffle_lowrank | 43 | 44 | 40 | 127 |

All numbers are full-domain exact solves out of 64 in each column,
with eight proposed checker calls per root. The three evaluation seeds
produce 192 distinct root IDs. The corresponding pooled mixed-family
OOD totals are:

| Arm | Exact mixed OOD / 48 |
|---|---:|
| semantic_lowrank | 20 |
| flat_lowrank | 20 |
| holoarc_gear_lowrank | 21 |
| holoarc_random_path_lowrank | 22 |
| holoarc_step_shuffle_lowrank | 21 |

The source workflow emits per-seed root-paired bootstrap intervals and
win/loss/tie counts versus semantic_lowrank, alongside JSONL-level
results and short-lived runner artifacts.

### Interpretation

The first-run shuffled-step lead did not persist: it trailed the
semantic scorer on the next two seeds and tied it on the third.
The primary gear arm solved 132/192 compared with 138/192 for
semantic_lowrank. Over the three declared replication runs the semantic
low-rank control had the highest total.

The synthetic families remain narrow and structurally related. The
geometry is a frozen task-conditioned transform, not a trained connection.
An eight-step circuit with fixed candidate ranking cannot measure online
reasoning that adapts after each observed outcome. Step reordering also
changes the learned candidate sets; an advantage from such a contrast
would not isolate the causal effect of visit order independently of
candidate coverage.

The experiment tests whether a fixed geometric prior helps a small
proposal ranker in a verifier-backed setting. It supplies no positive
evidence for integrating HoloArc into the 1B Aleph architecture. All
results, including failed and null contrasts, should be retained.

## Next prospective test

Develop DG-SGRT-03 only after its separate protocol is frozen:

- Build a genuine sequential task where independent feedback arrives
  between proposal visits, with task/state snapshots and typed verifier
  receipts.
- Preserve the same *multiset* of available candidate rewrites for all
  policies. Cross visit order with final candidate set so coverage and
  chronology are independently identified.
- Use real task-conditioned learned transport and an equally expressive,
  nongeometric learned recurrence. Match recurrent calls, checker calls,
  training data, parameter count and wall-clock cost.
- Group evaluation by independent storyworld program family. Measure
  verified discoveries, revision success after evidence changes, and
  abstention as well as holistic task completion.

The current best working baseline for this proxy remains a learned
semantic neighborhood followed by exact verification.
