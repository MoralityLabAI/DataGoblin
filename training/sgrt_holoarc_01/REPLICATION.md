# DG-SGRT-02R: exploratory independent-seed replication

**Frozen before the replication runs:** 2026-11-23. This follows the
first public-development evaluation on optimization seed 23 and evaluation
seed 61001. Results are exploratory and cannot be described as sealed
confirmation. The initial five-arm scoring design and all architecture
weights remain as in DG-SGRT-02.

## Motivation and declared seeds

The first run had 46/64 exact solves for the semantic low-rank control,
44/64 for original curved gear HoloArc, 45/64 for flat transport,
44/64 for smooth random arcs and 50/64 for shuffled gear edges. The 50/64
descriptive maximum motivated this follow-up; it is a selection on a
public-development outcome. An independent confirmation would need a
further external evaluator and a fresh registration.

Prospectively run exactly three additional paired matrices:

| Matrix | Optimizer initialization/order seed | Fresh evaluation seed |
|---|---:|---:|
| R1 | 41 | 61011 |
| R2 | 67 | 61013 |
| R3 | 89 | 61017 |

Each matrix trains on the same 96 training roots (train seed 31001),
uses 384 steps/arm at fixed learning rate 0.025 and compares the five
learned arms and the three untrained controls. Every learned arm receives
8 proposed candidate checks on each of 64 fresh roots, with a separate
16-root mixed-family OOD stratum. Independent evaluation root IDs differ
across R1/R2/R3 and from the first public run.

## Primary descriptive comparisons

- Whole-task exact solves with canonical demo-consistent selection.
- Root-paired difference in exact solves (percentage points) for each
  arm versus semantic_lowrank, with percentile 95% bootstrap over
  independent evaluation roots and counts of per-root wins/losses/ties.
- Report family-OOD and seen-family fresh-root strata separately.
- Keep target-reachable coverage distinct from canonical whole-task
  success and record verifier calls at the same eight-candidate budget.
- Report every seed and its pooled counts; avoid winner selection by
  post-hoc seed averages.

If shuffled-step still exceeds the semantic baseline in all three runs,
register a *new* matched candidate-set and online path-history test.
If it loses or varies sharply, retain the mechanism null and focus on
learned semantic neighborhoods plus exact verification.

## Reproduce

    python -m datagoblin.training_gym.sgrt_learned --seed 41 --eval-seed 61011 --train-roots 128 --updates 384 --eval-roots 64 --budget 8 --out runs/dg-sgrt-02-repl-41

Use corresponding seed and output pairs for 67/61013 and 89/61017.
The GitHub Actions replication workflow emits the per-seed summaries and
root-level, scorer-owned outputs as short-lived artifacts. No GPU or
external language-model endpoints are involved.
