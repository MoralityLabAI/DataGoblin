# ZKL-07: indexed proof-cost pilot (2026-11-25)

Original synthetic exercises: indexed predicate families over Nat, dependent rule instantiation, Peano terms with Add reduction. The environment contains typed hypotheses; the theorem is derivability relative to these assumptions. No upstream proof corpus or code is included. This remains a restricted fragment with unfolded proof trees and mandatory canonical index normalization. It does not implement a general dependent type kernel, Lean importer, shared-DAG optimum, or ZFC lower bound.

## Reproduce

From this directory:

    python -m pytest -q test_weighted.py
    python weighted.py --symbolic-only
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python weighted.py
    python lean_export.py --seed 10000 --limit 12 > /tmp/zklExercise.lean
    lean /tmp/zklExercise.lean

The neural run uses optional local CPU PyTorch: 901 parameters, 384 training problems, 12 epochs, seeds 171/173/179, training at six loops and evaluation at 3/6/12. Each of three public test splits has 128 problems, seed ranges separate from training: limit 12, limit 24, and limit 24 with rule offsets up to eight rather than five. No sealed benchmark or saved checkpoints. Search budget is 512 candidates per root. Ranker inputs include proxy local cost; exact dynamic programming supplies labels.

The score is structural rows + iota reduction rows + 2 * premise reads. Weights are illustrative, not measured Nova constraints. Each returned tree is independently replayed: rule identity, binding index, conversion trace, normalized premise type, backward references and final statement. Costs are recomputed from the accepted proof rather than candidate scores. Rules strictly increase the Nat index, so a topologically ordered DP exactly minimizes this additive tree objective among enumerated instantiations. Because every index expression is built from succ and affine additions and every normalization is explicit, the fixed normalization cost applies to the grammar; no claims about optimal conversion strategies in Lean are made.

## Measured aggregate

| split | unit DP optimum matches | weighted DP optimum matches | unit DP proxy sum | exact proxy sum |
|---|---:|---:|---:|---:|
| same | 42/128 | 128/128 | 9525 | 7883 |
| size | 34/128 | 128/128 | 29686 | 20592 |
| grammar | 23/128 | 128/128 | 19594 | 15162 |

Mean per-solved-instance cost ratio to the exact optimum:

| method | same | size | grammar |
|---|---:|---:|---:|
| unit DP | 1.232 | 1.511 | 1.402 |
| weighted DP 2 rounds | 1.322 | 1.465 | 1.545 |
| weighted DP 6 rounds | 1.021 | 1.259 | 1.158 |
| neural 3 loops, seed mean | 1.420 | 1.568 | 1.658 |
| neural 6 loops, seed mean | 1.442 | 1.480 | 1.609 |
| neural 12 loops, seed mean | 1.476 | 1.521 | 1.618 |
| exact DP | 1.000 | 1.000 | 1.000 |

All symbolic methods in the table solved 128/128 in each split. At six loops the three seeds solved 384/384, 383/384, and 381/384 (aggregated across splits), respectively, so solved-only ratios must be interpreted with coverage. Exact DP on the size split trades more structural applications (6305 vs 5336) and premise reads (1522 vs 1275) for fewer iota steps (11243 vs 21800). It reduces total proxy cost 30.6%. Neural recurrence failed to outperform six-round weighted DP, including on the size and grammar shifts. These are exploratory results, not promotion evidence.

Local focused tests: 30 passed, including 16 small exhaustive comparisons, 12 parametrized normalization checks, malformed certificate/goal cases and exporter shape. The Lean exporter emits a complete original theorem conditional on rule hypotheses with a recursive zklAdd matching the toy rewrite orientation; Lean was unavailable in the sandbox, so generated source awaits actual Lean 4 kernel replay. No paid compute. Repository-wide tests and lint remain subject to CI. No upstream code, private eval roots, or generated runs are committed.

## Next gate

Measure actual Nova constraint counts per rule family and authenticated memory depth in zk-rusto-no-bitcoino PR 34. Fixed-shape folding usually charges the same constraints even for inactive rules. Cost optimization must distinguish folded step count, per-step shape, proof representation, memory sharing and proving time. Re-evaluate weighted proof search once those costs exist; use exact symbolic DP as teacher. Any corpus integration requires provenance and independent verification under DataGoblin rules.
