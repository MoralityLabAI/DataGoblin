# BSAR-05: exact-premise recurrent correction gym

[Topher's BSA-R Step 7](https://github.com/VivienSweetbrier/bsa-r/blob/80c62433b0eb671175bd8d6169822d3ebd922b60/lean/Step7VorticityDirectionPdeRound6.lean) defines an inductive `LerayDoublingChain` with local integer premises. The [MathCan BSAR-05 audit](https://github.com/patrickdugan/MathCan/pull/175) independently kernel-checks an unbounded family of finite chains satisfying those rules, isolating the need for an extra uniform PDE budget.

This training gym generates **1,050 exact-integer examples** in **525 matched pairs** that ask whether one local chain step is admissible. The four checked conditions are:

```text
M > 0             (outer closure assumption)
d_next >= 0       (local nonnegativity)
M*(d_next-d_prev) <= I_m*d_next
2*I_m <= M
```

Each positive witness lies on the boundary of the latter two inequalities, with `M=2q`, `I_m=q`, `d_next=2*d_prev`. A paired counterexample alters precisely one of `d_next` or `I_m` to violate exactly one premise. Labels and the per-premise verifier trace come from Python's exact integer arithmetic. The generated dataset intentionally contains no floating point approximations and no externally authored truth claims.

## Split and oracle

| Split | Examples | Pairs | Numerical regime |
|---|---:|---:|---|
| Train | 600 | 300 | q=1..4, p=1..25 |
| Development | 180 | 90 | q=5..6, p=26..40 |
| Held-out test | 270 | 135 | q=7..9, p=41..55 |

The same English question appears with both labels across splits. The public evaluation artifact strips `gold_label`, `oracle_gates`, `parameters`, and `mutation_gate`. A constant-positive predictor scores 50% on examples and completes zero matched pairs, providing an immediate sanity check.

## Reproduce

```sh
python -m training.rlt_gym.bsar05.semantic_steps --out /tmp/bsar05
python -m pytest tests/unit/test_rlt_gym_bsar05_semantic.py -q
```

The manifest includes each output file's SHA-256 and the verified upstream BSA-R commit. Training examples retain the exact failed gate, allowing RLT training to learn the correction type. Held-out predictions should be scored by complete-pair correctness as well as per-example accuracy.

This gym evaluates integer premise recognition. It does **not** run a trained model, verify an analytic PDE estimate, or establish global regularity. A separate Lean kernel or theorem-prover oracle should check any future generated formal proofs, while new physical-analysis tasks should explicitly supply all required functional-analytic assumptions.
