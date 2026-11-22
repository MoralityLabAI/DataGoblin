# BSAR-09: exact rational-window recurrent repair gym

**Source:** [MathCan BSAR-09 PR #188](https://github.com/patrickdugan/MathCan/pull/188), merged at `0ce1a3c071142c502a41670a15c6cbca6fe14162`. The original BSA-R Step 7 source pin is `VivienSweetbrier/bsa-r@80c62433b0eb671175bd8d6169822d3ebd922b60`.

The [vendored Fraction oracle](mathcan_exact_oracle.py) is copied byte-for-byte from the merged MathCan artifact so DataGoblin can reproduce its deterministic generator without runtime GitHub or network dependencies. Its task labels are **exact local integer/rational arithmetic**, not results from a trained model or Lean proof checker.

## Task

The model sees a sequence of `K` windows with `M>0`, `0<=I_m<=M/2` and exact rational endpoints. It must choose `SATISFIES` or `VIOLATES` for the original source ODE

```text
M * (Dnext - Dprev) <= I_m * Dnext
```

and propose an exact rational repair of the final endpoint when the ODE is violated. The verifier requires the **maximal admissible** final value:

```text
Dnext_repaired = M * Dprev / (M - I_m).
```

Changing `M`, `I_m`, or any earlier state cannot satisfy the repair gate. Choosing zero is also rejected as a shortcut, even though it could satisfy the local inequality.

All positive rows saturate the original ODE. Their negative partners change only the last endpoint to sit between the exact ODE frontier and the weaker BSAR-08 product envelope. The latter remains valid for both labels, so a policy needs to check the **actual source condition**.

## Custody and verification

| Split | Examples | Matched pairs |
|---|---:|---:|
| Train | 256 | 128 |
| Development | 64 | 32 |
| Test | 144 | 72 |

Training JSONL contains model-visible user prompts, exact oracle teacher responses and one-visit verifier receipts with the failed gate. Development records carry reference actions. Public test JSONL strips labels and verifier traces. Numerical M and K regimes are disjoint across the splits, while the source question template is reused, and the corpus is publicly reproducible.

The verifier accepts a single prediction:

```json
{"id":"...","verdict":"VIOLATES","repair":"16/11"}
```

It reports verdict accuracy, exact repair rate and fully completed matched pairs. Each test case is scored under **one verifier call** without teacher feedback. Training may request diagnostic oracle feedback, visibly tagged as synthetic.

```sh
python -m training.rlt_gym.bsar09.gym freeze --out /tmp/bsar09
python -m training.rlt_gym.bsar09.gym score --predictions /tmp/bsar09/model_answers.jsonl
python -m pytest tests/unit/test_rlt_gym_bsar09_repair.py -q
```

The frozen manifest identifies both source SHAs and hashes the exported files. The constant-positive baseline obtains 50% classification accuracy, zero exact repairs on invalid examples, and zero fully completed pairs.

**No Inky Mini, Bonsai, or DataGoblin checkpoint has been evaluated here.** Before claiming recurrent improvement, test the actual checkpoint with fixed root IDs, exact tokenizer counts, decoding settings, and an independent training-data contamination audit. The next formal-math milestone remains the PDE-derived dimensionless aggregate budget discussed in [MathCan BSAR-09 issue #186](https://github.com/patrickdugan/MathCan/issues/186).
