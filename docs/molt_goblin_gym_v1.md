# MOLT-GOBLIN-01: research experiment array gym (development proposal)

Source: [Mixture-of-Loops Transformer](https://arxiv.org/abs/2611.34923).
Companion branches in LoopedTransformers, HybridTRMLDT, BPD and Compactification-Theory:
`research/molt-goblin-01-20261126`. Existing training-gym, RLT and LDT-array contracts remain unchanged.

## Train a researcher and an array curator

The producer proposes typed experiments. The LDT array sorter selects or abstains using immutable public receipts.
The executor is a separate operator-controlled worker. The steward independently scores receipts; the producer
cannot authorize spending, model downloads, hard deduction or promotion.

Stage 0 is executable offline: four deterministic task families, exact environment-owned answers, train-only
chat SFT export and separately written steward labels. Families: concurrent budgeted experiment arrays
(maximum verified utility under RAM/CPU/visit caps), interval Pareto status arrays, strict-prefix DAG
validation with stabilization depth, and exact resource/physical-dose matched sham selection.
The READY value is a structural eligibility label for the fixture, never statistical power or launch authorization.

```sh
python -m datagoblin.training_gym.molt_goblin generate --split train --seed 71 --per-family 128 --public runs/molt/train --steward runs/molt/train-labels
python -m datagoblin.training_gym.molt_goblin generate --split calibration --seed 91 --per-family 32 --public runs/molt/calibration --steward runs/molt/calibration-labels
python -m datagoblin.training_gym.molt_goblin score --public runs/molt/calibration --steward runs/molt/calibration-labels --predictions runs/molt/predictions.jsonl
```

Training SFT is a JSONL messages format; task/root IDs are preserved. Predictions are JSONL with task_id and answer.
A steward supplies fresh secret seeds for ordinary_test and ood_test; OOD uses larger arrays. Keep labels outside
builder access, use OS permissions, and never commit seed material or generated releases. Public algorithmic
labels are computable from the problem by design. Split namespaces alone do not establish distributional
independence; audit semantic duplicates and family holdouts before making generalization claims.

## Stage 1: model comparison, then curriculum feedback

Evaluate baby Goblin, Dory 4B, Nanbeige 3B, Qwen 4B, HRM Textual 1B and Inky Mini using local,
revision-pinned adapters where supported. Same serialized input, output/token and tool budgets; distinguish
native recurrent passes from emitted tokens. Keep inference and training costs separate. Add frozen-base,
random, exact-policy, typed LDT and TRM/learned-array comparators. Use the existing RLT rollout teacher
batch and adapter receipt interfaces for a later integration; this module is standalone.

Train on verified solver answers and exact task receipt records. For architecture research, extend the arrays
with candidate/source hash, hypothesis, falsifier, split/custody, predicted resource interval, independent
unit, matched control, causal dose, measured uncertainty and failure receipt. Human/operators approve run
manifests. Never feed holdout results or test seed material back into training. Negative and aborted runs
are useful train-only records with explicit status; do not manufacture hidden reasoning traces.

Proposed next research tasks: choose a TRM architecture; select U-MoLT versus residual loops; construct LDT
filter and Jinn/Beast membrane exercises from authorized public traces; design BPD intervention and controls;
choose 2/4/8/16 GiB calibration-bank budgets and estimate stale-cache risk; select the next LBF proof-search
run under verifier calls and time. These are stage 1 adapter work, not implemented task families today.

## Stage 2: feedback loop

Frozen checkpoint -> task proposals -> exact recipe and budget verification -> operator-approved capped
run -> independent score/uncertainty -> LDT array recommendation -> train-split trace export -> next checkpoint.
Require accuracy per family, exact selected set, unsafe-selection count, abstention, cost, held-out family
transfer, and root-level uncertainty. Report a Pareto frontier, keep exact-policy baselines, and measure
uplift per additional training token and compute unit. No automatic promotion. Consult EvalMath preflight
for causal comparisons; use independent roots and a separate locked confirmation release.

Stage 0 is a synthetic training-data and scoring interface, not an architecture result or a Verifiers v2
environment. Do not claim actual training, general research capability, or upstream MoLT reproduction.
