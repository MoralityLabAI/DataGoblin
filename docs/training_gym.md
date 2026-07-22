# Architecture training gym

The training gym compares parameter-efficient and recurrent candidates under a frozen evaluation contract. Version 1
is an offline deterministic scaffold. It does not train a checkpoint, rank real architectures, or establish alignment.

The preregistration is `configs/preregistrations/training_gym_v1.yaml`. Its primary question is a control/design
question: which candidate occupies a useful accuracy, calibration, robustness, termination, and resource-cost Pareto
point under matched budgets? There is deliberately no universal leaderboard scalar.

Version 1 currently executes only stage 0: constitutional verdict, violation, calibration, abstention, and loop-trace
evaluation. Architecture promotion remains disabled until four additional task families are executable: minimal
constitutional repair, counterfactual mechanism discrimination, experiment-outcome/resource forecasting, and
long-horizon state plus halt calibration. This prevents selection of a model that is merely good at one
classifier-shaped benchmark.

## Candidate families

- `lora_sft`: rank and target-module sweeps with a frozen-base and randomly initialized adapter control.
- `hrm_text_lora`: bounded recurrent-step and LoRA sweeps with tied single-state and two-timescale recurrent controls.
- `looping_transformer_jack_in_box`: a hypothesis consisting of a shared transformer block, persistent state token,
  learned halt signal, hard maximum loop count, and ponder cost. Required controls are one pass, matched untied depth,
  shuffled loop state, and fixed loop count.

The nickname is not a mechanistic claim. A looping candidate must beat matched controls on held-out evidence and declared
cost; visually interesting recurrence or state motion is diagnostic only.

## Evaluation firewall

| Layer | Splits | Use |
|---|---|---|
| Training | `train` | Parameter updates only |
| Selection | `calibration` | Hyperparameter and checkpoint selection |
| Evidence | `ordinary_test` | Frozen primary comparison |
| Robustness | `ood_test` | Unseen environment family |
| Sealed hazard | `sealed_poison` | Report-only, separately stewarded labels |

Primary evidence includes exact decision and exact decision-plus-violation accuracy, violation micro F1, Brier score,
expected calibration error, and area under the selective risk-coverage curve. Hazard metrics include critical false
acceptance, schema failure, abstention, nontermination, decision flips, cap breaches, and cleanup failure. Candidate
manifests that declare an evidence split as training data fail closed.

The evaluator never writes sealed authoritative labels into `scored_rows.jsonl`. It emits only sealed aggregates after a
separate steward label file is merged. Organizational custody still requires an external path and access policy; the
single-process mock demo is plumbing evidence only.

## Offline smoke demo

```bash
python -m datagoblin gym-demo --seed 7 --out runs/training-gym-demo
```

This produces a constant-accept negative control and a small structural two-loop control, then evaluates both. No model
weights, Torch runtime, GPU, network, training step, or checkpoint is involved. Every mock remains promotion-ineligible.

Evaluate an externally produced candidate envelope with:

```bash
python -m datagoblin gym-evaluate \
  --contract configs/preregistrations/training_gym_v1.yaml \
  --dataset-manifest runs/training-gym-demo/public_dataset/dataset_manifest.json \
  --candidate-manifest path/to/candidate_manifest.json \
  --predictions path/to/predictions.jsonl \
  --sealed-labels path/to/private/sealed_labels.jsonl \
  --out runs/candidate-evaluation
```

Compare two or more reports against a declared frozen-base candidate with `datagoblin gym-compare`. The comparison
emits matched deltas, promotion gates, and Pareto-dominance flags; it intentionally leaves `winner` unset.

## Training safety gate

The scaffold fixes task ID `architecture-training-gym-v1-smoke`, one-microbatch chunks, checkpoints every 100 steps or
300 seconds (whichever comes first), and hard caps of 2048 MiB RAM, 50% CPU, 50 MB/s I/O, 900 seconds, and zero swap.
Those limits are for mock and tiny adapter development and are not a claim that a real 1B checkpoint will fit.

Real training remains disabled until the operator supplies a checkpoint and candidate recipe, approves any cap change,
and uses `scripts/run_training_gym_capped_wsl.sh` or an equivalent candidate-specific cgroup-v2 wrapper. The wrapper
fixes the declared caps, disables swap, creates a private network namespace, contains every child in one owned systemd
unit, and passes the candidate command as an argument vector rather than shell text. Every real run must emit the event and summary schemas,
checkpoint on schedule, clean up only recorded owned processes, release CUDA objects explicitly, and run
`scripts/post_run_hrm_cleanup.ps1`. An abort or cleanup failure is a retained result, never silently retried as success.
Invoke the WSL launcher through `bash` and place the checkpoint, dataset, working directory, and output on the declared
IO-capped block device. Its initial summary deliberately leaves weight-load state unresolved and cleanup failed/pending;
only the candidate's explicit object/CUDA cleanup receipt plus the Windows post-run audit can complete those fields.
