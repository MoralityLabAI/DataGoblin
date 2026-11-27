# Lane: self_adapter (adapter on the driver model itself)

**Climbs** bench_accuracy on the dev bench: how often the model driving this app applies the hill-climb
rules exactly (gate verdicts, recipe-call legality, next eligible lane). Gate: up, min_delta 0.03,
patience 3, max 12 moves. The eval contract is the exact list of dev tasks evaluated, so every move
must score the same `bench_make` dev run with the same `limit`.

## Baseline move

1. `bench_make` with `split: "dev"`, `seed: 0`, `count: 120`. Keep its run id: it is the dev set for
   every later move in this lane.
2. `driver_bench` with `tasks: "run:<dev bench run>"`, `base_url` and `model` of the driver endpoint
   from `suite_status.endpoints` (the base model, no adapter), `limit: 120`. Needs tier loopback.
3. `record_result` on the driver_bench run, then `decide` (baseline).

`bench_score.json` in the driver_bench output has `by_type`: the weakest task type is where training
data should go first.

## Data moves (train split only)

- `bench_make` with `split: "train"`, a new `seed`, `count` 300-2000. Train and dev never share a task.
- Optional teacher pass: `driver_bench` on the train tasks against a bigger model endpoint. Its correct
  answers become `verified_teacher` rows.
- `export_driver_sft` with `tasks: "run:<train bench run>"`, optionally `answers: "run:<teacher
  driver_bench run>"` and `receipts: "runs/hillclimb/receipts.jsonl"` (this app's own transcripts;
  only runs whose tool calls all succeeded and that reached a decision are kept).
- `curate` with `inputs: ["run:<export run>/sft.jsonl"]`.

## Training and the evaluation move

Adapter training runs on the GPU, tier gpu (`data-goblin --approve all`):

- `plans/driver_adapter.json` is this lane's plan. Tag `train-driver` trains `b25_driver_fp_r8`, a
  rank-8 LoRA on Ternary Bonsai 2.5 27B, from the curated `admitted.jsonl` its bonsai-gym config names
  (the operator's curated set, not a `run:` output of this ledger). Tag `eval-driver` serves the base
  model, then the base with the adapter, and runs this lane's evaluation move through this app for each.
  Training on other curated rows is a plan change: report it for the operator.
- `adaptergym_status` / `adaptergym_preflight` with that plan show which steps are done and whether the
  card is free; `adaptergym_launch` starts steps detached through vramorama leases. You never edit plans.
- Inkling 2 Mini adapters train through Tinker (`Agonist/TINKER_GYM.md`), which needs the operator's
  credentials: report the command, do not run it. `tinker_fixture` is the offline check of that path.

When the adapter is served (the operator or the plan starts the server), propose the evaluation move:
`driver_bench` on the baseline's dev run, same `limit`, `base_url`/`model` of the adapter endpoint,
then `record_result` and `decide`.

## Hypotheses that make good moves

- More exact-oracle rows of the weakest task type.
- Adding verified teacher rows or clean transcripts to the exact rows.
- A different adapter rank or learning rate (a plan change: report it for the operator).
