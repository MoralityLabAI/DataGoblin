# Lane: agonist_rollouts (bigger local models produce rollouts)

A bigger local teacher answers staged RLT tasks once each; the RLT environment verifier rechecks every
answer, and verified answers become distillation rows for looped students and adapters.

**Climbs** rollout_accuracy: verified correct answers divided by staged tasks, on the train split.
Gate: up, min_delta 0.05, patience 3, max 12 moves. The eval contract is the suite digest: make the suite
once, and keep `split` and `max_tasks` fixed across moves so accuracies compare.

## A move

1. `rlt_make_suite` once (reuse its run id afterwards).
2. `rollout_stage` with `suite: "run:<suite run>"`, `split: "train"`, `max_tasks`, and `model` set to the
   teacher's model id exactly as `suite_status.endpoints` lists it: capture checks that the server
   returns the same id.
3. `rollout_capture` (tier loopback) with `stage: "run:<stage run>"`, the teacher's `base_url` and
   `model`, `max_requests` (at least `max_tasks`), `max_tokens`, `timeout_seconds`.
4. `rollout_score` with `stage`, `captures: "run:<capture run>"` and `suite`.
5. `record_result` on the score run, then `decide`.

## Move menu

- The teacher model (a bigger endpoint from `suite_status.endpoints`).
- `max_tokens`: answers cut off at the budget are censored, not wrong, but they lower accuracy.
- Changing `split` or `max_tasks` breaks comparability with the incumbent: leave both fixed.

## What the outputs are for

`rollout_score` writes `sft.jsonl`, `repairs.jsonl`, `preferences.jsonl` and `aleph.jsonl`. Train rows
go to the curation lane (`aleph.jsonl` rows carry root and split). Calibration-split stages are for
measuring a teacher, never for training data.
