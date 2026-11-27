# Lane: curation (training data for looped models and adapters)

**Climbs** admitted_roots: unique train roots admitted by `curate`. Gate: up, min_delta 1, patience 3,
max 20 moves. Curation decides what is safe to train on; whether the data helps is decided in the lanes
that train on it.

## The curate rules, in order

1. A row needs a root (`root`, `root_id`, `lineage_id`, `task_id` or `request_id`) and a `split`.
2. Only `train` rows are admitted.
3. A root that appears under more than one split anywhere in the inputs is dropped entirely: that is
   leakage (`leakage_roots` in the manifest).
4. Exact duplicate content is dropped.
5. A root keeps at most `max_per_root` presentations (default 4, DataGoblin's cap per root).

Admitted rows keep `training_ready: false`; rows needing rights review are counted in
`rights_review_pending`. Promotion to training is a separate decision for the operator.

## A move

`curate` with `inputs` (up to 16 JSONL paths, usually `run:<run_id>/<file>`) and `max_per_root`, then
`record_result` and `decide`.

Good inputs:

- `train_candidates.jsonl` from an `agonist_audit` run (agonist_evals);
- `aleph.jsonl`, `sft.jsonl` or `repairs.jsonl` from a `rollout_score` run (agonist_rollouts);
- `sft.jsonl` from an `export_driver_sft` run (self_adapter).

Adding a new verified source is the usual promoting move. Mixing sources that share roots across splits
lowers the admitted count, which tells you the sources disagree about custody: report which.
