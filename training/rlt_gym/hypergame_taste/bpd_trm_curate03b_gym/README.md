# BPD-TRM-GYM-01: Multi-Task TRM Gym Paid-Acquisition Evaluation & Inky Mini V3 Traces

**Evidence class:** `DEVELOPMENT_ONLY` (8 fresh independent roots `29201–29208` across 5 TRM gym tasks).
**Source:** `MoralityLabAI/BPD` PR #176 (`research/bpd-trm-adamw-screen-20261127`), pinned by commit SHA and canonical LF Git blob SHA-256 in `source_manifest.json`. Export files live in BPD and are not duplicated here.
**Decision:** EvalMath `REDESIGN_OR_PILOT_ONLY` is preserved; no training promotion.

## Export Scope (`bpd-trm-gym01-datagoblin-v3`)

- Strictly TRAIN-only (`split == "train"`, `training_eligible == True`, `evaluation_leakage == False`).
- **5 TRM gym tasks:** `E29_dyck1_bounded_depth`, `E31_2hop_pointer_chase`, `E28_dynamic_overwrite_register`, `E27_copy_retrieve_2slot`, `G05_3slot_copy_retrieve`.
- **60,286** total campaign windows observed across 6 paired arms (`A`–`F`), **13,329** scored decision rows exported (`C_analytic_full16`, `D_joint_early12`, `E_fast_analytic_k4`, `F_shuffled_k4`), 18 audited abstentions + 75 unaudited abstentions (`missing_unaudited`).
- `horizon_outcomes_gym_v3.jsonl` holds end-of-budget EVAL accuracies per `(task, root, arm)` separately and is never attached as a decision label.

## Root-Held-Out Paid-Acquisition Controller Results (Pooled Across All 5 Tasks, 1,579 `C_analytic_full16` Windows)

| Policy | Candidates Acquired | Retained Positive Gain | Est. CPU vs Always-16 | Meets 99.5% Bar |
|---|---:|---:|---:|---|
| `constant_ACQUIRE_16` | 25,264 | 100.00% | 1.000 | Yes |
| `full16_gate` | 25,264 | 100.00% | 0.995 | Yes |
| `early12_gate` | 24,788 | 99.82% | 0.979 | Yes |
| `early8_gate` | 23,984 | 99.62% | 0.951 | Yes |
| `adaptive_threshold` | 23,968 | 99.58% | 0.950 | Yes |
| `early4_gate` | 22,396 | 98.32% | 0.895 | No |
| `learned_logistic` | 20,860 | 96.53% | 0.852 | No |
| `learned_shuffled_labels` | 25,264 | 100.00% | 0.995 | Yes |
| `constant_ABSTAIN_at_4` | 6,316 | 0.00% | 0.293 | No |

Per-task breakdowns and fold parameters are recorded in `gym_policy_receipt.json`. Notably, pooling heterogeneous gym tasks without task conditioning causes the unconditioned `learned_logistic` model to fall below the 99.5% retention bar (`96.53%`), whereas simple zero-threshold early gates (`early8_gate`, `early12_gate`) and `adaptive_threshold` retain `>= 99.58%`.

## Inky Mini Multi-Task Gym Traces (`inky_gym_v3/`, deterministic `gzip -n`)

- `inky_gym_sft.jsonl.gz`: **13,329** multi-task state → action-sequence rows.
- `inky_gym_preferences.jsonl.gz`: **1,579** verifier-grounded `SELECT_BATCH` vs `ABSTAIN` preferences from matched-RNG TRAIN acceptance tests.
- `inky_gym_repair_traces.jsonl.gz`: **1,564** rollback → next-window repair traces.
- Uncompressed SHA-256 hashes pinned in `inky_gym_v3/inky_manifest.json`.

## Replay

```bash
python training/rlt_gym/hypergame_taste/bpd_trm_curate03b_gym/gym_policy_eval.py \
  <BPD>/mechinterp/bpd/experiments/trm_curation/results/curate03b/gym_campaign/datagoblin_export_v3 \
  --out training/rlt_gym/hypergame_taste/bpd_trm_curate03b_gym/gym_policy_receipt.json \
  --traces training/rlt_gym/hypergame_taste/bpd_trm_curate03b_gym/inky_gym_v3
```
