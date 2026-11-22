# MeTTa execution rollout custody

No rollouts are included in the seed. A future entry requires a reproducible execution receipt before it can be called verified.

Recommended JSONL record fields: `schema_version`, `rollout_id`, `source_id`, `upstream_blob_sha1`, `runtime` (name, git revision, flags), `task_family`, `lineage_root`, `split`, `input_query`, `actions`, `state_deltas`, `stdout_sha256`, `stderr_sha256`, `exit_code`, `assertion_results`, `verifier` (name, version, digest, verdict), `stop_reason`, `teacher_identity`, `generated_at`, and `tokenizer_id` with measured token count.

Stage output under an untracked run directory until an independent checker writes and signs a receipt. Keep development, calibration, held-out and sealed lineages separate; avoid using source file names or target symbols as retrieval shortcuts in test arms. Repair traces must preserve their unsuccessful attempts and visible feedback rather than synthesizing an idealized success path.
