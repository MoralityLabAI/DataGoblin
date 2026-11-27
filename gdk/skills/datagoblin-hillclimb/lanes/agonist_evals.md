# Lane: agonist_evals (bigger local models build new evals)

DataGoblin is the agonist: it turns research taste into AGN-01 teacher requests, a bigger local model
builds paired counterfactual worlds, and the independent world verifier decides which pairs count.

**Climbs** verified_pair_yield: accepted pairs divided by all replies the audit saw (unparsed or invalid
replies count as rejected). Gate: up, min_delta 0.05, patience 3, max 12 moves.

## A move

1. `agonist_stage` with `tastes` (default `Agonist/fixtures/tastes.jsonl`) and `max_context_tokens`
   (12,288 by default, at most 16,384).
2. `agonist_send` (tier loopback) with `requests: "run:<stage run>/teacher_requests.jsonl"`, the
   teacher's `base_url` and `model` from `suite_status.endpoints`, `max_requests`, `max_tokens`
   and `temperature`. The worker asks the server's `/tokenize` for each prompt's size and skips
   requests over the context cap; `provider_receipts.jsonl` records every request.
3. `agonist_audit` with the same `requests` and `replies: "run:<send run>/teacher_replies.jsonl"`.
4. `record_result` on the audit run, then `decide`.

`agonist_demo` runs the whole path on synthetic fixture replies with no model; use it to check the
wiring. The service refuses fixture worlds as results.

## Move menu

- The teacher: a bigger local model listed in `suite_status.endpoints`.
- `max_tokens`: thinking models need room for reasoning before the JSON object.
- `temperature`: lower for schema discipline, higher for variety.
- New taste seeds: the operator writes them as JSONL following `Agonist/SKILL.md` (taste_id, family_id,
  idea, source_kind, source_ref, rights, split). Families stay in one split.

## What the outputs are for

Accepted dev-split pairs (`public_worlds.jsonl` and `verifier_receipts.jsonl` in the audit output) are
new eval worlds. Train-split `train_candidates.jsonl` rows go to the curation lane. Every candidate keeps
`training_ready: false`; rights review and promotion are separate.
