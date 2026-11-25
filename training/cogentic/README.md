# COG-02: a checked lemma ledger

COG-02 supplies an executable proof-search environment and exports supervision
for COG-03 in LoopedTransformers, COG-04 in BPD, COG-07 in
Compactification-Theory and COG-08 in EvalMath.

The corpus comes from generated finite Horn proof DAGs. Every world declares
propositions, initial facts, implication rules and a goal. A solver proposes a
rule index. The environment checks its premises against the currently proved
facts and records the result. Accepted intermediate lemmas remain available for
later steps. Each attempted action costs one verifier call, including rejected
actions and repeated proofs of an existing lemma.

The [Cogentic paper collection](https://github.com/google-research/google-research/tree/4a7bc8ab0bc7/cogentic_research_papers)
motivates the experiment. This corpus contains zero traces from those papers.
The experiment asks whether persistent verified intermediate results help
small learned controllers allocate later search. The authors' execution traces
are outside the inputs used here.

## Run

From the DataGoblin checkout:

```bash
PYTHONPATH=src python -m datagoblin.training_gym.rlt_gym.cogentic_ledger \
  --out /tmp/cog02-run --per-split 64 --seed 20261126
```

Add `--lean /absolute/path/to/lean` to compile every completed teacher proof and
check its axiom report. The driver also compiles a wrong-term sentinel and
requires rejection. It uses `import Init`, so Mathlib is unnecessary for these
propositional certificates. The selected Lean binary and all source files have
SHA-256 receipts in the run manifest.

The output path must be new. Run artifacts live outside the repository,
following the contributor guide's rule for generated runs.

## Shared API

The source lives at
`src/datagoblin/training_gym/rlt_gym/cogentic_ledger.py` and uses the Python
standard library. A downstream copy must record the upstream commit and source
SHA-256.

| Function | Contract |
| --- | --- |
| `generate_tasks(per_split=64, seed=20261126)` | Four public development splits with unique root worlds |
| `initial_state(task)` | Copy of the declared initial facts |
| `public_observation(task, known, history=None)` | Deep copy of public world, proved facts and prior feedback |
| `candidate_features(task, known, action)` | Twelve public local features in `FEATURE_NAMES` order |
| `check_step(task, known, action)` | One checked application, one verifier call |
| `reference_action(task, known)` | Symbolic teacher for training labels and a named structural control |
| `collect_trace(task)` | Teacher trace with an explicit rejection and duplicate sentinel |
| `replay_trace(task, trace)` | Reconstruct every state, rerun every check and validate costs |
| `ranking_row(task, known)` | Separate public candidates and environment-owned labels |
| `export_streams(task, trace)` | Train-only SFT, repairs, outcomes and transitions |

`reference_action` and `useful_atoms` belong to the teacher/control path.
Learned policies consume the public observation or candidate features. A
declared goal and implication graph form part of the task; future outcomes,
split labels and root identifiers stay outside the model input.

## Trace contract

A task has `schema_version`, `root_id`, `task_id`, `task_digest`, `split`,
`family`, `world` and `provenance`. The world contains integer atom IDs, initial
facts, rules with `premises` and `conclusion`, and one goal. Rule order and atom
names are permuted independently after the root world is created.

An episode records immutable steps containing the pre-action public
observation, proposed action, verifier receipt, costs and a hash chain. The
receipt fields are `accepted`, `code`, `next_known`, `new_fact`, `solved` and
`verifier_calls`. Fresh lemmas use their integer atom ID in `new_fact`.
Rejected or repeated actions use JSON null. A duplicate rule application has
`accepted: true`, code `duplicate_lemma`, zero new facts and one charged call.
Missing prerequisites produce `missing_premise`. Invalid indices produce
`invalid_action`.

The replay starts from declared facts. It recomputes every checker result,
cost and public observation, including previous feedback. Rehashing a forged
receipt cannot change the replayed result. COG-08 can implement an independent
checker directly from this contract.

## Data custody

Train, calibration and ordinary-test roots share the four generator families.
Each uses depth two through five. OOD roots use depths six through nine. Every
unpermuted generated world has one root hash, and generation forbids duplicate
root hashes across all splits. These splits test new generated instances and
greater depth within a known rule language. Their public seeds support exact
development replay. They are unsuitable for claims about sealed evaluation or
unseen mathematical fields.

The teacher adds a rejection and duplicate control to each collection episode.
This permits checking negative supervision and repeated-work accounting.
Successful repairs in this corpus arise from the declared symbolic teacher.
Learned repair must be measured in a separately executed policy rollout.

Only training roots enter the SFT and repair streams. SFT targets are newly
proved local lemmas, each backed by exact step replay. Failed attempts remain
in outcome and transition streams with explicit rejection labels. Whole-trace
Lean compilation certifies completed propositional derivations; the local
outcome labels describe the finite environment's operational premise check.
These two checks have separate receipt fields and separate costs.

## Experiments supported

COG-03 can train a recurrent controller and compare it with a separately
trained one-pass controller at an equal verifier-call budget. COG-04 can
quantize shared routing components while measuring decision accuracy and
closed-loop proofs completed. COG-07 can compress training constraints for an
explicit finite hypothesis family and certify that reconstruction preserves
its version space. COG-08 can count unique proved facts and completed roots,
with rejected calls and duplicate work retained in the denominator.

The current feature vector includes direct applicability cues and local goal
relationships. Ready-first and symbolic goal-ancestry controls therefore belong
in every performance report. A learned gain requires comparison with those
controls and a separately executed held-out rollout.
