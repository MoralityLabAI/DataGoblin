# RLT rollout gym

The gym collects checked attempts across several task families and turns useful attempts into training data for a compact model. A local recurrent adapter exposes its state updates. Codex and Opus receive public task packets and can revise a proposal after the environment checks it. Both paths retain the failed attempts that explain a later repair.

The starting recipe keeps direct feedback and a constant mixture of task families. RLT and matched damping run as explicit controls. This follows the [routing decision](rlt_routing_decision.md), the [supervision experiments](rlt_trm_supervision.md), and the [Aleph protocol](../training/aleph_protocol.md).

## Run the CPU collection check

Python 3.10 or later and the normal development dependencies are sufficient:

```bash
python -m pip install -e ".[dev]"

python -m datagoblin rlt-gym make-suite \
  --seed 7 --per-family 4 --out runs/rlt-gym/suite

python -m datagoblin rlt-gym run \
  --suite runs/rlt-gym/suite \
  --seeds 7 19 --depths 1 2 4 8 \
  --modes direct rlt matched_damping \
  --splits train calibration \
  --max-wall-seconds 120 --out runs/rlt-gym/pilot

python -m datagoblin rlt-gym replay \
  --suite runs/rlt-gym/suite --run runs/rlt-gym/pilot \
  --out runs/rlt-gym/replay.json
```

Every output directory must be fresh. The default suite contains 140 tasks: seven families, five splits, and four instances for each family in each split. `--per-family 1` gives a smaller 35-task fixture. The run command defaults to the calibration split; the example includes training tasks so their failures can feed the teacher queue.

The supplied native adapter uses a frozen random recurrent reservoir and fits a ridge readout from independently checked training candidates. With its default width of 16, the fitted readout has 17 parameters. Its records carry `evidence_class: engineering_smoke` and `architecture_promotion_eligible: false`. These runs exercise collection, recurrence, comparison, and replay. Measurements of a fully trained RLT model belong to a subsequent adapter experiment.

The loop uses NumPy on CPU. The runner bounds the requested matrix, checks the wall budget between work units, and records process peak memory where the platform exposes it. `resources.json` describes these limits; an operating-system process cap belongs to the execution wrapper. Full checkpoint runs retain the [existing resource requirements](hrm_execution_safety.md).

## Task coverage

| Family | What the environment checks | OOD structure |
| --- | --- | --- |
| `graph_route` | A minimum-cost route through the currently active directed edges | Branches and cycles |
| `arithmetic_program` | The exact integer result of a bounded program | Conditional arithmetic |
| `constraint_assignment` | Every variable, fixed value, and coloring constraint | Higher-order constraint groups |
| `evidence_retrieval` | The answer and the exact live evidence supporting it | Alias resolution followed by a fact lookup |
| `table_transform` | Filtering, grouped sums, counts, and output order | Group multipliers joined into the calculation |
| `experiment_plan` | Required arms, resource budgets, split use, and independent checking | An intervention arm and its control |
| `finite_state_plan` | Each container action and the resulting goal state | Planning with an additional container |

These worlds are generated synthetic tasks. Each has a candidate pool for the compact ranker and a freeform answer grammar for teachers. The exact solver supplies a labeled reference control; candidate-order search measures the available pool. A teacher can propose an answer outside that pool, and the same environment checker evaluates it.

`make-suite` freezes task bytes, counts, lineage identifiers, and source digests. Instances sharing a lineage stay in one split. Loading also checks duplicate worlds across splits, while OOD generator identifiers remain separate from development. The suite uses `train`, `calibration`, `ordinary_test`, `ood_test`, and `sealed_poison` custody lanes. Freeze selection before opening the intended evidence lanes for a study.

## Compare fit and inference schedules

The local state follows

\[
x_{k+1}=x_k+\eta_k(h_{k+1}-x_k).
\]

Here, `h` is the block's proposed endpoint, `x` is the carried answer state, and `k` starts at zero. RLT sets `eta = 1 / (K - k)` for the fixed planned depth `K`. Direct feedback uses one throughout. Matched damping uses the mean of the nonterminal RLT coefficients, then replaces the state fully on the last visit. At depth one, each schedule uses one. The shared implementation is [`rlt_feedback.py`](../src/datagoblin/training_gym/rlt_feedback.py); the schedule follows the [RLT paper](https://arxiv.org/html/2611.16265v2).

A verified stop records the realized visit count while preserving the planned clock. The default readout is `carried_answer`. Use `--readout-site endpoint` for the endpoint readout and keep that choice fixed within a comparison. `--terminal-only` runs to the planned endpoint and records earlier checks, making losses of an earlier correct answer visible.

Cross the fitting schedule with every runtime schedule when investigating a schedule effect:

```bash
python -m datagoblin rlt-gym run \
  --suite runs/rlt-gym/suite \
  --training-modes direct rlt matched_damping --training-depth 4 \
  --modes direct rlt matched_damping --depths 1 2 4 8 \
  --seeds 7 19 --splits calibration --readout-site carried_answer \
  --max-wall-seconds 120 --out runs/rlt-gym/crossed
```

For this CPU adapter, the fitting schedule changes the states used to fit the readout. The recurrent weights remain frozen. Seeds share their initial block, public feature encoder, and clock inputs across fitting arms. Runtime comparisons pair the same task and model digest. The report groups results by task family and records regressions alongside verified coverage and checker calls.

## Send useful tasks to Codex or Opus

Start with a balanced training batch:

```bash
python -m datagoblin rlt-gym teacher-prepare \
  --suite runs/rlt-gym/suite --provider codex --model YOUR_CODEX_SESSION_ID \
  --split train --max-tasks 28 --max-visits 4 \
  --out runs/rlt-gym/codex-batch-01
```

`requests.jsonl` is the teacher-facing file. Each packet includes the public world, the answer grammar, and any visible history. The evaluator keeps `contexts.jsonl` and the batch manifest. Teacher packets omit candidate pools, generator identifiers, split metadata, seeds, and source paths.

Run the chosen teacher separately against those packets and save one JSON reply per request. The CLI prepares and checks files; the teacher runner owns provider calls and their budget. Use the actual model or session identity when preparing the batch. An Opus collection uses `--provider opus --model YOUR_OPUS_MODEL_ID` in its own batch directory.

A reply copies `request_id` and `task_digest`, then supplies `candidate` in the task's declared JSON shape. Optional fields are `public_steps`, `checks_requested`, `revision_summary`, `confidence`, and token `usage`. Public work artifacts remain attributed teacher claims. The importer obtains the answer verdict from the environment and records provider execution as unattested for an operator-supplied response file.

```bash
python -m datagoblin rlt-gym teacher-import \
  --suite runs/rlt-gym/suite \
  --requests runs/rlt-gym/codex-batch-01/requests.jsonl \
  --responses runs/rlt-gym/codex-responses-01.jsonl \
  --out runs/rlt-gym/codex-round-01

python -m datagoblin rlt-gym teacher-continue \
  --suite runs/rlt-gym/suite \
  --rollouts runs/rlt-gym/codex-round-01/rollouts.jsonl \
  --out runs/rlt-gym/codex-batch-02
```

Continuation carries the previous visible attempt and checked feedback into the next request. A verified answer stops the episode. Recognizable malformed replies consume a visit and remain parse failures; missing replies retain their own outcome. Import artifacts preserve the original response bytes and their digest, including rejected lines. Token counters distinguish reported usage from unknown usage.

### Choose a frontier from training failures

The frontier command checks source rollouts, selects training tasks with failures, and balances families. Within a family, schedule disagreement receives priority, followed by failure rate.

```bash
python -m datagoblin rlt-gym frontier \
  --suite runs/rlt-gym/suite --rollouts runs/rlt-gym/pilot/rollouts.jsonl \
  --max-tasks 28 --out runs/rlt-gym/frontier-01

python -m datagoblin rlt-gym teacher-prepare \
  --suite runs/rlt-gym/suite --frontier runs/rlt-gym/frontier-01/manifest.json \
  --provider codex --model YOUR_CODEX_SESSION_ID --max-tasks 28 --max-visits 4 \
  --out runs/rlt-gym/frontier-batch-01
```

This form asks for a fresh solution to the selected task. To give the teacher an observed native failure as repair context, use the repair bridge:

```bash
python -m datagoblin rlt-gym teacher-repair \
  --suite runs/rlt-gym/suite --rollouts runs/rlt-gym/pilot/rollouts.jsonl \
  --provider codex --model YOUR_CODEX_SESSION_ID --max-tasks 28 --max-visits 4 \
  --out runs/rlt-gym/repair-batch-01
```

The repair batch includes one independently checked native failure per selected training task. That context occupies the first visit, leaving three teacher replies when `--max-visits 4`. Each later attempt records its producer. The source episode, original visit, and candidate receipt remain bound into the repair lineage. Import and continue these packets with the same commands used for ordinary teacher batches.

## Export data for the compact learner

```bash
python -m datagoblin rlt-gym export \
  --suite runs/rlt-gym/suite \
  --rollouts runs/rlt-gym/codex-round-01/rollouts.jsonl \
  --out runs/rlt-gym/dataset-01
```

Export rechecks the candidate receipts and admits training-split examples. The output manifest records excluded splits and artifact hashes.

| Stream | Intended supervision |
| --- | --- |
| `sft.jsonl` | Independently verified answers with their public input and prior history |
| `transitions.jsonl` | Observable attempts, verifier results, and attributed work artifacts |
| `outcomes.jsonl` | Outcome prediction before the final verifier result, with incomplete attempts marked as censored |
| `repairs.jsonl` | A verified revision following a failed attempt |
| `preferences.jsonl` | Independently checked pass/fail pairs sharing the same task input and visible history |

Preference records keep both producers and their source receipts. Sequential repair steps have different histories, so their repair evidence travels in `repairs.jsonl`. The control examples from the native smoke run require `--include-controls` and retain their control labels. A native failure supplied to a teacher remains visible in the repair context even when its target is excluded from training.

Keep the constant family mixture when assembling the first compact-model batch. Use the frontier to allocate additional attempts within each family, then retain the broad coverage batch as a stable comparison. Count examples by source lineage and verifier outcome before choosing training weights; repeated successful variants of one root can otherwise dominate the update.

## Carry the observations into BPD

RLT supplies a scalar control over a proposed state change. A BPD adapter can later attach component-specific interventions to the same visit and observation point. The rollout contract already records the planned segment, placement, readout site, state and endpoint digests, update norms, and intervention metadata.

The supplied native adapter accepts a bounded pulse at the carried answer boundary. A random pulse configuration can be stored under the run directory:

```json
{
  "kind": "random_pulse",
  "boundary": "carried_answer",
  "visit": 1,
  "relative_dose": 0.1,
  "seed": 101
}
```

Pass its path through `run --intervention PATH --depths 4`. The visit must fit every selected depth. `direction_pulse` accepts an explicit finite direction; the trace records the requested and realized dose. Compare a pulse with an untreated run from the same checkpoint, and use random directions matched for dose when testing a chosen direction. Record effects at the next named readout and at the terminal result.

`bpd_mechanism_verified` stays false in these records. Mechanism claims require the BPD study's donor, sham, and held-out effect tests. Codex and Opus episodes carry `dynamics_kind: black_box_refinement`; their traces describe visible proposals and revisions. Local adapters can expose measured recurrent-state summaries through `RecurrentAdapter.visits`.

## Extend the next collection batches

### Collect Lean selections with OxiBonsai

Tetsuya Kitahata's [OxiBonsai](https://github.com/cool-japan/oxibonsai) serves
PrismML's [1-bit Bonsai-8B](https://huggingface.co/prism-ml/Bonsai-8B-gguf).
The model chooses a tactic from the grounded Lean gym's library and can try again after a rejection.
The lexical baseline and Bonsai see the same candidate pool and have the same attempt budget.

Build OxiBonsai with CPU support and the server feature, then start one model replica on loopback.
Use the published `Bonsai-8B.gguf` and the Qwen3-8B `tokenizer.json` documented by OxiBonsai.
The checkpoint hash check also accepts a local copy with a different filename.

```bash
cargo install oxibonsai-cli --version 0.2.3 --features server --locked
oxibonsai serve --model /path/to/Bonsai-8B.gguf --tokenizer /path/to/tokenizer.json \
  --host 127.0.0.1 --port 8080 --pool-size 1 --max-concurrent-requests 1
```

Run the collector in another terminal. Pass the actual Lean toolchain executable.
The example collects four train goals using at most four choices per goal and arm.

```bash
python -m datagoblin rlt-gym lean-bonsai \
  --checkpoint /path/to/Bonsai-8B.gguf --lean /path/to/toolchain/bin/lean \
  --max-goals 4 --attempts 4 --out runs/lean-bonsai/train-01
```

Supply `--fixtures /path/to/fixtures.json` to use the existing Lean gym's frozen cohorts.
Otherwise the collector generates synthetic goals from the existing templates and `--seed`.
Set `--split held_templates` for a held-template run. Every winning tactic elaborates against the exact
statement under Lean, with an axiom audit. Each rollout carries the tactic, verdict, timing and token usage
when the server reports it. `sft.jsonl` contains only the model's verified train proofs, with the goal,
proof script and receipt hashes. Held-out runs leave this file empty. `plan.json` records the local
checkpoint digest and the server's model ID. The operator binds the running server to the checkpoint file.

The collector measures black-box tactic selection and records each attempt for later training review.

### Extend task coverage

The next collection should use the checked synthetic tasks to establish throughput and the frequency of useful repairs. Follow with source-pinned tasks from the existing DataGoblin work. Lean can reuse the grounded ranker and composition paths in [`rlt_lean_train_rank.py`](../src/datagoblin/training_gym/rlt_lean_train_rank.py), [`rlt_lean_rank_eval.py`](../src/datagoblin/training_gym/rlt_lean_rank_eval.py), and [`rlt_lean_composition.py`](../src/datagoblin/training_gym/rlt_lean_composition.py). Preserve the theorem-root split and bind every accepted proof to its kernel receipt. Measure the full candidate-pool ceiling so proof generation receives attention when the ranker has exhausted that pool.

The route-certificate work in [`rlt_route_certificate.py`](../src/datagoblin/training_gym/rlt_route_certificate.py) supplies revision and reuse cases for rule graphs. Exact traversal should remain an available solver there. The existing [supervision study](rlt_trm_supervision.md) supplies a place to compare exact intermediate labels with teacher-proposed revisions.

For a new family, add its public input contract and bounded checker before collecting teacher attempts. Pin the generator and external source revision, assign lineage custody, and give OOD its own structural change. Reserve fresh roots for the compact learner's evaluation. The [training collection notes](../training/rlt_gym/README.md) describe how these batches feed the Aleph program.

## Artifacts and CI

The native run writes `plan.json`, `rollouts.jsonl`, `summary.json`, `REPORT.md`, `measurements.jsonl`, `resources.json`, model snapshots, and a completion manifest. Episode records are flushed as collection proceeds. An interrupted fit can leave partial evidence without a completion manifest. Replay checks source fingerprints, the frozen suite, model snapshots, matrix completeness, and reproduced trajectories. Timing measurements live outside the canonical rollout stream.

Generated suites, snapshots, and rollouts belong under ignored `runs/`. Source code, documentation, and test fixtures belong in the repository. The [CI workflow](../.github/workflows/rlt-rollout-gym.yml) exercises public synthetic fixtures on Python 3.10 and 3.12, replays the run, and stages seven teacher requests. Hosted run status comes from the workflow result; private-repository billing can prevent a job from starting. CI artifacts expire after seven days.
