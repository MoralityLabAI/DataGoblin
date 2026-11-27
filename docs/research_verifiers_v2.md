# Research Verifiers v2

`datagoblin-research-v2` evaluates a checkpoint as the proposer in bounded research
episodes. Its first checkpoint entry is Baby Goblin after the direct-A replacement-v2
10M pilot. The comparison registry also contains Dory, Nanbeige, Qwen3, HRM-Text and
the open variable-loop counterpart to Inkling 2 Mini.

The environment executes the proposed experiments and computes the outcome. It
retains failed attempts and invalid submissions in the denominator. Every checkpoint
receives the same task instances and experiment allowances. Reports keep exact task
scores beside learning quality, critical false accepts, format failures and measured
work.

The environment contract is version 2. Its optional Prime Intellect adapter pins
`verifiers==0.4.1` and uses the upstream `verifiers.v1` API. See the
[adapter instructions](../environments/datagoblin_research_v2/README.md) for `vf-validate`
and `vf-run`. The core completion runner implements the common 2,048-token comparison.

## Research tasks

The eight families receive equal weight. A small public development suite contains
one case per family; the default suite builder creates four per family.

| Family | What the proposer does | What the environment verifies |
|---|---|---|
| `trm_training` | Fit a tiny tied reasoning/answer learner on original and revised graphs; choose update count, width and recursion depth. | Real weight changes, independently traversed graph targets, calibration claims, unseen graph roots and longer paths. |
| `ldt_filter` | Fit an additive three-class head over explicit evidence and quality tags. | Separate raw and guarded accuracy; critical tags retain their exact veto under positive-feature pressure. |
| `persona_externalization` | Fit an external persona head and reconstruct a scoped preference store from revision events. | Owner and character separation, revocation, future-version exclusion, scope matching and the exact critical veto. |
| `experiment_design` | Specify arms, replicates, token budgets, split roles and metrics for a controlled experiment. | The existing native RLT gym experiment-plan verifier checks every constraint. |
| `boolean_repair` | Establish Boolean proposal equivalence or supply a concrete counterexample. | Exhaustive finite semantics checks the claim or witness. Any valid counterexample is accepted. |
| `belief_update` | Update a small hypothesis set after observations. | Exact rational Bayes updates, reduced fractions and normalization. |
| `evidence_custody` | Admit eligible training records and quarantine the remainder. | Authority, receipt binding, rights status and root custody. A test row holds every training-labelled descendant of its root. |
| `cursor_recovery` | Select a usable checkpoint and state the remaining position count. | Weights, optimizer, RNG and data cursor agree at one durable group with the expected recipe and corpus. |

The TRM-style learner is a small graph-conditioned experiment with explicit NumPy
gradients. Its shared cell controls reasoning transport and answer carry. The LDT
experiments fit 75-parameter additive heads over 24 tags. These experiments measure
the proposing language model's ability to use a bounded research interface. They do
not instantiate or train the comparison language models themselves.

### Relation to Baby Goblin's training

The task selection follows the repository's
[native RLT/TRM training code](../src/datagoblin/training_gym/rlt_trm.py),
[research gym](../src/datagoblin/training_gym/rlt_gym/tasks.py),
[evidence receipts](../src/datagoblin/data_etiquette/evidence_receipts.py) and
[Model Atlas](../training/model_atlas/README.md).

The replacement corpus's research lane contains solver tariffs, minimax probe
purchases, counterfactual sensor choices, Boolean witness costs and experiment-plan
audits. Its other verified material includes exact belief updates and formal proof
sources. The source builder and recipe are pinned on the recovery branch at
[`b078ab75274a2f95449b1daf99b80529049663e7`](https://github.com/MoralityLabAI/DataGoblin/tree/b078ab75274a2f95449b1daf99b80529049663e7/training/aleph_next).
The [10M training receipt](../pre-trains/aleph-direct-a-10m-v2/README.md) identifies
the trained export and the replacement index.

This suite creates new instances. Several skill families overlap training by
design. Exact-input split hashes and new instances do not establish unseen task
templates, semantic novelty, or absence from other models' pretraining. The initial
report therefore measures performance on this research interface. It does not make
a broad claim about scientific competence or parameter efficiency.

## Actions and scores

Every turn supplies one JSON object. Examples:

```json
{"action":"run_experiment","config":{"steps":0,"recursions":8}}
{"action":"run_experiment","config":{"steps":64,"recursions":8}}
{"action":"submit","answer":{"artifact_id":"a2","baseline_id":"a1","claim":"improves"}}
```

The two experiment calls above apply to `trm_training`. Artifact IDs are assigned
by the environment and resolve only inside that episode. The second fit must use
the first fit's initialization and the same configuration except for its update
count. An improvement claim concerns the visible calibration accuracy. Each
training call creates a content-bound weight artifact and receipt in the evaluator.
Submitted receipts, scores and arbitrary code have no execution path.

The five symbolic families accept `check` at most twice and a final `submit`.
Checks return one pass/fail bit. Learner families expose calibration results;
their evaluation and longer-path/adversarial tests run only after submission.
Final test metrics stay out of model observations.

For symbolic tasks the score is exact correctness, 0 or 1. Learner quality is the
mean of ordinary evaluation accuracy and OOD accuracy. That quality becomes the
primary score when the submitted calibration claim is correct, the fitted weights
changed, the initialization control matches and the required policy checks pass.
Persona tasks also require the correct external memory. The learner task's success
flag requires mean quality at least 0.75 and positive improvement over initialization.

The report preserves continuous quality and the control delta even when an integrity
condition fails. Raw LDT scores remain visible beside the effect of the structural
veto. This keeps a useful diagnostic for a learned head that relies on its veto.

## Fixed budgets

| Resource | Limit |
|---|---:|
| Turns per episode | 8 |
| Experiments per episode | 4 |
| Symbolic check calls | 2 |
| Optimizer updates per fit / episode | 128 / 256 |
| Training examples per fit | 256 |
| TRM width / recursions | 32 / 8 |
| Training cell calls per episode | 1,600,000 |
| Worker CPU / wall time | 10 / 15 seconds |
| Worker address space | 512 MiB |
| Episode wall time | 120 seconds |
| Model context / generated tokens per turn | 2,048 / 512 |
| Attempted actions per checkpoint run | at most 64 |
| Run wall time | at most 900 seconds, further bounded by the supplied cap receipt |

The worker runs fixed allowlisted operations in a separate Linux process. Input is
bounded JSON. A failed fit consumes its reserved experiment and update budget.
The remaining episode and run deadlines bound each worker, including final grading.

The endpoint runner reserves two HTTP slots per action: one prompt tokenization
request and one inference request. Both share a request deadline. It checks the
rendered prompt token count before inference, then checks the returned token usage
and served model ID. The effective action, HTTP and wall caps enter the comparison
hash. A smaller cap produces a different comparison contract.

Generation uses temperature zero. The model receives its original packet and the
complete current public observation each turn. Earlier raw turns stay in the saved
transcript. The current observation includes every fitted artifact's configuration
and calibration summary, so the prompt preserves the state needed for the next
decision while respecting Baby Goblin's context length. This is a fixed external
state protocol, and its results should be labelled accordingly.

## Checkpoint identities

The complete pins, configuration hashes and tokenizer-configuration hashes are in
[`configs/research_v2/models.json`](../configs/research_v2/models.json).

| Registry ID | Checkpoint | Parameter accounting / recurrence |
|---|---|---|
| `baby-goblin-10m-v2-native` | Direct A native FP32 export | 999,352,320 tied parameters; ten core blocks visited twice. |
| `dory-4b-bf16-loop2` | `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` | 3,975,229,440 total; two recurrent loops. |
| `nanbeige42-3b-loop2` | `Nanbeige/Nanbeige4.2-3B` | 4,169,800,704 total; the published 3B label excludes embeddings; two loops. |
| `qwen3-4b` | `Qwen/Qwen3-4B` | 4,022,468,096 total. |
| `hrm-text-1b-h2-l3` | `sapientinc/HRM-Text-1B` | 1,182,795,264 total; two H cycles, three L cycles. |
| `inky-mini-varloop-research-pass3` | `thinkingmachines/TML-d24-varloop-research` | 26,388,646,440 total; about 5.4B effective active parameters at three passes. |

The TML release is the variable-loop research counterpart to Inkling 2 Mini. Its
later training differs from the fixed-loop Mini. The publisher explains the
distinction in its [model card](https://thinkingmachines.ai/model-card/tml-d24-varloop-research/).

Baby Goblin's native weights have SHA-256
`2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8`.
A GGUF, expanded BF16 export, changed recurrence or alternate thinking mode gets
its own registered identity. The stored parameter count of an expanded export is
recorded separately from the native model's tied parameter count.

[HRM's model card](https://huggingface.co/sapientinc/HRM-Text-1B) requires a condition
envelope and PrefixLM attention over the full prompt. Its serving adapter must pass
ones for every prefill token's `token_type_ids`. Baby Goblin uses raw completion
framing. The transport keeps these protocols explicit and requires an operator's
attestation for the serving implementation.

An endpoint's response model ID establishes only its declared identity. The
operator binds the weight digest, revision, precision, recurrence and template to
the serving process and supplies a hard resource-cap receipt. These bindings are
labelled `operator_attested` in reports. The runner never allocates a runtime or
downloads a checkpoint.

## Run the environment

From the repository checkout, using Linux and Python 3.11 or later for the Verifiers
adapter:

```bash
python -m pip install -e '.[dev]' -e environments/datagoblin_research_v2
python -m datagoblin.research_v2 build --split development --seed 20261126 \
  --per-family 4 --out runs/research-v2-dev
python -m datagoblin.research_v2 validate --suite runs/research-v2-dev
python -m datagoblin.research_v2 control --suite runs/research-v2-dev \
  --policy reference --max-episodes 8 --out runs/research-v2-reference
python -m datagoblin.research_v2 control --suite runs/research-v2-dev \
  --policy noop --max-episodes 8 --out runs/research-v2-noop
vf-validate datagoblin-research-v2 --runtime.type subprocess -n 8
```

The reference control trains the tiny learners and solves the symbolic worlds.
Its score is an environment check. The no-op control scores zero. Neither result
belongs in a checkpoint leaderboard.

Create an evaluation panel in an externally held directory after freezing the code.
Omitting `--seed` draws a fresh seed. Register any existing held roots through
`--exclude-registry` before writing the panel:

```bash
python -m datagoblin.research_v2 build --split heldout --per-family 4 \
  --exclude-registry /private/existing-held-roots.json \
  --out /private/research-v2-confirmation
```

The builder writes a public prompt file, an evaluator-only case file, a held-root
registry and a hashed manifest. It refuses held suites inside a Git checkout.
Private filesystem permissions support local custody; steward access policy must
also keep case files and Verifiers traces away from the model's tools. Public
development cases are permanently excluded from any claim of sealed confirmation.
Later checkpoints can reuse the panel for a declared comparison series. Model
selection based on those results consumes its role as an untouched confirmation
panel, so final confirmation needs a separately held panel.

Bind an already running endpoint using the
[endpoint schema and example](../environments/datagoblin_research_v2/README.md).
The tokenizer route must count the exact rendered input that the inference route
will use. A server lacking that route needs a local normalization adapter.

```bash
python -m datagoblin.research_v2 run --suite /private/research-v2-confirmation \
  --model-id baby-goblin-10m-v2-native --binding /private/baby-binding.json \
  --max-episodes 32 --out runs/baby-10m-research-v2
python -m datagoblin.research_v2 run --suite /private/research-v2-confirmation \
  --model-id qwen3-4b --binding /private/qwen-binding.json \
  --max-episodes 32 --out runs/qwen-research-v2
python -m datagoblin.research_v2 compare \
  runs/baby-10m-research-v2 runs/qwen-research-v2 --out runs/research-v2-comparison.json
```

Each run writes `results.jsonl`, a transcript and `report.json`. The evaluator also
stores its tiny fitted weight artifacts, and the report binds their bytes for replay.
An exhausted budget, malformed action or endpoint failure leaves a zero-scored row for every affected
selected case. A failure prevents ranking eligibility while preserving diagnostic
scores. `compare` requires identical ordered cases, roots, suite, implementation,
effective budget, experiment runtime and comparison role. It reports per-family deltas and a paired,
within-family root bootstrap interval. Eight-case smoke intervals are descriptive.
The comparison leaves the winner field unset.

### Future checkpoints

Write a new entry containing the fields listed in the model registry's
`future_checkpoint_contract`, plus parameter count and recurrence. Register it into
a new registry file:

```bash
python -m datagoblin.research_v2 register --entry /private/baby-next-entry.json \
  --out runs/models-with-next-checkpoint.json
python -m datagoblin.research_v2 run --registry runs/models-with-next-checkpoint.json \
  --model-id baby-next --binding /private/baby-next-binding.json \
  --suite /private/research-v2-confirmation --max-episodes 32 --out runs/baby-next-research-v2
```

The registration rejects an existing ID. Preserve training positions and the
training index SHA alongside the exact evaluated weight hash. Keep the task panel,
decoding condition and caps fixed when measuring a checkpoint's change over time.

## Jinn/Beast specialist intake

The pinned source is the private `MoralityLabAI/jinn-or-beast-paper` repository at
`01512c40be082ad88aa69594abca43a2b41908cc`.
[`specialist_sources.json`](../configs/research_v2/specialist_sources.json) records
the exact source bytes and original split counts. The importer reads a local
checkout or the three pinned files and recomputes the supplied policy labels with
exact rational arithmetic.

```bash
python -m datagoblin.research_v2.specialist --source-root /private/jinn-source \
  --held-registry /private/research-v2-confirmation/held_registry.json \
  --out /private/jinn-specialist-v2
```

The verified source has 2,304 distinct audited tag sets: 1,440 training, 432
validation and 432 test. Its two policy frames yield 2,880 training targets from
1,440 independent training roots. The importer checks all 4,608 original labels.
Original validation/test roots and the prospective research panel stay excluded
from training, including alternate views and counterfactual descendants.

The train-only output contains structured inputs, policy-attributed targets, root
identities and file hashes. It is staged for the specialist corpus's next intake.
Aleph tokenization, accepted token counts and mixture admission remain a separate
step, so these rows carry no credit toward the frozen 10M run or a larger token
quota. The supplied policy defines each target's meaning.

The source permits the user's internal intake in this workflow; public redistribution
rights are unestablished. Keep the training bundle and source policy definitions
private. This repository contains the importer, source metadata and synthetic tests.
The external persona task uses freshly generated generic policy tags and owner
events, preserving the private source's original holdouts.

## Validation

Run the repository gates and the adapter check:

```bash
python -m pytest
ruff check .
vf-validate datagoblin-research-v2 --runtime.type subprocess -n 8
```

The dedicated workflow exercises the focused tests and the actual pinned Verifiers
package. Tests compare every tiny-learner analytic gradient with central differences,
attack artifact authority and root custody, check deadline propagation, replay
reports, reject cap mismatches and test the transport with a local fixture server.
Generated runs, held seeds and model weights stay outside committed files.
