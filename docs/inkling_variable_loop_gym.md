# Inkling variable-loop distillation gym for DataGoblin

The gym uses Thinking Machines Lab's variable-loop research checkpoint as a
*teacher* for the existing RLT rollout and Aleph programs. It measures how
solution correctness changes with architectural passes, then turns independent
verifier successes into small-model training examples. A separate student
selection command ranks RLT, TRM, LDT, Bonsai, and other compact candidates on
the calibration split, with the ordinary and OOD test splits reserved for
fresh confirmation.

The teacher changes the number of forward passes through layers 6–17. This is
not the RLT student's recurrent visit count. One teacher generation at pass 10
is one checked answer, not ten visible reasoning steps.

## CPU-only staging on Windows, WSL, Linux, or macOS

Requires Python 3.10+ and the existing DataGoblin development dependencies.

    python -m pip install -e ".[dev]"
    python -m datagoblin rlt-gym make-suite --seed 91 --per-family 4 --out runs/inkling/suite
    python -m datagoblin.training_gym.rlt_gym.inkling stage \
      --suite runs/inkling/suite --out runs/inkling/stage \
      --passes 1 2 3 5 7 10 --split train --max-tasks 28

The last command creates p01, p02, p03, p05, p07, and p10 teacher request
batches. Each contains identical task IDs and source worlds. A unique
model identifier binds each batch to the intended architectural pass count.
All requests are independently checkable with the existing RLT verifier.
Split custody, task digests, and task-visible history use the existing
RLT teacher-batch contract. Staging runs without model weights or a GPU.

For a smaller preflight, use --per-family 1, --max-tasks 7, and
--passes 1 3. The scorer handles real partial batches as missing responses.

## Capture from an operator-started local vLLM server

Model: thinkingmachines/TML-d24-varloop-research.
Provider documentation: https://vllm.ai/blog/2026-11-20-inkling2-mini
Research note: https://thinkingmachines.ai/news/inkling-2-mini/

The model's native on-disk footprint is about 20 GB and real inference needs
additional cache and runtime memory. Prepare on the RTX 5080 without loading
the full teacher; use a memory-suitable device or an operator-approved
offload/quantization setup for collection. The provided vLLM image targets
supported NVIDIA GPU platforms. An M4 48 GB run needs a compatible local
inference server implementing the same API and loop configuration.

The vLLM loop pass count is a SERVER-STARTUP setting. Restart the model
between the depth arms. Example for pass 1 on a supported NVIDIA host:

    docker run --rm --gpus all --ipc=host -p 8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      vllm/vllm-openai:inkling-2-mini \
      thinkingmachines/TML-d24-varloop-research \
      --reasoning-parser inkling \
      --hf-overrides '{"text_config":{"loop_num_passes":1,"loop_share_cache":false}}'

Run a no-network preflight first:

    python -m datagoblin.training_gym.rlt_gym.inkling capture \
      --stage runs/inkling/stage --passes 1 --server-loop-passes 1 \
      --out runs/inkling/captures/p01 --dry-run

Then run the same command without --dry-run to collect. The command writes
responses.jsonl, observations.jsonl, and a receipt in a FRESH directory. A
single local /v1/chat/completions call handles each staged task. The
max-requests, max-tokens, timeout-seconds, and max-wall-seconds flags cap
collection. The default URL accepts only loopback HTTP. The optional
--api-key-env reads a secret from an environment variable and excludes its
value from artifacts.

For each next arm, restart vLLM with loop_num_passes set to 2, 3, 5, 7, or
10, then change both --passes and --server-loop-passes, and the output
directory name, to p02/p03/p05/p07/p10. Keep loop_share_cache the same for
the whole sweep. The native checkpoint was midtrained on 1–5 passes, and 7/10
are extrapolative research conditions. Disable the fixed three-pass DSpark
speculator when collecting variable-loop comparisons.

The capture records the declared pass count and server response identity.
The ordinary OpenAI-compatible completion API cannot independently prove
which weight revision or loop configuration served a request; the provenance
flag explicitly says server_configuration_verified=false. Run outputs
should remain in the ignored runs directory.

## Score and export checked data

After every capture directory exists:

    python -m datagoblin.training_gym.rlt_gym.inkling score \
      --stage runs/inkling/stage --suite runs/inkling/suite \
      --captures runs/inkling/captures --out runs/inkling/scored

This routes EVERY captured answer through the existing RLT teacher importer.
The scorer replays the environment-owned receipts and writes:

* depth_matrix.jsonl: paired task × teacher pass outcomes and reported usage;
* sft.jsonl: one verified, train-split completion per lineage, chosen from
  the shallowest passing teacher pass count;
* preferences.jsonl: a matched-task pass/fail pair where a deeper pass
  succeeds after a shallower pass fails;
* curriculum.jsonl: train-only tasks to target in a later collection, with
  first passing depth and failed pass counts;
* manifest.json: source artifact hashes, exact receipts, unknown-token
  accounting, and extrapolation flags;
* pXX/: the original verifier-checked standard teacher import artifacts.

Token accounting uses server-reported prompt and completion counts, where
available. The layer-visit-token proxy is (12 + 12 × teacher passes) ×
reported tokens. It is not wall time, FLOPs, GPU power, or actual cost.

Models often vary in output length as internal computation increases. Keep
the correct-at-lowest-depth, fastest-effective, and fixed-budget curricula
separate in later experiments. The first export uses the shallowest
verifier-correct pass as a deterministic policy.

Any synthetic fixtures used to test the collector must declare
source_kind=synthetic_fixture. The scorer refuses these by default. With
--include-fixtures the rows are retained for engineering tests, visibly
marked fixture=true and student_training_eligible=false.

The student targets are environment-verified public answers and optional
public work artifacts, without claims about hidden recurrence or private
reasoning state. Future activation interventions must use a separately
instrumented native model and donor/sham/random controls.

## Hill-climb the small recurrent students

After training a local small-model variant, produce a prediction JSONL with
one row per CALIBRATION task, for every model being compared:

    {"model_id":"rlt_1b_depth4","task_id":"...","task_digest":"...","candidate":{"value":3},"visits":4,"latency_ms":12.4}

Use the identical frozen suite and candidate IDs for competing RLT, TRM,
LDT, and 1-bit Bonsai variants. The environment verifier checks predictions
independently. Exact full calibration coverage prevents cherry-picking.

    python -m datagoblin.training_gym.rlt_gym.inkling hillclimb \
      --suite runs/inkling/suite \
      --predictions runs/inkling/students/calibration_predictions.jsonl \
      --out runs/inkling/students/ranked

Ranking prioritizes verified task accuracy, then fewer recurrent visits;
self-reported latency is recorded but not used for promotion. A winner
requires a separately frozen confirmation on ordinary_test and ood_test.
The sealed_poison split is never part of architecture selection. Continue
using the existing RLT gym native runner for matched feedback-schedule
comparisons, state-pulse perturbation and exact replay.

## Extensions

Recommended next step after the first genuine local capture:

1. Add an SFT importer that maps the checked sft.jsonl to the existing
   1B RLT pretraining and HRM-Text corpus tokenization format.
2. Collect matched student RLT/standard-feedback/matched-damping
   rollouts, including first-valid visits and regressions.
3. Add a Q5-style spectral gate for recurrent stability and an optional
   nnsight/NDIF activation tracing lane for loop routing probes.
4. Extend training tasks with Lean goals, SQL/MCP lookup and longer
   serial-reasoning graphs, each with an independent verifier and
   lineage-isolated holdout.

The current implementation supports CPU staging and testing, bounded local
teacher capture, verified train-only export, and calibration ranking.
It does not claim to have run the research checkpoint, trained a 1B model,
or obtained real teacher scores.
