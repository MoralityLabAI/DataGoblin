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

Stage calibration separately with `--split calibration` and a fresh path.
Staging, capture and scoring all check that every pass receives the same
ordered requests, after excluding the depth-specific model and request ID.

## Pick a teacher runtime

The [official variable-loop IQ4_NL file](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF)
is 17,627,940,800 bytes (16.42 GiB), before the runtime or caches. The
[Unsloth Inkling 2 Mini GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
serves the standard fixed-three-pass model. The released variable-loop GGUF
supports 1–6 passes and uses separate per-pass caches. Use it for a matched
1-vs-3 feasibility pilot. Use a supported native runtime such as vLLM for a
matched six-arm sweep that includes 7 and 10. Keep the quantization, runtime,
cache policy, and output token budget fixed across a comparison.

For a memory-suitable host with a current Inkling-compatible llama.cpp build,
start with the seven-task, 1-vs-3 stage. Reserve 25 GiB on the target volume for
one text-only GGUF, download scratch and pilot artifacts. Verify headroom for
the 16.42 GiB file plus caches and runtime, and use an OS memory limit on a
dedicated host. The path below isolates the download for inventory and cleanup.

```sh
MODEL=thinkingmachines/TML-d24-varloop-research
GGUF_REV=618c48f1e3c54094333e3e652a92b1d99c52e34d
GGUF=TML-d24-varloop-research-IQ4_NL.gguf
python -m datagoblin rlt-gym make-suite --seed 7 --per-family 1 --out runs/inkling/fixture-suite
python -m datagoblin.training_gym.rlt_gym.inkling stage \
  --suite runs/inkling/fixture-suite --out runs/inkling/fixture-stage --passes 1 3 --max-tasks 7
hf download ggml-org/TML-d24-varloop-research-GGUF "$GGUF" \
  --revision "$GGUF_REV" --local-dir runs/inkling/weights
llama-server --model "runs/inkling/weights/$GGUF" --alias "$MODEL" \
  --host 127.0.0.1 --port 8000 --ctx-size 4096 --parallel 1 --no-mmproj \
  --override-kv inkling2_loop.loop.count=int:1
python -m datagoblin.training_gym.rlt_gym.inkling capture \
  --stage runs/inkling/fixture-stage --out runs/inkling/q4-captures/p01 \
  --runtime llama_cpp --weights-revision "$GGUF_REV" \
  --passes 1 --server-loop-passes 1 --max-requests 7 --max-tokens 512 \
  --timeout-seconds 180 --max-wall-seconds 1200
```

Run the capture in a second terminal. Restart the server with
`inkling2_loop.loop.count=int:3`, save its startup log, then repeat capture
with `--passes 3 --server-loop-passes 3` and `q4-captures/p03`. Score the
two depths against the 35-task fixture suite. Save the startup logs, runtime
version, weight file hash, peak RAM/VRAM, and time alongside the capture.
Shut down the dedicated server and record the retained files and released
memory after collection.

    python -m datagoblin.training_gym.rlt_gym.inkling score \
      --stage runs/inkling/fixture-stage --suite runs/inkling/fixture-suite \
      --captures runs/inkling/q4-captures --out runs/inkling/q4-score

## Capture from an operator-started local vLLM server

Model: thinkingmachines/TML-d24-varloop-research.
Provider documentation: https://vllm.ai/blog/2026-11-20-inkling2-mini
Research note: https://thinkingmachines.ai/news/inkling-2-mini/

The model's native on-disk footprint is about 20 GB and real inference needs
additional cache and runtime memory. Prepare on the RTX 5080 without loading
the full teacher. Use a memory-suitable device or an operator-approved
offload/quantization setup for collection. The release lists Blackwell,
Hopper, and DGX Spark support for its vLLM image. An A100 is outside that
documented support list. An M4 48 GB run can use an Inkling-compatible
llama.cpp build for the 1–6-pass quantized pilot.

The vLLM loop pass count is a SERVER-STARTUP setting. Restart the model
between the depth arms. Example for pass 1 on a supported NVIDIA host:

    docker run --rm --gpus 'device=0' --ipc=host --memory=32g --memory-swap=32g \
      -p 127.0.0.1:8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      vllm/vllm-openai:inkling-2-mini \
      thinkingmachines/TML-d24-varloop-research \
      --revision "$NATIVE_REV" --served-model-name thinkingmachines/TML-d24-varloop-research \
      --max-model-len 4096 --max-num-seqs 1 --gpu-memory-utilization 0.70 \
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

Set `NATIVE_REV` to the explicitly selected Hugging Face commit before
launch, and add `--weights-revision "$NATIVE_REV"` to each capture. Run
on a dedicated GPU with enough free VRAM for the model and caches. The
loopback client uses no redirects or proxy. It records the actual returned
model label and excludes mismatched responses from verification.

For each next arm, restart vLLM with loop_num_passes set to 2, 3, 5, 7, or
10, then change both --passes and --server-loop-passes, and the output
directory name, to p02/p03/p05/p07/p10. Keep loop_share_cache the same for
the whole sweep. The native checkpoint was midtrained on 1–5 passes, and 7/10
are extrapolative research conditions. Disable the fixed three-pass DSpark
speculator when collecting variable-loop comparisons.

Reserve 35 GiB on the checkpoint cache volume and pin the serving image.
For a depth-specific server identity, start vLLM with
`--served-model-name "thinkingmachines/TML-d24-varloop-research#passes=$PASSES"`
and pass the same value to `capture --served-model`. For llama.cpp, use
`--alias` with the same value. A response labeled for another arm will be
excluded. Record `--runtime-version "$RUNTIME_VERSION"` and
`--weights-artifact "$WEIGHTS_ARTIFACT"` alongside `--weights-revision`.
Use the Hugging Face repository for native weights or the repository plus
filename for a GGUF. Keep revision, artifact, and runtime matched across arms.

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

* depth_matrix.jsonl: paired task × teacher pass outcomes, usage and verifier receipts;
* sft.jsonl: one verified, train-split completion per lineage, chosen from
  the shallowest passing teacher pass count;
* preferences.jsonl: a matched-task pass/fail pair where a deeper pass
  succeeds after a shallower pass fails;
* curriculum.jsonl: train-only tasks to target in a later collection, with
  first passing depth, failed passes and priority for verified depth rescues;
* outcomes.jsonl: checker labels for unique candidate attempts, with the
  verdict outside the input;
* repairs.jsonl: shallower wrong answer paired with a deeper verified answer
  to the same task, with both receipts;
* aleph.jsonl: response-only Aleph corpus rows keyed by train root for SFT,
  outcome prediction, and repair. These rows use the existing corpus
  `root`, `input`, `target`, and `loss` fields. Fixtures are excluded;
* manifest.json: source artifact hashes, exact receipts, unknown-token
  accounting, and extrapolation flags;
* pXX/: the original verifier-checked standard teacher import artifacts.

Token accounting uses server-reported prompt and completion counts, where
available, including failed captures with reported usage. Scores show
both per-staged-task and per-definitive-outcome accuracy, plus the number
of jointly verified roots in every pass-pair comparison. The layer-visit-token proxy is (12 + 12 × teacher passes) ×
reported tokens. It is not wall time, FLOPs, GPU power, or actual cost.

Models often vary in output length as internal computation increases. Keep
the correct-at-lowest-depth, fastest-effective, and fixed-budget curricula
separate in later experiments. The first export uses the shallowest
verifier-correct pass as a deterministic policy.

The paired table gives each task family its own count of successes, failures,
rescues, regressions, and tasks without two definitive verdicts. The Aleph
adapter uses the corpus history and checker-code fields.

Any synthetic fixtures used to test the collector must declare
source_kind=synthetic_fixture. The scorer refuses these by default. With
--include-fixtures the rows are retained for engineering tests, visibly
marked fixture=true and student_training_eligible=false.

The student SFT targets contain only environment-verified answers.
Optional teacher public steps are retained as attributed, unverified metadata
rather than training targets. Future activation interventions require a separate
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

Supply `visit_candidates` with one candidate per recurrent visit to score
first-valid visit and regressions. Its last entry must match `candidate`.
An optional `fit_depth` reports how many episodes exceed that training depth.

### Reproduce the CPU control and held-out confirmation

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
python -m datagoblin rlt-gym run --suite runs/inkling/suite \
  --seeds 7 19 91 --depths 1 2 4 8 --training-modes direct rlt matched_damping \
  --modes direct rlt matched_damping --splits calibration --terminal-only \
  --max-wall-seconds 900 --out runs/inkling/cpu-calibration
python -m datagoblin rlt-gym replay --suite runs/inkling/suite \
  --run runs/inkling/cpu-calibration --out runs/inkling/cpu-calibration-replay.json
python -m datagoblin.training_gym.rlt_gym.inkling_controls \
  --suite runs/inkling/suite --calibration runs/inkling/cpu-calibration \
  --calibration-replay runs/inkling/cpu-calibration-replay.json \
  --out runs/inkling/cpu-heldout --max-wall-seconds 720
```

Run this lane under a 768 MiB OS job/cgroup limit. Calibration evaluates
first-verified stopping from exact prefixes of terminal trajectories and
charges every verifier call in those prefixes. It does not infer latency
for those prefixes. It locks choices by solved count, verifier calls, then
depth before confirmation, which executes and times real early stopping on
the ordinary and OOD splits. The sealed split stays sealed.

## Instrumented loop probes

### NDIF smoke test

Run the smoke test in a separate environment with access to PyPI, GitHub,
Hugging Face and NDIF. The script prompts for an NDIF key or reads the
process variable `NDIF_API_KEY`. It uses one public prompt and prints the
GPT-2 activation shape and next token or the Inkling layer-17 active-block
norms and observed pass count.

```sh
python -m pip install nnsight==0.8.3
python scripts/inkling_ndif_smoke.py
python -m pip install "transformers @ git+https://github.com/huggingface/transformers.git@add-inkling2-mini"
python scripts/inkling_ndif_smoke.py --model thinkingmachines/TML-d24-varloop-research
```

The hosted deployment sets the architectural pass count for each remote
trace. Use NDIF for mechanistic measurements at that setting and the local
server lane above for the matched teacher depth sweep.
[NNsight Inkling deployment and trace guide](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/)

### Local instrumentation contract

`inkling_instrumentation.LoopProbe` is separate from the completion API.
Implement it at a reproducible pass boundary with task digest, layer, token
and component identity. Expose the same one-pass map for activation reads,
Jacobian-vector and vector-Jacobian products, and sparse interventions.
Record the feature basis, dose, seed and donor provenance. Match donor
patches and lesions with sham and random-feature trials, then independently
verify affected readouts. The supplied diagnostic reports local spectral
and finite-dose directional gain. A sampled gain below one does not
establish a contraction bound across states, passes or task distributions.
