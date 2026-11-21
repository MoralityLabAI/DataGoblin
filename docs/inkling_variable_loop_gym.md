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

`inkling_instrumentation.LoopProbe` is separate from the completion API.
Implement it at a reproducible pass boundary with task digest, layer, token
and component identity. Expose the same one-pass map for activation reads,
Jacobian-vector and vector-Jacobian products, and sparse interventions.
Record the feature basis, dose, seed and donor provenance. Match donor
patches and lesions with sham and random-feature trials, then independently
verify affected readouts. The supplied diagnostic reports local spectral
and finite-dose directional gain. A sampled gain below one does not
establish a contraction bound across states, passes or task distributions.

### Native GGUF boundary capture on Windows

The native tracer reads the ggml eval callback from llama.cpp b12602
(`418fe97b5`). Build it in an x64 Visual Studio Developer PowerShell
with the headers from that exact revision. The build script derives the
import libraries from the installed DLLs and runs a local binding check.
Keep the downloaded headers, binaries and activation files outside Git.

    ./scripts/build_inkling_trace.ps1 -Runtime <llama-dir> -Headers <pinned-headers> -Out <build-dir>

Freeze raw public-context prefills from the same suite as the teacher
packets. Use the separate train and calibration stages to preserve lineage
custody through capture and fitting.

    python -m datagoblin.training_gym.rlt_gym.inkling_trace stage --suite <suite> --stage <train-stage> <calibration-stage> --out <prompts>

Run the native executable under the same guarded Windows process supervisor
used for the model. Supply the GPU layer count and the newline-delimited
CPU tensor patterns from the measured fit plan. Bind the RTX 5080 by its
runtime device name and put the matching DLL directory on `PATH`.

    inkling_trace <llama-dir> <gguf> Vulkan1 3 <gpu-layers> <cpu-patterns.txt> <prompts> <trace> mmap

The runner also accepts `direct-io`, which requests unbuffered reads and
avoids the memory-mapped loader's whole-file prefetch. It requires commit
headroom for the CPU buffers and still uses the same host and GPU memory
guards. The native config records the requested mode. Check the llama.cpp
load log for actual direct I/O and any fallback to buffered reads.

On Windows, the optional `scripts/inkling_no_prefetch.patch` for
llama.cpp `418fe97b5` keeps CPU-resident weights memory-mapped while
skipping the full-file prefetch. Apply it to a local source copy and build
the `llama` target against the matching portable GGML build. Set
`DATAGOBLIN_SKIP_GGUF_PREFETCH=1` for the guarded `mmap` launch and
confirm the loader reports that it disabled prefetch. Record the loaded
`llama.dll` digest alongside the backend archive digest on each arm.

If device upload still fills the workstation's file cache, apply
`scripts/inkling_stream_gpu_weights.patch` after the no-prefetch patch
using `git apply --unidiff-zero` at the pinned revision. Rebuild
`llama` and set `DATAGOBLIN_STREAM_GPU_WEIGHTS=1` for the guarded
`mmap` run. CPU tensors stay mapped. Device tensors use 4 MiB
unbuffered reads. Check for the device-weight byte count in the load
log and use the same loaded library for all matched depth arms.

The pass-0 record is the shared prelude output. The first core pass ends at
`l_out-17`. Repeated core passes end at logical cache slots 35, 47 and later
steps of 12 while reusing the original physical weights. The tracer checks
the 2,304-wide float32 tensor shape for every token at each boundary.

    python -m datagoblin.training_gym.rlt_gym.inkling_trace seal --prompts <prompts> --trace <trace> --passes 3 --weights-sha256 <weight-sha> --runtime-sha256 <runtime-archive-sha>
    python -m datagoblin.training_gym.rlt_gym.inkling_trace export-sae --prompts <prompts> --trace <trace> --out <sae-input>

The export provides independent `train.npz` and `calibration.npz` matrices
under the VPD trainer key `inkling_core_prefill`. Row manifests retain
the task digest, lineage, tensor and token position. Fit on the train
matrix and use the calibration matrix for reconstruction checks. These
raw prefills provide an observation lane. Behavioral intervention studies
need a shared rendered prompt, a verified answer endpoint and matched sham
and random controls.

The bounded NumPy fitter writes a dictionary with the same weight arrays
as the VPD top-k trainer. It reports both per-pass and pooled calibration
reconstruction with a bias-only baseline.

    python -m datagoblin.training_gym.rlt_gym.inkling_sae --inputs <sae-input> --out <sae>

Train comparable feedback heads using the same measured prelude and core
states. Each head fits the one-step target after applying its feedback
coefficient, then rolls forward from the held-out prelude with its own
predicted states. The report gives activation and SAE-code errors for
direct feedback, RLT and matched damping.

    python -m datagoblin.training_gym.rlt_gym.inkling_rlt_trace --prompts <prompts> --trace <trace> --inputs <sae-input> --sae <sae> --out <heads>

The feedback test measures prefill-state prediction. Verify task-level
effects separately through the rollout checker.

### Native answer evaluation and SAE screening

Stage the train and calibration packs with matched depths and completion
mode. The prompt uses the checkpoint's text-only chat format, zero
thinking effort, and a prefilled answer channel. Keep this protocol fixed
across the native arms. It is a separate comparison from the server's
reasoning-enabled runs.

    python -m datagoblin.training_gym.rlt_gym.inkling_trace stage --suite <suite> --stage <train-stage> <calibration-stage> --out <prompts> --per-stage 7 --completion

Append a generation budget of up to 384 tokens to the native executable
command. It uses greedy decoding and flushes each completed prompt and
reply. The import can recover a completed prefix if a resource guard stops
the process. Import each depth and split before running the existing
Inkling scorer. Keep the native runtime, executable and model digests in
the capture manifest.

    inkling_trace <llama-dir> <gguf> Vulkan1 3 <gpu-layers> <cpu-patterns.txt> <prompts> <trace> mmap 192
    python -m datagoblin.training_gym.rlt_gym.inkling_native --stage <train-stage> --prompts <prompts> --run <trace> --out <captures/p03> --passes 3 --weights-revision <revision> --weights-sha256 <weight-sha> --runtime-sha256 <runtime-archive-sha> --program-sha256 <executable-sha>

For the SAE screen, match first and final core states at the same token
position. Select feature IDs and directions on training roots, then
evaluate them on calibration roots with a max-statistic sign-flip test.
Each root contributes one observation per feature. The shortlist prepares
feature interventions. It measures directional changes in activation and
does not establish an effect on verified outcomes.

    python -m datagoblin.training_gym.rlt_gym.inkling_vpd_screen --inputs <sae-input> --sae <sae> --out <feature-screen>

If both native answer arms complete, refit the dictionary and feedback
heads using the trace from the same rendered prompts as the evals. The
depth selector uses predicted first-to-final movement from the prelude to
choose between one and three passes. The threshold is fitted on verified
training outcomes. On calibration, replay the selected depth against the
already observed outcomes and compare it with both fixed-depth policies.
The recorded layer-visit token measure is an analytical cost proxy. The
selector replay does not include a live deployment of the prelude router.
For a policy that runs depth one and restarts selected prompts at depth
three, charge the observed depth-one work on every root plus the full
depth-three work on escalated roots. Compare that total with both fixed
depths. If routing precedes verification, count checker calls on the chosen
outputs and record both generations on escalated roots.

    python -m datagoblin.training_gym.rlt_gym.inkling_routing --prompts <prompts> --trace <three-pass-trace> --sae <sae> --heads <heads> --train <verified-train> --calibration <verified-calibration> --out <routing>

### Held-out depth evaluation

Use the frozen native holdout plan with ordinary and out-of-distribution
test lineages. Run each staged batch at depths 1, 3 and 6 with the same
checkpoint, prompt, placement and output budget. The scorer reads each
arm from `<runs>/<batch>/p<depth>/raw` and checks the sibling
`command.json` against the frozen plan and model provenance.
To resume an interrupted arm, retain its original capture and write the
remaining prompts to the next numbered `continuations/<number>/raw`
segment. The segment's `command.json` must record its zero-based
`prompt_offset` and the same runtime provenance. The scorer joins
segments only when each starts after the previous durable prefix.

    python -m datagoblin.training_gym.rlt_gym.inkling_holdout --suite <suite> --holdout <holdout-plan> --runs <runs> --reference <native-eval-02> --fit-prompts <fitting-prompts> --out <audit>

The report scores durable partial results and compares depths on
matched roots. It records the first successful depth, regressions,
parser failures, and independent checker work. The frozen routing
replay uses only the one-pass prelude state to choose a depth and
charges both generations when it restarts at three. Test results are
held in the evaluation audit.

### Matched SAE readout perturbation

Screen the same token and core output that the executable will perturb.
Pass `--prompts <prompts> --trace <three-pass-trace> --target-pass 3` to
`inkling_vpd_screen` for the last prompt token at the final core output.
Use `--target-pass 2` to study propagation into the third pass. Freeze the
first train-ranked feature admitted by that screen. On fresh ordinary-test
roots, a signed decoder direction opposes the training-set contrast.
The executable records the realized vector norm.
Each root runs untreated, sham, orthogonal matched random, and feature
arms under identical prompt text and decoding budgets.

    python -m datagoblin.training_gym.rlt_gym.inkling_feature_trial stage --suite <suite> --sae <sae> --screened <screen> --out <trial>
    inkling_trace <llama-dir> <gguf> Vulkan1 3 <gpu-layers> <cpu-patterns.txt> <trial/prompts> <run> mmap 192 <trial/deltas>
    python -m datagoblin.training_gym.rlt_gym.inkling_feature_trial score --suite <suite> --trial <trial> --run <run> --out <verified> --native-program-sha256 <executable-sha>

Supply the registered intervention pass after `<trial>/deltas` on the native
command line. The default is the final pass. A pass-two perturbation keeps
the decoder at depth three, and its matched comparisons report each arm's
last-token state distance from untreated at pass three.

For a signed query-gain parameter trial, stage the pass-two screen with
`--parameter-preflight <frozen-factors-dir> --parameter-coefficient 1` or
`--parameter-coefficient -1`. The preflight binds its rank-one factors
to the checkpoint, dictionary and trace, with equal Frobenius doses for
the selected and orthogonal random directions. At the registered token
and pass, the executable adds the factorized parameter delta's product
with the live gate input to the output of
`blk.6.loop_q_ctrl_gain.weight`. This is equivalent to using perturbed
weights for that matmul. The checkpoint stays read-only. The receipt
records both the parameter dose and the realized projection change.

The verifier checks each answer and enforces the existing BPD matched-arm
contract on complete roots. This study supplies a signed SAE intervention
receipt. Parameter-level VPD and BPD decomposition require their own
analysis and validation.
