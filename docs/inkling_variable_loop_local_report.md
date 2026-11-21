# Inkling variable-loop gym local run, 2026-11-21

## Collection and resource receipts

The seven-family, 35-task fixture supplied seven identical training requests
at passes 1 and 3. The 140-task suite used seed 91 and four roots per family
in each of train, calibration, ordinary test, OOD test, and sealed poison.
Train and calibration each have 28 requests per pass at 1, 2, 3, 5, 7, and
10, for 336 staged requests across the two splits. All three stages passed
the ordered request comparison. Staging made zero model calls.

| Resource | Preflight |
| --- | --- |
| CPU | 20 logical processors |
| Host memory | 48,249 MiB visible to Windows, 4,887 MiB available, 7,088 MiB commit headroom |
| RTX 5080 Laptop | 16,303 MiB total, 5,602 MiB free, 93–100% active, driver 577.05 |
| Disk | About 60 GiB free |
| CPU control limit | One math thread, no CUDA, 768 MiB Windows job commit cap |
| Calibration | 416.84 s wall time, 153.5 MiB peak job commit |
| Calibration replay | 123.2 MiB peak job commit |
| Held-out execution and replay | 161.1 MiB peak job commit, 176.22 s rollout wall time across arms |

The official variable-loop IQ4_NL weight file is 17,627,940,800 bytes at
revision `618c48f1e3c54094333e3e652a92b1d99c52e34d`. It exceeds total
VRAM on this card. The measured CPU and GPU headroom ruled out a
resident/offloaded run with working memory while the other workload was
active. The release's llama.cpp port supports 1–6 passes. A dedicated
memory-suitable host can run the first pilot on this quantization. Native
vLLM is the documented path for the full six-arm study. The alternative
M4 48 GB and A100 80 GB hosts were not allocated in this run.
[Official GGUF release and loop configuration](https://huggingface.co/ggml-org/TML-d24-varloop-research-GGUF/blob/618c48f1e3c54094333e3e652a92b1d99c52e34d/README.md)
[vLLM release instructions](https://vllm.ai/blog/2026-11-20-inkling2-mini)

The local Python environment had PyTorch, Transformers, Hugging Face Hub,
and NumPy. It had neither the vLLM nor llama.cpp Python package. The
existing llama-server executable predated this release, and its support
for the checkpoint was unverified.

Genuine teacher calls, checked teacher attempts, teacher tokens, and
production Aleph examples were all zero. No weights were downloaded and
no teacher server was started. The offline scoring fixture produced 14
checked matrix rows, seven SFT rows, seven preferences, seven curriculum
rows, 14 outcome rows, and seven repairs. The adapter excluded all fixture
rows. The fixture tests exercise receipt binding and export routing.

## Recurrent proxy controls

The CPU controls fit only the train split. They use a frozen NumPy recurrent
reservoir with a fitted readout and a supplied candidate pool. Calibration
crossed direct, RLT, and matched damping at fit and runtime across seeds
7, 19, and 91 and depths 1, 2, 4, and 8. The 3,080 calibration episodes
include 56 oracle/candidate-order controls. All replayed exactly and replay
independently checked 11,436 candidates. The rollout SHA-256 is
`c8d2637ca6273bda799151f12965a51f991f5d0c64a1195641d08bddecff9e2c`.

Each cell uses 84 episodes sharing 28 calibration roots across the three
seeds. Calibration fixed-depth scores use the terminal candidate and
charge checks at every visit. Early-stop scores use the exact prefix through
the first verified candidate. Selection ranked solved count, then fewer
checks, then lower maximum depth.

| Prespecified arm | Runtime depth | Calibration solved / 84 | Checks |
| --- | ---: | ---: | ---: |
| Direct fixed | 1 | 25 | 84 |
| Direct early | 4 | 28 | 253 |
| RLT fixed | 1 | 27 | 84 |
| RLT early | 8 | 29 | 481 |
| Damping fixed | 1 | 27 | 84 |
| Damping early | 2 | 29 | 141 |
| RLT fit, direct early | 8 | 31 | 461 |

Depth 8 extrapolates from the depth-4 fit. In the matched RLT depth-8
terminal arm, 29 episodes found a valid candidate and 26 finished valid.
The analogous numbers were 28 and 25 for both direct and damping.

All seven frozen arms ran on fresh ordinary and OOD roots and independently
replayed. The 1,176 episodes covered 56 distinct held-out roots across the
three seeds and accounted for 3,244 verifier calls. Each split contributes
84 episodes per arm. The checks column sums both splits.

| Confirmed arm | Ordinary solved | OOD solved | Checks | Checks / solution |
| --- | ---: | ---: | ---: | ---: |
| Direct fixed | 26 | 27 | 168 | 3.17 |
| Direct early | 27 | 28 | 510 | 9.27 |
| RLT fixed | 23 | 26 | 168 | 3.43 |
| RLT early | 27 | 28 | 980 | 17.82 |
| Damping fixed | 24 | 26 | 168 | 3.36 |
| Damping early | 26 | 28 | 284 | 5.26 |
| RLT fit, direct early | 27 | 28 | 966 | 17.56 |

The crossed calibration winner tied direct early on both holdouts and spent
456 more checks. On those two splits, time per crossed episode averaged
203.9 and 228.2 ms, including verification and trace output. Its first
valid visit among solved cases averaged 1.22 and 1.04. Direct early averaged
143.6 and 144.9 ms with first valid visits of 1.07 and 1.04. The trace
summaries retain visit-level state, candidate, verifier and latency records.

Use depth-1 direct as the default low-check control for the next experiment.
Challenge it with confidence-gated second visits and depth-2 matched damping
under the same verifier allowance on fresh roots. The held-out improvement
from always permitting deeper recurrence was two episodes out of 168 at
an additional 342 checks for direct early, and the crossed arm matched
that solve count at higher cost. These comparisons are limited to the
fitted CPU candidate ranker. There were no trained TRM, LDT, Bonsai or
teacher-distilled Aleph students in this run.

Across calibration and holdout, all 4,256 trajectories replayed with
14,680 independently rechecked candidates. The run's data artifacts
occupy about 40 MB outside Git.

## Checks and resumption

The focused Inkling suite passed all six tests. It covers matching, fixture
exclusion, capture identity and token accounting, re-verification, export,
selection custody, and a known linear stability probe. Ruff passed on the
changed Python modules and tests. The Q4 planning command wrote a dry-run
receipt for seven requests and made zero network calls.

The editable development install succeeded in a fresh local environment.
The six Inkling tests and Ruff also passed there. The full local suite
completed with 283 passed and one skipped in 1,091.59 seconds, peaking at
159.1 MiB under the 768 MiB Windows job limit.

PR #5's previous Inkling job failed before runner assignment. The
job record has no runner name or steps and the logs endpoint returned
`BlobNotFound`. The base CI and RLT jobs on the same SHA also had no
recorded steps. The Actions API exposed no scheduler rejection reason.
The change removes the duplicate feature-branch push trigger while
keeping PR and manual runs. [Previous hosted Inkling run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44871178089)

The first hosted run after the change also had an empty step list and a
`BlobNotFound` response for its logs. [Post-push hosted run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44875460358)

See [the runbook](inkling_variable_loop_gym.md) for the exact 1-vs-3 and
six-arm commands and the CPU control reproduction. Local output is in the
`work/inkling-varloop-local-20261121` sibling directory, including full
trajectories, replay receipts, frozen requests, and scored fixture data.

## Follow-up on the Windows workstation

The tracked source was recovered from the prior local copy and verified
file-by-file against PR head `5a6e08030997c5536c6d9939f3d9ba45225595d0`.
The editable development install succeeded in a fresh Python 3.12.10
environment using cached wheels. The focused Inkling suite passed all
six tests, and repository-wide Ruff passed. The full suite completed
with 263 passed and 21 skipped in 60.06 seconds in the development-only
environment. The run used a short, writable temp directory outside the
checkout for Windows path length and custodian tests. The NDIF smoke
script passed its CLI check.

The seed-7 smoke suite regenerated 35 tasks and 14 matched requests across
the seven training roots. Seed 91 regenerated 140 tasks and 168 requests
in each of the separate train and calibration stages. The large stages
each cover 28 roots at six depths. Ordered match checks passed for all
three stages. The depth-bound IQ4_NL capture dry run planned seven calls
and made zero.

The synthetic verifier fixture produced 14 matrix rows, seven SFT rows,
seven preferences, seven curriculum rows, 14 outcome rows, and seven
repairs. It produced zero eligible SFT rows and zero Aleph training rows.
The scorer now includes the verifier receipt with each outcome, breaks
down paired gains and regressions by family, prioritizes verified rescues,
and writes Aleph-compatible repair history and checker codes. Capture
accepts a per-depth alias and keeps runtime version and weight artifact
matched across arms. Identity mismatches retain reported usage without
attributing it to the intended architecture.

### Fresh CPU control replay

A new suite used generator seed 19 with one root per family in each
split. The proxy fit crossed direct, RLT, and matched damping at fit and
runtime, using model seeds 7, 19, and 91 and depths 1, 2, 4, and 8.
Calibration ran 770 episodes in 25.47 seconds. The six selected arms
ran 252 ordinary and OOD episodes in 11.03 seconds of rollout time.
All 1,022 trajectories replayed and 3,183 candidates were independently
rechecked. Calibration rollout SHA-256 was
`982a1e51b8b2ce5259b2a0d8f6bc3544190d689dc1292449aecf76410a9b45fc`.

| Selected arm | Depth | Calibration solved / 21 | Ordinary solved / 21 | OOD solved / 21 | Held-out checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct fixed | 2 | 5 | 6 | 5 | 84 |
| Direct early | 2 | 5 | 7 | 6 | 72 |
| RLT fixed | 1 | 6 | 7 | 5 | 42 |
| RLT early | 1 | 6 | 7 | 5 | 42 |
| Damping fixed | 1 | 6 | 7 | 5 | 42 |
| Damping early | 1 | 6 | 7 | 5 | 42 |

For direct early, ordinary and OOD latency averaged 31.29 ms and
29.76 ms per episode respectively. The proxy early-stop arm made one
additional solve on the combined holdouts relative to either depth-1
matched arm at a cost of 30 additional checks. Keep a depth-1 floor and
compare confidence-gated second visits against a random gate with the
same check allowance on new calibration roots. Freeze that threshold
before ordinary and OOD confirmation. The held-out episodes share roots
across model seeds. These are CPU proxy results. Genuine teacher attempts
and teacher-trained student fits remain at zero.

### Hardware, collection, and instrumentation

| Measurement | Windows workstation |
| --- | --- |
| NVIDIA driver / CUDA driver API | 577.05 / 12.9 |
| RTX 5080 Laptop VRAM | 16,303 MiB total |
| Free VRAM | 5,602 MiB initially, 15,915 MiB on the later sample |
| Host RAM | 48,248 MiB visible, 12,298 MiB available on the later sample |
| Host commit headroom | 24,556 MiB on the later sample |
| Disk | 71.38 GiB available on the later sample |

The native checkpoint requires a larger single-device memory budget.
The 16.42 GiB IQ4_NL file also exceeds this GPU's total VRAM. The later
CPU and GPU samples make partial offload worth testing with a compatible
llama.cpp build and the runbook's one-stream, 4,096-token pilot. This
shell could not resolve Hugging Face or PyPI, and no checkpoint cache
or compatible runtime was confirmed. WSL returned access denied. No
download started, no teacher server ran, and no unrelated GPU job was
interrupted. This recheck retained about 11 MB of generated run data
under the ignored `runs/inkling-recheck` directory.
[Native checkpoint requirements](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)

NDIF documents a deployment of this exact research checkpoint. The
deployment fixes the pass count for a remote trace. The optional smoke
script uses one public prompt for either GPT-2 or the Inkling layer-17
active-block readout. This shell could not install NNsight or resolve
`api.ndif.us`, so no authenticated trace ran.
[NDIF Inkling tracing and pass configuration](https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/)

The latest inherited Inkling check failed two seconds after creation.
The job API returned `runner_id: 0`, an empty runner name, and
`steps: []`. The logs returned `BlobNotFound`. This was a pre-runner
failure. GitHub records an annotation, but the available job and logs
endpoints do not expose its text, so the reason for declining assignment
is unconfirmed.
[Hosted run and job](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44875895278/job/136760385113)

The [runbook](inkling_variable_loop_gym.md) contains the exact collection
and scoring commands. Start with the seven-task 1-vs-3 pilot on a
memory-suitable host and keep each arm's startup log with its receipt.
Collect the large train and calibration stages into separate directories.
