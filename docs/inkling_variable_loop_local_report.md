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
