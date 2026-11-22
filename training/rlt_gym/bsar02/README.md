# BSAR-02: held-out Step 7 proof-scope evaluation

**Source:** [MathCan BSAR-02 PR #168](https://github.com/patrickdugan/MathCan/pull/168), merged at `53351f295aaaf669857638a081a4ac22f02ba0f9`. The upstream BSA-R Lean source is pinned to `VivienSweetbrier/bsa-r@80c62433b0eb671175bd8d6169822d3ebd922b60`.

The vendored `six_round_catalog.json` preserves all **129 original theorem declarations** and their premises. `boundary_curriculum.py` is copied from the merged MathCan reproducibility package, then styled for DataGoblin's Python linter. Its generator keeps the original deterministic 20261122 seed. Repeated helper names are held out with the **latest** round that contains them, so identical theorem names never span train/dev/test.

The Step 7 test split yields **164 examples from 36 declaration roots**, testing six claim categories. These labels distinguish source-presence evidence, missing mathematical premises, missing analytic types, unsupported PDE extrapolation, and unexecuted Lean-kernel evidence.

## Rebuild public evaluation inputs

```sh
python -m training.rlt_gym.bsar02.evaluation prepare --output-dir /tmp/bsar02
python -m training.rlt_gym.bsar02.evaluation majority-baseline --output-dir /tmp/bsar02
```

The model-facing `step7_public_tasks.jsonl` has IDs and source signatures with questions, **without** gold labels or explanations. Each result must be a line of JSON:

```json
{"id": "<ID from step7_public_tasks.jsonl>", "label": "KERNEL_CHECK_REQUIRED"}
```

Valid prediction labels: `SOURCE_DECLARATION_PRESENT`, `UNSUPPORTED_PDE_LIFT`, `MISSING_ANALYTIC_TYPE_BRIDGE`, `KERNEL_CHECK_REQUIRED`, `NAMED_PREMISE_PRESENT`, `MISSING_PREMISE`, or `ABSTAIN`.

```sh
python -m training.rlt_gym.bsar02.evaluation score \
  --predictions /tmp/bsar02/model_predictions.jsonl \
  --output /tmp/bsar02/model_metrics.json
```

The scorer enforces task IDs, rejects duplicate predictions and leaked gold-label fields, and reports accuracy including abstentions, selective answer accuracy, macro-F1, per-template performance, and complete-root success. The constant-label `majority-baseline` is a **non-model control**, expected to score 36/164 (21.95% strict accuracy) and 0.06 macro-F1.

## Run a local GGUF checkpoint

Start a local OpenAI-compatible llama.cpp server with your chosen GGUF and a context window appropriate to VRAM. For the memory-constrained Inky Mini probe, begin with **12,288 tokens** and test 16 tasks before running all 164:

```sh
llama-server -m /path/to/model.gguf -c 12288 --port 8080
python -m training.rlt_gym.bsar02.local_gguf \
  --model YOUR_SERVER_MODEL_ID \
  --predictions /tmp/bsar02/inky-mini.jsonl \
  --limit 16
```

Rerun with `--limit 164` to resume uncompleted IDs. The client accepts loopback HTTP only, temperature zero and at most 48 generated tokens per question, retaining raw answers and token counts. It rejects mixed-model resumes. A server connection failure stops the run with existing completed results preserved.

This local client was implemented and mock-tested in CI; **no real Inky Mini, Bonsai or DataGoblin checkpoint has been run by this PR**. Measured checkpoint results need separate receipts identifying the GGUF weights and quantization, runner version, GPU, context size, and input hash.

## Validity limitations and next experiment

These labels describe the **scope of a published theorem statement**, and the repo remains openly available. The test split is public, meaning a model may have seen some theorem signatures or the six question templates. Identical prompt templates across source splits allow shallow template learning and should be treated as a leakage risk even with theorem-name separation.

Before claiming genuine reasoning improvement, create a fresh confirmation set using **new theorem bodies and unseen claim formulations**. Require statement-level mathematical judgments on independently type-checked proof targets, and report performance by root as well as by example. Any classification of a Navier–Stokes global-regularity claim remains conditional on the actual analytic proof and cannot be established by this labeling gym.
