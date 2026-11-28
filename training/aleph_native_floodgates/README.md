# AFG-01 native Aleph Floodgates

This is a source-compatible decoder for the frozen Direct A training topology.
See [registered protocol](PROTOCOL.md), recovery [PR #74](https://github.com/MoralityLabAI/DataGoblin/pull/74),
and the [native 10M export receipt](https://github.com/MoralityLabAI/DataGoblin/tree/master/pre-trains/aleph-direct-a-10m-v2).
The model class and independent cold reference are imported from the exact
pinned training source. This branch is stacked on recovery PR #74 because
the training source does not exist on the current default branch.

## CPU fixture

From the repository root, using Python 3.11, PyTorch CPU 2.15.0 and NumPy:

```sh
python training/aleph_native_floodgates/native_cache.py --fixture-out /tmp/afg01
python training/aleph_native_floodgates/native_cache.py --audit /tmp/afg01
```

The fixture instantiates the native Aleph class at width 32 and checks 24
logical visits, 14 physical blocks, GQA, absolute-offset interleaved RoPE,
tied output, chunked append, physical truncation, forced mismatches,
generation cap, stale/foreign/misordered caches and cold-reference argmax
parity. It trains two tiny rank-four mask-drafting adapters on a frozen
random-weight target, then independently refits and replays 16 episodes.
This is a source plumbing fixture, with no claim of learned language skill.
FP32 must pass; any BF16 mismatch blocks that precision separately.

## Run the real checkpoint

The actual 10M export is recorded under
`MyDrive/aleph-v2-run/export/native/model.safetensors`. Stage that file
locally under an operator-controlled directory. The current development
environment cannot access the stored weights. This script does not fetch
or provision them.

Create a small JSON file containing one to four arrays of already
tokenized prompt IDs (two to 32 tokens each). Use the frozen SmolLM2
tokenizer and its original raw completion framing for actual text prompts;
the qualification script does not tokenize or assume a chat template.
Example engineering prompt panel: `[[1,2,3,4],[100,101,102,103]]`.

```sh
python training/aleph_native_floodgates/native_cache.py \
  --checkpoint /local/export/native/model.safetensors \
  --ids-json /local/prompt_ids.json --precision fp32 --max-new 8 \
  --threads 2 --out /local/afg01-full-fp32.json
```

Require the 3,997,421,432-byte native FP32 safetensors file with SHA-256
`2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8`.
Load via a meta-device model and strict state_dict assignment; no optimizer,
untied expansion, guessed tensor-key remapping, or quantized substitute.
If the artifact's keys differ from the exact native source class, abort
and inspect its manifest before adding a separately registered mapping.
Full-size allocation and execution require external process-tree RAM and
wall caps. This implementation is CPU-first and does not start a GPU or
Colab runtime. It qualifies parity, not latency or sampling equivalence.

After FP32, run a separate BF16 qualification if supported by the local
CPU. The target has 24 per-visit unexpanded KV pairs, consuming 98,304
bytes per committed token in FP32 or 49,152 in BF16, excluding all
temporary buffers and allocator overhead. Full context of 2,048 committed
tokens would require 192 MiB or 96 MiB of persistent KV, respectively.
No sharing the two core passes' K/V despite tied weights.

Successor gates: real checkpoint parity, teacher/adapter train-custody
panel on a competent checkpoint, draft-only adapter fit, full greedy
qualification, then actual serving timing at matched precision and
context. Do not infer native acceleration from the smaller HybridTRMLDT
results. All fixture traces are development-only and training_ready=false.

## AFG-01 result

[Native fixture campaign 46104294299](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46104294299) passed FP32 and BF16 cache qualification and audited all 16 adapter rollouts. FP32: 119,552 compared values, 116 exact argmax positions, max absolute difference 2.92e-6, zero greedy mismatches. First/deep cache emitted 1.778/1.600 tokens per verification. Random-weight reduced-width target only. The separate repository CI passed 660 tests (27 skips) and found nine new lint diagnostics in native_cache.py; these are corrected without changing the archived training sources.

## AFG-02: trained native synthetic targets

See [prospective protocol](AFG02_PROTOCOL.md). Two independently fitted native targets, disjoint target/adapter/evaluation prefixes, a declared CAL competence gate, trained and untrained first/deep adapters, and serial/repetition baselines. CAL selects a draft length for each policy; all final IDs must equal the independently checked greedy target. The audit refits both targets and adapters, replays CAL selection and compares every trace. The default campaign plans 192 dedicated trace episodes. All output is nonadmitted public development material. This does not use the actual 1B checkpoint.

```sh
python training/aleph_native_floodgates/afg02_trained.py --out /tmp/afg02
python training/aleph_native_floodgates/afg02_trained.py --audit /tmp/afg02
```

## AFG-02 results

[Both native jobs passed](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46104930110).
Two competent targets, four adapters and 192 audited traces, zero greedy
mismatches. Development free-running accuracy was 93.75% and 96.875%.
Shift accuracy was 27.34375% and 31.25%. Mean development structural
tariff/token: repeat 77.53125, serial 120, first trained 159.421875,
deep trained 173.671875, first initial 214.640625, deep initial 189.703125.
Both trained drafters improved relative to their untrained controls,
but neither repaid the 14-visit draft cost. The cheap task-specific
repeater was roughly 2.34x faster than serial in the diagnostic decode
timing (mean of two paired seed ratios; excludes prefill). No native
1B speed claim. One remaining repository lint diagnostic (import order
in afg02_trained.py) was corrected.

## AFG-03: actual checkpoint on the 5080

[Prospective protocol](AFG03_PROTOCOL.md). The new runner first requires
the exact native 10M checkpoint SHA and a separately hash-bound local
token panel. It uses real checkpoint weights and exact teacher-generated
continuations; the frozen target remains FP32, dense compute defaults
to BF16, TF32 disabled. Each train prefix has a CPU-resident unexpanded
KV bank; only the selected slice is transferred for one adapter update.
Host bank tensor budget 256 MiB; PyTorch CUDA allocator default 10 GiB.
The latter does not cap all WDDM/driver/process memory. Set an external
RAM and wall watchdog, and leave GPU headroom.

Token panel JSON (illustrative single row only; supply the full roster
of 8–32 train, 4–16 cal, 4–16 dev rows):

```json
{
  "schema": "aleph.afg03.token_panel.v1",
  "tokenizer_sha256": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
  "rows": [
    {"split": "train", "group": "source-1", "ids": [1,2,3,4,5,6,7,8]}
  ]
}
```

Use already-tokenized raw completion prompts with the pinned tokenizer.
Keep all derivatives of each source document in the same split. No
answer labels, chat templating or cross-split exact prefix containment.
Do not pass proprietary prompt IDs to a remote service. On Windows
PowerShell from the clean checkout:

```powershell
$panel = 'D:\\aleph\\afg03-panel.json'
$ckpt = 'D:\\aleph\\export\\native\\model.safetensors'
$sha = (Get-FileHash -Algorithm SHA256 $panel).Hash.ToLower()
python training/aleph_native_floodgates/afg03_checkpoint.py --execute `
  --checkpoint $ckpt --data $panel --data-sha256 $sha `
  --device cuda:0 --precision bf16 --allocator-gib 10 --steps 64 `
  --max-new 8 --repeats 3 --threads 2 --wall-seconds 1800 `
  --out 'D:\\aleph\\afg03-5080-bf16'
```

The paths are examples; point them to your actual files. Use a CUDA
PyTorch build qualified for the RTX 5080, plus NumPy and safetensors.
If parity fails, inspect progress.json and stop. An explicitly separate
FP32 run can be made with a fresh output directory; it is a different
precision cell, not an automatic retry. No GPU run or model retrieval
is triggered by repository CI.

Compare all six policies at their CAL-selected lengths, including
ordinary serial and repetition fallback. Measurements include prefill,
decoder calls, cache rollback and host accounting, and synchronize CUDA.
Three repeats per policy must agree in emitted token IDs and work.
The report preserves phase memory/fit timings, teacher work, policy
receipts, weights and the calibration-selected policy. The 10M
checkpoint is not a competent language-model benchmark.

CI exercises the same code on a reduced-width CPU fixture:

```sh
python training/aleph_native_floodgates/afg03_checkpoint.py --fixture --out /tmp/afg03
python training/aleph_native_floodgates/afg03_checkpoint.py --audit-fixture /tmp/afg03
```

The fixture refits and replays the local runner; a real GPU run requires
the operator's checkpoint and prompt panel. No GPU speed claim before
that qualification.

### Local RTX 5080 procedure

Four helpers sit beside the runner:

- `afg03_panel.py` builds the hash-bound token panel with the pinned
  SmolLM2 `tokenizer.json`: 32 source documents (eight each of general
  prose, Python standard-library code, worked mathematics and public
  protocol text from this repository), two line-aligned windows of
  12–32 tokens per document, documents 1–4/5–6/7–8 of each category in
  train/CAL/dev, giving 32/16/16 rows with no cross-split groups. It
  writes the panel and a hash-only inventory to a local directory.
- `afg03_watchdog.py` runs a command as a child and samples process-tree
  working set and private bytes, system available memory, the child's
  dedicated and shared (WDDM) GPU memory from the Windows GPU Process
  Memory counters, and GPU memory, temperature and power from nvidia-smi.
  It refuses to launch below 10.5 GiB free VRAM and kills the tree on a
  wall, RAM, VRAM, shared-memory or temperature breach.
- `afg03_local.py` runs the BF16 cell under the watchdog and, only if it
  stops on an exact-parity assertion, a separately identified FP32 cell
  in a fresh directory; completed cells get `--audit-report`.
- `afg03_checkpoint.py --audit-report DIR` checks a finished run without
  the weights: adapter, trace and panel digests, receipt hashes,
  development aggregates, CAL selection and timing summaries.

On a full card the GPU is shared through a vramorama lease:

```powershell
$py = 'C:\path\to\cuda-venv\Scripts\python.exe'
python training/aleph_native_floodgates/afg03_panel.py --tokenizer D:\aleph\llama24\tokenizer.json --out D:\aleph\panel
vramorama run --vram 11G --owner afg03-native-floodgates --guard-power 120 --wait 21600 -- `
  $py training/aleph_native_floodgates/afg03_local.py --checkpoint D:\aleph\native\model.safetensors `
  --panel D:\aleph\panel\afg03-panel.json --root D:\aleph\runs\c1 --python $py
```

The CUDA path streams each checkpoint tensor straight to the card
(`safe_open(device="cuda:0")`), so host RAM never holds a second copy of
the 4 GB model. On Windows the runner reports the process peak working
set and peak private commit. A parity failure writes `progress.json` with
the failing prompt or decode receipt and a cached-versus-cold probe
(argmax, top-two margin and largest logit difference per step).
