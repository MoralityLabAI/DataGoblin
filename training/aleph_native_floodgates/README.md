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
