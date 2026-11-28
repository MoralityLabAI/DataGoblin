# AFG-03: checkpoint-bound native drafting and RTX 5080 qualification

Prospective engineering experiment following AFG-02
([CPU evidence](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46104930110)):
both trained reduced-width targets passed the CAL competence gate,
but the 14-visit neural drafters were costlier than serial; the
cheap suffix-repetition control was faster. Preserve those outcomes.

## Operator inputs and bounds

Use the exact native 10M FP32 export pinned in native_cache.py, not
a GGUF, converted untied Llama checkpoint or guessed tensor mapping.
No full-size model download, data retrieval, GPU provision or paid
backend call. Require explicit --execute, fresh local output directory,
and --data-sha256 for an operator-provided JSON document:

    {"schema":"aleph.afg03.token_panel.v1",
     "tokenizer_sha256":"9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
     "rows":[{"split":"train","group":"source-document-1",
              "ids":[...]}]}

Use 8–32 train, 4–16 CAL, 4–16 development roots; each prompt 8–32
already-tokenized IDs, all strictly in the frozen vocabulary. Each
group belongs to one split; reject duplicate token prefixes and any
cross-split exact prefix containment. Input hash and group/row inventory
bind the run. Content similarity beyond these checks remains the
operator's responsibility; no claim of sealed or globally unseen data.
No target answer labels are supplied. Keep private tokens/artifacts local.

Generate 8–16 tokens per prompt, default 8. Train rank-four Q/V/O
core adapters for 64 updates each by default (1–256), batch one, same
initial mask and LoRA seeds and cuts. Train only on target-generated
continuations. No full-size target fitting. Exact teacher streams
require cached/cold argmax agreement. Prefill their complete cache once;
keep compact unexpanded K/V on CPU under a 256 MiB tensor budget.
For each update, gather only the selected prefix slice, preserving
logical visit and position identity; do not retain target gradients.
The actual target and adapter state identities, teacher/cache work,
transfer bytes, batch choices, peak memory and optimizer work are metered.

## Hardware qualification

CUDA BF16 dense math with FP32 frozen weights by default; FP32 option.
Disable TF32. Require one selected CUDA device and BF16 support when
requested. Bound PyTorch caching allocator by explicit --allocator-gib
(default 10 GiB) and check available memory with extra headroom before
model transfer. This is an allocator cap, not total GPU memory/WDDM
or a process-tree RSS hard limit. The local operator sets an external
RAM/wall watchdog. Report device name, driver/runtime where available,
allocated and reserved peaks, and host RSS where available. Default
wall limit 1800 seconds, checked between bounded operations; external
watchdog is necessary for a blocked kernel. CPU mode is an explicit
alternative, not a silent fallback.

First qualify two CAL prompts using AFG-01 independent original cold
forward, chunked append, forced rejection/acceptance, stale cache and
rollback checks at chosen precision. Any failure stops before adapter
training. Generate independent serial references; exact output parity
is mandatory for every policy, including every timing repeat. BF16
may fail because different GEMM shapes change argmax; record failure,
do not widen token identity tolerance or silently switch precision.

## Comparison

Train first/deep rank-four core adapters separately, frozen target.
Compare serial, committed suffix repetition, first/deep untrained
and first/deep trained adapters, CAL-selected block length 1/2/4.
Serial stays eligible in the final policy menu. CAL chooses minimum
4*serialized calls + token-block visits, ties by action order.
Report all development policies plus the CAL-selected choice. Actual
wall/CPU time includes prefill and draft/verify/cache rollback/host
decision work; CUDA synchronizes immediately before and after measured
work. One untimed warmup per policy using CAL only, three measured
repeats per development panel (configurable 1–3), identical output
and work required. Reference construction and adapter fitting excluded
from decode timing and separately charged. Actual latency can differ
from the structural proxy; no speedup claim until hardware evidence.
Compare both time and serialized/parallel work, acceptance survival,
peak provisional KV, and controller/host costs. No EOS early stop:
bounded fixed-token engineering benchmark only.

## Amendment before the first RTX 5080 run (2026-11-28)

Recorded before any actual-checkpoint execution; gates are unchanged.
Panel: 32 train, 16 CAL, 16 development rows from 32 source documents
(eight each of general prose, Python code, worked mathematics and
structured protocol text), both windows of a document in one split.
The CUDA path loads each checkpoint tensor directly to the device; the
strict key, shape, dtype, size and SHA checks are the same. Windows runs
report peak working set and peak private commit. A parity or greedy
failure keeps its receipt (failing prompt or decode trace, plus a
cached-versus-cold probe of argmax, top-two margin and logit difference
per step); the probe is diagnostic and never relaxes a gate. Execution
runs under an external watchdog (1,800 s wall, 12 GiB process-tree RAM,
1 GiB system-available floor, 10.5 GiB free VRAM at start, 11.5 GiB
per-process dedicated and 1 GiB shared GPU memory, 95 C) inside an 11 GiB
vramorama lease; the runner deadline is 1,740 s so it stops before the
watchdog. FP32 runs only after a BF16 exact-parity failure, in its own
directory and watchdog, as a separately identified precision cell.

Save all local model-bound adapter artifacts, token-panel hash, source
manifest, device identity, qualification, fit and evaluation receipts,
exact decoded IDs, timing samples, and explicit incomplete/blocked
status on failure. No training_ready records, no external transmission,
no claim of competence for the actual 10M checkpoint. CPU CI uses a
small source fixture and the identical adapter/benchmark code, never
publishes actual weights or opens an accelerator.
