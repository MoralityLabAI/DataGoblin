# AFG-01: native Aleph Direct A cache and masked-block drafting qualification

Prospective engineering protocol. Source base
[b078ab75274a2f95449b1daf99b80529049663e7](https://github.com/MoralityLabAI/DataGoblin/tree/b078ab75274a2f95449b1daf99b80529049663e7)
from recovery PR #74. Use the original Aleph class and explicit dense /
RMSNorm / interleaved RoPE math from the actual training backend, rather
than converting to an untied Llama layout. The published 10M pilot
checkpoint is engineering evidence only, without heldout language quality.

## Frozen topology

Target visits: [0, 1..10, 1..10, 11, 12, 13].
Draft visits: [0, 1..10, 11, 12, 13].
The target stores 24 distinct visit K/V pairs, each in unexpanded GQA
coordinates. Core visits i and i+10 use the SAME physical weights but
different cached activations. Draft prelude and coda read their matching
target visits (0, 21, 22, 23); draft core reads either visits 1..10 or
11..20. Absolute token positions are preserved. The mask suffix gets a
learned input embedding independent of the target vocabulary. A rank-four
Q/V/output LoRA is attached to each physical core block only. No adapter
is active during target execution. No draft K/V is promoted into the
target cache. This is a native Aleph adaptation of Floodgates, not an
upstream model replication.

FP32 residuals and default FP32 RMSNorm epsilon; interleaved rotary with
base 10000, four KV heads and sixteen query heads in full geometry. K/V
remain unexpanded until attention. Explicit dense compute is selectable
FP32 or BF16 (the latter matches training math), with FP32 weights.
No global precision conversion. One unpadded document/request, batch one,
absolute context at most 2048, greedy decoding only.

Target cache state is bound to a decoder instance, ordered visits and
committed token IDs. Append uses fresh tensors, validates shapes and
prefix length, and never mutates the previous cache. Verifier processes
the last committed anchor plus a proposed token block, accepts matching
prefix and emits the next verifier token subject to output limit. Crop
by the new committed prefix excluding its final anchor, physically
clone retained tensor slices. Reject cross-decoder, wrong-prefix, stale
and misordered caches. Validate exact greedy IDs against ordinary
cached and independent cold native execution. A numerical tolerance
diagnostic does not excuse a wrong greedy token. Finite tested parity is
not proof of bitwise equality at every unseen input or GPU kernel.

## Qualification sequence

1. CPU source fixture uses the actual Aleph class at width 32, vocab 64,
   four query heads, one KV head, core FF 140. Test original 14 physical
   blocks / 24 visits, per-visit shape and storage, native FP32 reference
   logits, chunked append, cache crop, forced mismatches at positions
   0..4, full acceptance, generation cap, stale/foreign cache rejection,
   causal prefix, draft-only adapter isolation. BF16 is separately
   tested and any exact ID mismatch fails that precision cell.
2. Train only the rank-four adapters on short target-generated token
   streams in the reduced-width fixture. Two cache modes, paired starts
   and draws, 48 updates, batch one, 16 distinct training prefixes;
   eight separate dev prefixes and eight generated tokens. Full target
   weights remain frozen and are checked by hash. Record acceptance,
   structural work, exact greedy parity. This is a plumbing fixture
   with randomly initialized target weights, not a competence experiment.
3. On operator-supplied native model.safetensors, require exact byte
   length 3,997,421,432 and SHA-256
   2e40e32da6a6d975fbd7dc7bd56cd440321531e2e755b9bcc215a289c64232a8.
   Construct the exact full geometry on meta, use strict safetensors
   state_dict assignment, load no optimizer state. Require 999,352,320
   unique FP32 parameters and the frozen source/recipe. No unverified
   key aliases, padding or quantization conversion.
   Run bounded prompt-ID panels with cold/native parity and forced
   rejection/acceptance only; no full-size adapter fitting yet. The
   operator must set machine-specific memory/wall/compute caps.
   Real checkpoint absence is a blocker, not a random-weights substitute.

## Work

Report actual logical visits and block-token visits, raw unexpanded
cache tensor bytes, provisional bytes, rollback, exact ID counts,
max/mean logit deviations and timing diagnostics. Full FP32 persistent
KV is 24*2*4*128*4 = 98,304 bytes per committed token per request;
BF16 is half. Draft uses 14 versus 24 applied blocks per slot, while
verifier retains all 24. No latency gain is assumed. No GPU provisioning,
external model download, weight publication, tokenizer changes or corpus
admission. No native full-size acceleration claim until qualification
with real weights succeeds.
