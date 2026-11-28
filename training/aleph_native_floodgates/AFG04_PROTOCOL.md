# AFG-04: draft depth, readout and initialization for native drafting (registered, not run)

Prospective follow-up to AFG-03 (RTX 5080, actual native 10M checkpoint;
see the AFG-03 result in the README). The verifier is the same frozen
native target, bit-identical weights and the same 24-visit word; nothing
here trains or edits it. Serial decoding stays an eligible fallback.

## Why the draft depth is the first question

A draft call runs the 14-visit word `0, 1..10, 11, 12, 13` over k+1
positions; a verifier call runs the 24-visit target word over k+1
positions. At batch one on the 5080 each call is bound by reading the
block weights, so wall time tracks visits per call far more than token
positions. One speculative round therefore costs about 14 + 24 = 38
visit-reads and emits 1 + a tokens (a = accepted draft tokens); serial
costs 24 per token. Break-even needs a > 38/24 - 1 = 0.583 accepted
tokens per round before any controller cost, and a full k = 4 acceptance
gives at most 5 x 24 / 38 = 3.16x. A drafter with d visits breaks even at
a > d/24. Shallower drafters lower the bar; the question is how much
acceptance they keep.

## Arms (one matched comparison)

All arms share the AFG-03 token panel or a fresh panel built by the same
builder and split rule, the AFG-03 teacher streams (exact cached/cold
greedy agreement), 64 updates, batch one, the same mask/LoRA seed and the
same (prefix, cut) draw sequence, rank four, CAL-selected k in {1, 2, 4}
and three measured repeats.

Draft schedules (visits read, KV source per visit as in AFG-03):

| arm | draft word | visits |
|---|---|---|
| D14 (AFG-03 control) | 0, 1..10, 11, 12, 13 | 14 |
| D8 | 0, 1..5, 11, 13 | 8 |
| D6 | 0, 1..4, 13 | 6 |
| D2 | 0, 13 | 2 |

Intermediate readout arms: run 0 then the first-pass core to visit j
(j = 5 or 10), then the answer block 13 and the tied head through the
final norm, with first-pass KV. Report each j separately.

Initialization arms (each on D14 and on the best shallow schedule by
CAL):

- Zero-B LoRA with seeded A (AFG-03 control).
- BPD-guided: rank-four Q/V/O factors from the leading Behavioral
  Parameter Decomposition components (MoralityLabAI/BPD) of each shared
  core block, ranked by attributed effect on next-token logits over the
  train teacher streams only, scaled to the control's initial update norm.
- Matched-norm random orthogonal factors (control for "any nonzero start").

Mask-block objectives (each on the best schedule x initialization by CAL):

- Cross-entropy on the k target tokens from mask positions (control).
- Survival-weighted cross-entropy, weights 1, 1/2, 1/4, 1/8 by position.
- Distribution matching: KL to the frozen target's full next-token
  distribution at each teacher-forced position.
- Teacher-forced block: mask position j sees teacher tokens before j
  (parallel teacher forcing) instead of pure masks.

## Measurements and decision rule

Same receipts as AFG-03: actual development tokens/s and speedup over
serial (median of three, all three must agree on IDs and work), accepted
prefix survival by draft position, tokens per verifier call, draft and
verifier block-token visits, peak allocated/reserved VRAM, peak host RAM,
teacher-cache transfer bytes and seconds, training and decoding time,
exact greedy mismatches (must be zero) and the CAL-selected policy.

Primary endpoint: development tokens/s relative to serial at the
CAL-selected length, on the RTX 5080 at the precision that passed AFG-03
qualification. An arm repays its draft only if every one of its three
measured repeats beats the serial median by more than 5 percent. Report
every arm, including the ones that lose to serial and to suffix
repetition. The 10M checkpoint is close to unigram noise; repeat the
comparison on a later, more competent native checkpoint before drawing
conclusions about language drafting.

## Bounds

RTX 5080 only, no paid compute: 10 GiB PyTorch allocator cap, at least
10.5 GiB free VRAM at start under a vramorama lease, 12 GiB process-tree
RAM, 256 MiB host teacher KV bank, two CPU threads, external watchdog for
RAM, VRAM, shared (WDDM) GPU memory, temperature and wall time, 1,800 s
initial wall cap per cell. Keep the checkpoint and panel local.
