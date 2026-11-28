# AFG-03 on the RTX 5080: native 10M checkpoint, Floodgates adapters

Native Direct-A export, 999,352,320 parameters, 14 physical blocks and 24
executed visits (`0, 1..10, 1..10, 11, 12, 13`), loaded strictly from the
pinned FP32 safetensors (3,997,421,432 bytes, SHA-256 `2e40e32d…a8`; 128
keys, all FP32). Device: RTX 5080 Laptop GPU (sm_120, driver 577.05),
torch 2.11.0+cu128, TF32 off, 10 GiB allocator cap, two CPU threads, one
vramorama lease (11 GiB) per run, external watchdog. Panel: 32 train, 16
CAL and 16 development rows from 32 source documents, panel file SHA-256
`3da8c6f5…`, panel hash `848ef37c…` ([inventory](panel_inventory.json)).
This checkpoint is close to unigram noise: on every development prompt
the greedy continuation is one token repeated eight times (`,`, `.`,
` the`, ` of`, newline, `1`; seven distinct tokens). Nothing here is a
language-modelling result.

## Precision cells

| cell | commit | outcome | wall |
|---|---|---|---|
| BF16 dense, FP32 weights (lease 1) | 1cc24e8 | CAL qualification passed (96 exact argmax positions, max abs logit error 0.039); **teacher generation failed exact greedy parity** on a dev prompt (`Solve x^2 - 5x + 6 = `): cached `,` vs cold `1`, cold top-two margin 0.03125 = one BF16 ulp; probe: 1 of 8 steps disagree, 0 tolerance failures | 253 s |
| FP32 (lease 1) | 1cc24e8 | **no result**: the laptop clamped the GPU to a 15 W software limit at 86 C; the lease power guard stopped the cell after 121 s capped; Windows logged a GPU reset (nvlddmkm 153) six seconds later | about 345 s |
| FP32 (lease 2) | 85de7e0 | **complete**, receipt audit passed | 540 s |

FP32 parity (lease 2): CAL qualification 96 exact argmax positions over
12,042,240 compared values, max abs error 1.24e-5; chunked append with
offset RoPE, forced rejection at every draft position (accepted prefix
0–4 with 4–0 positions rolled back), cache rollback, stale, foreign and
misordered caches rejected, the two core passes' K/V physically distinct.
Teacher streams: 512 exact cached/cold positions, max abs error 1.72e-5.
All 96 development rollouts (6 policies x 16 prompts x 3 timed repeats)
equal the independent greedy reference: zero mismatches; repeats agree
in IDs and work. Target weights hash unchanged after training and
decoding; both adapter arms share initialization and (prefix, cut) draws.

## Development decoding (FP32, CAL-selected draft length, 128 tokens)

| policy | k | tok/s (median of 3) | vs serial | tokens per verifier call | accepted / drafted | survival by draft position | verifier visits | draft visits |
|---|---|---|---|---|---|---|---|---|
| serial | – | 11.17 | 1.000x | 1.000 | – | – | 3,072 | 0 |
| repeat | 4 | 23.24 | **2.080x** | 2.723 | 97/157 | 32/47, 32/47, 17/32, 16/31 | 4,896 | 0 |
| first, untrained | 4 | 12.37 | 1.107x | 1.939 | 73/217 | 22/66, 21/60, 20/53, 10/38 | 6,792 | 3,962 |
| deep, untrained | 1 | 11.09 | 0.993x | 1.855 | 60/69 | 60/69 | 3,312 | 1,932 |
| first, trained | 1 | 7.40 | 0.662x | 1.196 | 22/107 | 22/107 | 5,136 | 2,996 |
| deep, trained | 1 | 7.08 | 0.634x | 1.103 | 12/116 | 12/116 | 5,568 | 3,248 |

Timed samples (tok/s): serial 10.76/11.17/11.64, repeat 23.56/22.77/23.24.
Wall time includes 16 prefills (8,376 block-token visits), draft,
verification, rollback and host decisions, synchronized. No timed sample
ran under a power cap (NVML before and after each); GPU 79–81 C,
71–79 W. CAL-selected policy: **repeat at k = 4** (CAL tariff 75.4 per token
vs serial 120). Serial stays eligible.

## Adapters

Rank-four Q/V/O LoRA on the ten shared core blocks plus a mask vector,
432,128 parameters each, 64 AdamW updates (lr 0.01), batch one, 4
teacher tokens per update from 8-token target continuations. Both fits
**diverged**: mean loss over the first/last eight updates rose 4.69 →
9.56 (first-pass KV) and 5.64 → 7.21 (second-pass KV); final 13.18 and
11.28. The registered learning rate came from the width-32 fixture and
is too high at width 2,048. Trained adapters accepted fewer tokens than
their untrained controls (zero-B LoRA, so the untrained drafter is the
frozen model reading a random mask embedding), and lost to serial.

## Cost of the 14-visit drafter (post-hoc)

A least-squares fit of median wall time on calls across the six policies
(not a registered analysis): 1.51 s for the 16 prefills, 78.6 ms per
verifier call, **65.1 ms per draft call**. The draft runs 14 of 24 visits
(58%) yet costs 83% of a verifier call: at batch one the decoder is bound
by per-call overhead (kernel launches; the GPU was 8–18% busy in uncapped
watchdog samples), and the draft adds three LoRA paths to each core
visit. A speculative round therefore needs about 0.83 accepted tokens to
break even, not 0.58. Only the untrained deep drafter (0.87 per round)
reached it. Residuals of the fit are under 0.64 s on runs of 5.5–18 s. This motivates the shallower
schedules registered in [AFG-04](../../AFG04_PROTOCOL.md).

## Memory, transfers and time (FP32, lease 2)

| quantity | value |
|---|---|
| peak allocated / reserved VRAM (PyTorch) | 4,052,477,440 / 4,108,320,768 B |
| per-process dedicated / shared GPU memory (WDDM counters) | 4,369,391,616 / 845,152,256 B |
| peak host working set / private commit (runner) | 5,472,985,088 / 10,144,194,560 B |
| teacher KV bank on host | 95,453,184 B (cap 256 MiB), device-to-host 0.130 s |
| per-fit prefix KV host-to-device | 151,388,160 B, 0.176 s (first), 0.171 s (deep) |
| checkpoint load + hash, CAL qualification | 18.2 s, 7.8 s |
| teacher references (64 prompts, cached and cold), bank | 79.4 s, 2.6 s |
| adapter training | 8.20 s (first), 7.95 s (deep) |
| decoding (median wall, 16 prompts) | serial 11.46 s, repeat 5.51 s, first/deep untrained 10.35/11.54 s, trained 17.30/18.08 s |
| total cell | 535.7 s (watchdog 540.3 s) |

BF16 (lease 1) peaks: 4,237,183,488 B allocated, 4,562,329,600 B
dedicated and 845,152,256 B shared GPU memory, 5,457,752,064 B runner
working set, 83 C, 78 W. (The lease-1 commit message gave the card-wide
4,350 MiB as "4.35 GiB dedicated"; these are the receipt values.)

## Gates and limits

- Failed: BF16 exact greedy parity (teacher stage). FP32 attempt 1
  produced no result (power guard). Both adapter fits diverged; trained
  adapters were slower than serial and than their untrained controls.
- Watchdog violations: none in any cell (wall, 12 GiB tree RAM, 1 GiB
  available floor, 10.5 GiB free VRAM at start, 11.5 GiB dedicated,
  1 GiB shared, 95 C). Free VRAM at start 15,685,648,384 B (14.6 GiB);
  only `dwm.exe` held GPU memory. Minimum system available memory 8.0 GiB (BF16), 10.0 GiB
  (FP32).
- Lease-level events: 15 W power cap at 86 C during FP32 attempt 1,
  guard stop, GPU reset six seconds later.
- Shared GPU memory of 0.79 GiB appears during checkpoint load in every
  cell and stays; dedicated memory never exceeded 4.4 GB of 16 GB.
- The FP32 wall cap was raised from 1,800 s to 3,600 s before the second
  lease (protocol amendment); the cell finished in 540 s, inside the
  initial cap.
- One watchdog sample read 138,567 W from NVML (logged samples peak at
  80.2 W); the watchdog now drops readings above 1 kW.
- On Windows the runner writes the trace with CRLF line endings; the
  committed `development_rollouts.jsonl` is byte-identical to the bound
  `trace_sha256` (`-text` in `.gitattributes`).

## Files

`lease1/`: BF16 failure receipt with parity probe, both watchdog logs,
vramorama log, GPU-reset note. `lease2_fp32/`: report, progress, campaign
(with the receipt audit), development trace, watchdog log and summary,
vramorama log. Prompt token IDs are replaced by SHA-256 and length; the
checkpoint, panel, adapters and unredacted receipts stay local
(`D:\aleph-afg03`). torch was imported from the `bluebeam` venv's
site-packages through a `.pth` in the `dg-rv2-torch` venv.
