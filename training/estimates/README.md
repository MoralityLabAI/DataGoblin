# Aleph 1B: 10B-token and 700 H100 GPU-hour planning envelope

**Date:** 2026-11-19. **Status:** analytic, not a quoted price or measured run. The Goodfire/Silico grant is separate from Spiral.

## Proposed target

- 1,007,141,013 unique neural parameters in the specified four-operator architecture.
- 10,000,000,000 processed, non-padding token exposures. Count repeated token presentation; also track distinct source tokens.
- 8 H100 SXM GPUs as a benchmark target; DDP and 4-operator × 2 data replica layouts both require real throughput tests.
- 2 recurrent rounds as default, 3 as alternative.
- **700 aggregate H100 GPU-hours** as a scenario. This is 87.5 hours with all 8 GPUs allocated continuously.
- **Silico conversion unknown.** "$1,560 of remaining Silico RUs" is user-reported balance, not evidence of H100 entitlement or applicable dollar-per-hour pricing.

## Reproducible calculator

Run `python training/estimates/aleph_compute.py` from repository root.

The default estimates use 10B tokens, context 4096, 50% prompt-prefix fraction, H100 SXM dense BF16 peak ≈989.4 TFLOP/s, 30% active model FLOP utilization, and 25% additional runtime. They approximate matrix and attention training work and are sensitive to actual kernel schedules, topology, effective context, padding, and measurement overhead.

| Model | Estimated FLOPs | GPU-hours at 30% MFU incl. allowance | 8-GPU wall time | At illustrative $4/GPU-hour |
| --- | ---: | ---: | ---: | ---: |
| Aleph, 2 rounds | 1.3627293696e20 | 159.41 | 19.93 h | $637.65 |
| Aleph, 3 rounds | 1.9704840192e20 | 230.51 | 28.81 h | $922.03 |

MFU sensitivity, 4096-token context, same runtime allowance:

| Assumed active MFU | 2-round GPU-hours | 3-round GPU-hours |
| --- | ---: | ---: |
| 10% | 478.24 | 691.53 |
| 20% | 239.12 | 345.76 |
| 30% | 159.41 | 230.51 |
| 45% | 106.28 | 153.67 |

These estimates cover one training campaign and no independent full-scale control, VPD/BPD analysis, idle queue time or agent-token debit. A separate full-scale control doubles the relevant training compute. A 10B-token target with **400 GPU-hours reserved for training** requires 25M tokens per GPU-hour, about 6,944 tokens per GPU-second equivalent. If all 700 hours were assigned to training, it requires about 14.29M tokens per GPU-hour.

## Example cap allocation

| Category | Reserved GPU-hours |
| --- | ---: |
| Architecture/compilation preflight | 35 |
| 10B-token main training | 400 |
| Independent mechanistic analysis and steering controls | 150 |
| Final evaluation/export | 45 |
| Recovery and stalled/failed jobs | 70 |
| **Total** | **700** |

This allocation is a spend policy to test, not an authorization. A multi-hour held allocation is charged even if a model is paused for user review. Export model and optimizer states, release GPUs, then restart only with explicit, auditable resume receipts.

## Measure before scaling

Use an actual approximately 10M-token timed training smoke with the chosen attention backend, gradient horizon and distributed layout. Record (i) paid GPU occupancy, (ii) exact processed tokens, (iii) tokens/sec, (iv) memory, (v) network/collectives, (vi) checkpoint time, (vii) Fable/agent-token debit, (viii) account RU change, and (ix) uninterrupted and resumed optimizer parity.

Define all step limits in *processed token exposures*. Store consumed cost separately from committed evidence. Each mechanistic probe receives an independent hour cap. Benchmark training both with and without probe hooks at identical global batch.

## Preserved source estimates

The [2026-11-19 research bundle](../SOURCES.md) holds the original `aleph_compute.py`, full case JSON, exact parameter ledger, 48-paper catalog and original experiment specification. The script here is a compact reimplementation with explicit 700h accounting. Original file checksums are recorded in `../SOURCES.md`.
