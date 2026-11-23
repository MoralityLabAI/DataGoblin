# DG-SGRT-01 | HoloArc exploratory recurrent sidecar

**Status:** CPU-only synthetic wiring experiment; no learned model, downstream creative benefit, or confirmatory advantage is claimed. Staged 2026-11-23.

**Repository baseline:** DataGoblin master at a1646c0318df6942db6c58bc78da4084599df7d6.

This isolates a possible **one-operator sidecar** for Data Goblin Aleph's four-operator recurrent graph. Aleph's published plan retains four distinct operators, seven tied blocks per operator, a 4x4 message router, two synchronous rounds, and an independently owned verifier. DG-SGRT-01 changes none of the model weights, router, or pretraining schedule. The optional host adapter would receive a bounded residual after a registered recurrent boundary.

## Original sources and boundaries

- [Spiroscopic Recursive Transformers, August 2026](https://github.com/MoralityLabAI/morality-lab-site/blob/main/papers/Spiroscopic_Recursive_Transformers.md) reported 47/64 versus 35/64 exact synthetic ARC-like solves for learned neighborhood expansion against its strongest matched residual search. Its same-set order shuffle tied at 47/64; the underlying checkpoint and rows are unavailable after the documented filesystem recovery. Do not treat this historical result as a reproduced model finding.
- [Original SGRT appendix](https://github.com/MoralityLabAI/jinn-or-beast-paper/blob/main/experiments/model_selected_scaffold_mesh_v3/appendix_constitutional_ideology_srt/SRT_PROTOCOL.md) distinguished rolling-gear SGRT from the earlier polar-spiral SRT. Its proposed itinerary was G[5:2:0.8]^12.G[7:3:0.65]^12.S.C^6; the registered SGRT arm remained design-bound.
- [RSI-Topology holonomy construction](https://github.com/MoralityLabAI/RSITopology/blob/main/reports/monitor_holonomy_insight.md) demonstrates path-dependent transport in a planted Grassmannian synthetic setting, including determinant and gauge-invariant loop diagnostics. DG-SGRT-01 independently implements small, explicit plane rotations; these sources are references, not model weights or licensed code imports.

## Mechanism

The base coordinate has dimension 12: six independent 2-planes use rational hypotrochoid gears with independently seeded phases, scale and eccentricity. All paths make an exact closed return after eight microsteps. A matched smooth random Fourier path has the same total sampled arc length.

A 16-dimensional fiber carries a feature vector and its dual. Per-step transport is an ordered product of exact plane rotations, with Q = product_i R_{a_i,b_i}(1.6 * delta_u_i). In the flat control, all generators commute. Curved mode uses overlapping rotation planes, so the ordered product over a closed base path may produce H != I. The computed receipts include path closure, orthogonality error, det(H), 1-tr(H)/16, absolute eigenphases, and the maximum absolute drift of c^T v. A constant orthogonal frame change must conjugate H and leave the registered trace and determinant unchanged.

The transport remains a **fixed mathematical construction**. The proposed future learned version would condition amplitudes, gear phases, connection generators, and a low-rank output map on the research task. Its host interface is the bounded residual:

    state' = state + gate * project((H-I) * lift(state))

with gate in [0,1]. A proper trained adapter needs a matched identity bridge, a learned non-geometric bridge, a direct TRM continuation, and gradient/access parity. No existing Aleph block has been replaced.

## Exact offline program-repair gym

The toy world has four-bit states and a two-operation program. Operators include xor, modular addition, bit rotation, bit swap, mask and set-bit. A synthetic hidden program supplies four public input-output demonstrations and a first guess with one altered operator.

Each target-blind policy receives the same first guess, demonstrations, and root identifier, then proposes at most eight distinct, legal one-step rewrites. The exact checker tests all public demonstrations. An evaluator-only full 16-state truth table determines whether the canonical selected survivor is semantically equivalent to the hidden target. Scoring all candidates and choosing a canonical survivor makes equal-set shuffles a strict final-solve negative control, while first-valid visit index still measures ordering.

**Arms:** rigid top-1, fixed semantic heuristic, repeated-edit proxy, flat HoloArc, curved HoloArc, same-set phase shuffle, and matched-length smooth random arc. All arms except rigid top-1 use the same verifier-call budget. The semantic coordinate is a fixed Fourier chart, **not** a learned representation. The two proxy arms are **not** pretrained TRMs.

### Reproduce the public smoke run

    python -m pip install -e ".[dev]"
    python -m pytest tests/unit/test_sgrt_holoarc.py -q
    ruff check src/datagoblin/training_gym/sgrt_holoarc.py src/datagoblin/training_gym/sgrt_holoarc_gym.py tests/unit/test_sgrt_holoarc.py
    python -m datagoblin.training_gym.sgrt_holoarc_gym --seed 23 --roots 8 --budget 8 --out runs/dg-sgrt-01-public-smoke

Output paths must be new or empty. Maximum local limits are 128 roots, 16 candidates per policy, and eight small 16x16 transport steps; there are no LLM requests, downloads, GPU operations, background workers, or paid-provider calls. Files are written only to the requested local output folder.

Outputs:
- public_tasks.jsonl: visible predecision program/demonstration packets.
- evaluator_scores.jsonl: target-bearing evaluator results, proposal hashes, full-domain pass results, visit indices and geometry receipts; custody-limited.
- report.json: root counts, exact solves, oracle-reachable candidates, per-arm verifier calls and hashes of both streams.

The public seed and toy generator test that the code is reproducible. **They are engineering checks and are never a sealed claim.** For real confirmation, generate independent targets and keep the seed key and target tables outside the proposer process and version control.

## Next scientific decision

Follow [PREREG.md](PREREG.md) before a trained-model test. First validate the planted flat/curved loop invariants and the exact checker. Train a small task-conditioned low-rank proposal map only on training roots. Freeze the policy, candidate budget and full program-DSL/generator lineage for withheld families. Report candidate-set coverage, canonical exact solves, full-task generalization, checks per solve, first-valid position, loop diagnostics, root-clustered confidence intervals and actual CPU/GPU time.

An order-sensitive holonomy claim requires an intervention on itinerary order that preserves the candidate multiset, checker access, and evaluation budget. A geometry-specific model claim requires gains beyond a matched learned semantic sampler and a same-capacity nongeometric adapter. Both can fail while the verifier-backed generator remains useful.

The running local Inky Mini endpoint is explicitly outside this CPU pilot. If a later rollout uses it, preserve the existing server, pin revision/quant/tokenizer/sampler settings, and use 12k context by default with 16k hard max.
