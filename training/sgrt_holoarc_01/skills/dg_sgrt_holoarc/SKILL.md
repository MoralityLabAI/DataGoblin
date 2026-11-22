---
name: dg-sgrt-holoarc
description: Explore task-conditioned low-rank parameter arcs using bounded holonomy and exact verification.
---

# DG-SGRT-01 | Holonomy exploration skill

## Allowed scope

Work only in MoralityLabAI/DataGoblin. Read the recovered SRT paper, SGRT
protocol, and RSI-Topology holonomy report as historical references.
Keep this experiment isolated under training/sgrt_holoarc_01 and the two
sgrt_holoarc training_gym modules. Preserve the four-operator Aleph design,
all existing weights and local model services.

## First pass

1. Pin the exact DataGoblin source revision and test/symbol versions.
2. Execute the geometry unit tests and the small public smoke only.
3. Check path closure, determinant, orthogonality, gauge covariance and
   dual pairing. Distinguish zero curvature from high curvature.
4. Compare all seven policy arms on identical public tasks; all non-rigid
   arms use equal candidate and verifier budgets.
5. Check the exact-set phase shuffle checksum and canonical solve parity.
6. Report first-hit differences separately from candidate-set coverage.
7. Retain malformed cases and failed verifications, including roots with no
   valid candidate.

## When extending toward a learned sidecar

- Replace fixed Fourier program coordinates with a task-conditioned,
  low-rank learned chart trained only on admitted training roots.
- Keep the connection and rank capacity matched to a nongeometric bridge.
- Condition gear frequency/amplitude/phase only on public task context.
- Pass proposals through the environment-owned verifier, preserving
  pre-verification predictions and post-verification results.
- Freeze a prospective family-level split and minimum gain before sealed
  evaluation. Prevent seed, target and oracle table access in proposer code.
- If holonomy is nonzero but success is unchanged, retain the null.
  If flat and curved candidate sets differ, compare against the stronger
  coverage baseline before attributing any value to path order.
- Promote a single recurrent boundary only after matched tests; do not
  replace an entire Aleph operator during the first pilot.

## Machine and billing boundary

CPU smoke only, no provider keys, hidden weights, unapproved downloads, paid
fallback, new background worker, or GPU use. Do not stop, restart or displace
a running llama.cpp/Inky Mini service. If an authorized later pilot uses
that server, pin the live model revision, quant, tokenizer and sampling config,
reuse the existing endpoint, default to 12k context, hard-cap at 16k,
and log real token and inference costs.

## Evidence contract

A successful experiment provides code revision, protocol revision, root
and family manifests, exact replay, all policy outcomes, verifier calls,
wall time, memory headroom, and a confidence interval over independent
lineages. Report implementation, synthetic mechanism and learned task results
as separate evidence classes. Document every failed gate.
