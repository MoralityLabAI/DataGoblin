# RLT-HARD-02F: real Bonsai-8B Q1 proposal baseline with Lean kernel checking

Date: 2026-11-22. Development registration and first live CPU-inference pilot.

This experiment continues [DataGoblin PR #35](https://github.com/MoralityLabAI/DataGoblin/pull/35) on [PR #41](https://github.com/MoralityLabAI/DataGoblin/pull/41), addressing [issue #24](https://github.com/MoralityLabAI/DataGoblin/issues/24). It ports the existing 1-bit Bonsai tactic-selection adapter from [PR #3](https://github.com/MoralityLabAI/DataGoblin/pull/3), preserving its HTTP and subprocess interfaces.

## Exact source/model pins

| Item | Pinned input |
| --- | --- |
| Bonsai-8B GGUF | `prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf` |
| Model SHA-256 | `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54` |
| OxiBonsai source | `cool-japan/oxibonsai` commit `1d10c48e14754ff6999158f8fa13c7c92b40b0dc` |
| Tokenizer | `Qwen/Qwen3-8B/tokenizer.json` with computed SHA-256 retained |
| Mathlib source | `leanprover-community/mathlib4` commit `809c3fb3b5c8f5d7dace56e200b426187516535a` |
| Lean toolchain | Lean 4.12.0 |
| List generated source | RLT-HARD-02E seed `2026112247` |
| Development pilot | First four OOD roots, maximum two kernel attempts per policy |

The pinned Bonsai GGUF is a general-purpose 8.19B-parameter Qwen3-family model, with grouped 1-bit weights and about 1.16 GB of file bytes. It is **not** a proven Lean-specialist checkpoint. The model hash comes from PrismML's published Hugging Face GGUF file metadata. The 1-bit backend is built from the exact commit above; the produced binary and generated Cargo.lock are SHA-256 recorded in the CI provenance archive. This distinguishes runtime source pins from possible registry-dependency drift.

## Live inference and custody

The model is started on a single `127.0.0.1` HTTP endpoint through `oxibonsai serve` with an explicitly SHA-verified checkpoint and tokenizer path. The runner preflights the executable version and local hashes, probes `/v1/models`, and records its self-reported model ID. **The API endpoint by itself cannot cryptographically attest its loaded checkpoint**; the GitHub Actions script is the documented operator-owned source of its server startup command. The separate subprocess transport passes the exact checkpoint path directly and offers a stronger process-level binding, with cold-load cost charged on each invocation.

The public prompt contains the generated theorem statement, previously rejected tactic choices, and a menu of five candidate proof scripts. Menu index `1` means STOP, while indices `2..6` correspond to the original fixed candidate IDs `0..4`. Any malformed or out-of-menu response is recorded separately from an explicit STOP; an invalid model response cannot be counted as a verified success. No test-only witness flags, kernel gold labels or sealed data appear in the model's prompt.

Each selected tactic is verified through DataGoblin's `LeanKernelChecker`, which compiles source with pinned `lake env lean` and checks that all allowed axioms belong to `propext`, `Quot.sound` or `Classical.choice`. Native goal states come from actual `trace_state` output. All five possible tactics in each root are independently compiled and labeled before model evaluation, with labels held strictly in an evaluator ledger. The runner rejects a discrepancy between a model-selected proof receipt and the evaluator's independent receipt, then recompiles every selected Bonsai action once more. Certified proof sources, source hashes, candidate receipts and model response/request digests are retained; model bytes and raw prompts are excluded from the artifact.

The first live pilot exercises four roots from the frozen List OOD source pool. Three have an intended witness within the five-action menu; the fourth has no matching tactic. The native Lean verifier must validate those labels. The pipeline records model wall time and native verifier wall time independently, including invalid-choice rate and exact model inference call counts.

## Matched baselines

All policies share the same maximum two candidate Lean checks per root and the same generated theorem and candidate menu. The comparator policies are fixed menu order, deterministic pseudorandom full search and the **strong source-visible suffix-length matching rule** from HARD-02E. This last control can choose the matching arithmetic congruence immediately, or STOP with zero verification attempts if no matching candidate is present. A Bonsai speedup over random is insufficient for a model-value claim if this cheap symbolic rule performs as well or better.

The Bonsai arm may make multiple model calls when a selected tactic fails and it receives the actual Lean error status. Verification calls and model generation calls are counted distinctly. Any held-out training artifact is prohibited: this is an inference-only development evaluation.

## Independent evidence audit

`scripts/audit_rlt_hard02f.py` checks the exact public model SHA, pinned Mathlib/OxiBonsai source IDs, tokenizer and binary hashes, the full native five-action verifier ledger, individual accepted proof axioms, per-arm source/candidate alignment, model response/request hashes and false STOPs versus fixed full search. The audit runs over retained artifact bytes and makes no native verifier/model calls. Refusal to parse an unsafe ZIP, changed source, invalid checkpoint identity or forged kernel proof is tested separately.

## Lean/OxiLean compatibility boundary

OxiLean's independent export verifier is separately pinned to Lean `v4.32.0-rc1` and `lean4export` NDJSON v3.1.0. This experiment is pinned to Lean 4.12.0, so it **does not claim** that the OxiLean checker independently verifies its proof sources. In a later optional secondary proof-audit, export the same theorem declarations using an explicit compatibility-matched Lean toolchain, then require separate counts of `verified`, `unsupported` and `rejected`. Treat `unsupported > 0` as incomplete coverage even when the OxiLean process exits zero.

## Limits

These are generated List length/congruence tasks, with a fixed candidate menu and public goal statements. Bonsai-8B inference here is a model-proposer baseline. Hard unseen Mathlib/LeanGym multi-step proof discovery and target-blinded MCP retrieval remain open. Sealed confirmation is untouched.

## Reproduction

Use [the native Q1 smoke workflow](https://github.com/MoralityLabAI/DataGoblin/actions/workflows/rlt-hard02f-q1-live.yml). It builds OxiBonsai from source, downloads and checks the public GGUF, fetches the Qwen3 tokenizer, pins Lean and Mathlib, launches a resident CPU model, and runs:

```bash
python -m datagoblin.training_gym.rlt_gym.hard02f_oxibonsai \
  --lake "$HOME/.elan/bin/lake" --mathlib /tmp/dg-rlt-mathlib \
  --binary /tmp/oxibonsai/target/release/oxibonsai \
  --checkpoint /tmp/bonsai-q1/Bonsai-8B.gguf \
  --tokenizer /tmp/bonsai-q1/tokenizer.json \
  --transport http --port 18783 --timeout 240 \
  --source-seed 2026112247 --split ood_test \
  --roots 4 --attempts 2 --out runs/rlt-hard02f-q1-native-pilot
```

Native metrics are withheld until the real-checkpoint workflow completes and passes the independent artifact audit.
