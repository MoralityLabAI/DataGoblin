# Upstream inventory

Inventory date: 2026-07-17. Both sources were inspected read-only. DataGoblin does not import either source at runtime and does not copy their code.

## Small Control Harness

- Candidate: `C:/projects/SmallControlHarness`
- Git root: `C:/projects/SmallControlHarness/storyworld-whitebox-control-starter/storyworld-whitebox-control-starter`
- Branch: `feat/hrm-text-gameplay-qlora`
- Commit: `c938da0f3d4320201d4539ccf3c639891ff5b180`
- Remote: `https://github.com/MoralityLabAI/Control-Harness.git`
- Inspected: `README.md`, `pyproject.toml`, `agent/START_HERE_PROMPT.md`, the external manifest and ingestion scripts, `src/harness/`, `src/evals/`, experiment cards, run manifests, receipts, and reports.
- Reused clean-room concepts: preregistration, manifest-selected ingestion, denylisted secrets/weights/checkpoints/caches, deterministic controls, bounded runs, receipts, claim boundaries, and first-class negative results.
- Adapter: discovers cards, manifests, JSONL, receipts, and entry points; constructs commands but never executes or rewrites the upstream.
- Dependencies: Python `>=3.10`; model/training extras are not DataGoblin dependencies.
- License: no root license grant was found. No source code was copied; conceptual influence is cited in `THIRD_PARTY_NOTICES.md`.

## Hybrid TRM/LDT research gym

- Candidate: `C:/projects/HybridTRMLDT`
- Git root: `C:/projects/HybridTRMLDT/ldt_trm_research_gym_v0_0_2/ldt_trm_research_gym`
- Branch: `master`
- Commit inspected: `978c4929b872edbc3c1e3a5f0201ff89fd1c87fb`
- Current commit at final discovery: `b846d6d641eeb1473418fdda75ae97ca1824a1e3`. The clean checkout advanced during this implementation; DataGoblin did not modify it.
- Remote: `https://github.com/MoralityLabAI/HybridTRMLDT.git`
- Inspected: `README.md`, `pyproject.toml`, `docs/hybrid_contract.md`, typed soundness, hybrid membrane, frames, training review, agent cards, frame generation, and a deterministic benchmark/report path.
- Reused clean-room concepts: public `a_t` versus private `h_t`, typed provenance, claimed-versus-verified disagreement, certified monotone refinement, bottom rejection, soft storage, JSONL frames, and receipt claim boundaries.
- Adapter: maps public research episodes to a compatible-in-spirit frame and maps evidence without promotion; optional upstream import checks are diagnostic only.
- Dependencies: dependency-light Python `>=3.10`; no runtime dependency is carried forward.
- License: no root license grant was found. No source code was copied; conceptual influence is cited in `THIRD_PARTY_NOTICES.md`.

## Exclusions

Adapters cover public artifact discovery, command construction, and typed translation. They do not inherit training code, private evaluator material, weights, checkpoints, storyworld mechanics, provider bindings, or safety claims. Upstream results are not evidence for `causal_switches_v1`, and toy findings are not evidence about HRM, TRM, LDT, or a future 1B specialist.

## Optional HRM-Text milestone target

- Official source: `https://github.com/sapientinc/HRM-Text`.
- Released checkpoint: `https://huggingface.co/sapientinc/HRM-Text-1B`.
- Local read-only generated source cache inspected:
  `D:/hf-cache/hrm-text-1b/modules/transformers_modules/HRM_hyphen_Text_hyphen_1B/92110a91000941b5`.
- Cached source revision directory: `92110a91000941b5`; an older configuration-only revision `47a984ac27f51d76`
  was also present. These are recorded as Hugging Face dynamic-module revision identifiers, not asserted to be Git commits.
- Relevant implementation: `HrmTextModel.forward` calls `L_module` three times followed by `H_module`, repeated across
  two high cycles; `HrmTextAttention` exposes a sigmoid gate derived from the fourth GQKV projection slice.
- Reused concepts: only structural call boundaries, configuration values, and optional Transformers loading conventions.
  DataGoblin's trace protocol, streaming metrics, mock model, calibration, and reports are clean-room implementations.
- Adapter: hooks `L_module` and `H_module` without modifying the source and refuses architectures other than two H cycles
  with three L cycles. Checkpoint-provided Python requires a separate explicit opt-in.
- License: the inspected generated files and official repositories identify Apache-2.0. DataGoblin does not redistribute
  the source or weights.
