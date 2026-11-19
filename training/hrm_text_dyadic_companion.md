# HRM-Text-1B as a dyadic research companion to Aleph 1B

*Proposal, 19 November 2026. No paired-model benchmark has been executed.*

## Thesis

Use **Aleph** as the local research operator and **HRM-Text-1B** as an independent, intermittently invoked mathematical and reasoning critic. The pairing creates a controlled cognitive workgroup whose value is measured by independently verified research outcomes per unit of inference and training cost. Leave the weights and tokenizers distinct.

HRM-Text is trained with its released H/L recurrence and PrefixLM conventions. Aleph is the proposed learned graph of four recurrent operators with a text/research control interface. Mixing their internal states or transferring mechanistic features is a later hypothesis, subject to faithfulness tests.

## Roles and interface

1. Aleph receives a typed learner snapshot, proposes hypotheses, selects a data recipe and predicts outcome/cost.
2. The **dyad gate** examines uncertainty and expected value of further computation. It can dispatch a concise, self-contained question to HRM-Text.
3. HRM-Text returns a structured critique with an answer, confidence, assumptions, error modes and potential falsifier. The critic is advisory; its response is saved verbatim for audit.
4. Aleph revises its proposal or abstains. A trusted builder may prepare code or a test fixture within the bounded experiment contract.
5. Independent verifier checks actual task outcomes. Its receipt, rather than either model's claim, enters the training ledger.

Suggested typed envelope:

```json
{
  "schema_version": "dyad-v0",
  "episode_id": "...",
  "question": {"task_family": "math|code|mechanism|curation", "prompt": "..."},
  "aleph_proposal": {"answer": "...", "confidence": 0.0, "assumptions": []},
  "dispatch": {"critic": "hrm-text-1b", "reason": "uncertainty|predicted_value", "max_new_tokens": 256},
  "critic": {"answer": "...", "confidence": 0.0, "counterexample": null, "assumptions": []},
  "decision": {"answer": "...", "accept_critic": false, "needs_verification": true},
  "independent_receipt_sha256": null
}
```

The exact vocabulary and decision grammar must be frozen before the held-out study. Use the HRM checkpoint's native tokens, prefix masks and model revision; record parse failure as failure.

## Evaluation design

Compare five arms using identical episodes and fixed external tool access: Aleph alone; HRM-Text alone; Aleph with an HRM critic called on every case; Aleph with a learned budget-aware call gate; and Aleph with a shuffled, same-length critic response. Include a matched-cost Aleph self-consistency control to test whether any gain comes from extra inference rather than complementary reasoning. For code tasks, record actual compiler/test outcomes; for mathematics, use Lean or exact checking where available. A stronger external frontier critic can be included as an upper bound with its real token price.

Primary outcome: verified whole-task success *subject to a capped compute budget*. Secondary outcomes: calibration, false acceptance, critique usefulness, new-gym validity, cold-start transfer, additional GPU-seconds, invoked-token count, and critical abstention. Evaluate on held-out generator families, not merely fresh seeds from the same templates.

**Pass criterion (proposed):** the gated dyad improves verified task success by a preregistered absolute margin over the best same-cost control, with independent-root confidence intervals and a separately reported excess-cost curve. Estimate the margin from development power analysis before opening a sealed test.

## Training stages

- Train Aleph toward the 10B-token objective, retaining a fixed research-episode format and authoritative receipt provenance.
- Calibrate a small call gate on development outcomes. Its input may use uncertainty, task family, forecast cost and disagreement from *prior* observations. Decisions must rely only on information available before the critic is invoked.
- Add critic-distillation examples only from independently verified decisions. Keep a record of the original unmodified HRM response and both models' proposed revisions.
- Try internal-state bridging only after text-level transfer has a measured gain and a faithful recurrent trace implementation.

## Cost and failure modes

The released `sapientinc/HRM-Text-1B` checkpoint is approximately 1.18B parameters. Pairing with a proposed 1.007B Aleph model yields roughly 2.19B unique model weights loaded across two independent networks. Calling HRM selectively matters more for cost than parameter accounting alone. BF16 storage of the weights is roughly 4.4 GB total before KV cache, transient activations, framework overhead and simultaneous scheduling; measure actual local memory.

Known hazards: critic agreement without correctness, answer leakage through shared histories, teacher-contaminated evaluation labels, prompt/tokenizer mismatch, and recurring false confidence. Preserve an independent verifier and use cost-matched random/unchanged-response controls.

## Implementation pointers

- [DataGoblin HRM cycle tracing](../src/datagoblin/hrm_text/instrumented_forward.py)
- [Research orchestration](../src/datagoblin/research/orchestrator.py)
- [Training gym](../docs/training_gym.md)
- [HybridTRMLDT typed verifier](https://github.com/MoralityLabAI/HybridTRMLDT/blob/2b989d9400cf2d2f7642e3ad31f774d3bb5ee211/research_gym/core/hybrid.py)

This is a proposed research contract. Current DataGoblin HRM tracing and scoring need real-checkpoint fidelity checks before any claim about recurrent mechanism, advice quality or dyadic benefit.
