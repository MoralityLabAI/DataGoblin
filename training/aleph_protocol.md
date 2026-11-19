# ALEPH-1B training and dyad experiment protocol (proposal)

**Registered status:** PROPOSAL ONLY, written 2026-11-19. The development changes in this document do not change previous experiment registrations or authorize compute. Freeze a new commit, corpus hashes, scoring code, thresholds and job caps before opening a sealed evaluation split.

## Research aim

Produce a checkpoint from 10B processed non-padding training tokens for an approximately 1.007B-parameter four-operator looped model, plus evidence on whether independently measured causal routing targets improve outcome on new task compositions.

Secondary aim: test an optional frozen HRM-Text-1B critic as a budget-aware, dyadic research companion.

## Architecture and comparator

- Two untied input blocks; a bank of four distinct operators, each containing seven tied-across-round blocks; two untied output blocks; tied 65,536 × 1,536 token embedding/readout.
- Operators run from the previous round's snapshot; all four execute. Round one identity communication yields different operator states. From round two, a dense 4×4 router sends learned messages; row-normalized coefficients and a conservative initial gate.
- Use the same task corpus, tokenizer contract, context, step schedule, optimizer and parameter budget across comparison arms where applicable.
- Controls: fixed communication graph, task-trained graph router, causal-guided graph router and shuffled-target graph router. A 3-round branch checks depth transfer once two-round competence is established.
- HRM checkpoint is a separate companion, not folded into the 1.007B trained parameter ledger.

## Mechanistic intervention and training target

1. At registered snapshots freeze a *measurement copy* and select a development calibration cohort.
2. Choose a bounded candidate list of routing alternatives and a fixed set of module/invocation/position sites using VPD/BPD diagnostic recommendations; report source of each candidate.
3. Evaluate candidates and matched random/shuffled controls under identical cap, pass, donor and readout definition.
4. Train auxiliary router targets only from independently scored evidence. Record no-change cases. Causal targets are detached from the main task-gradient update.
5. Report intervention effects on unobserved task families and previously withheld joint interventions. Keep the authoritative scorer separate.
6. Maintain a task-loss-only graph-router comparator on the same data. Distinguish differences due to routed computation, data selection, and auxiliary targets.

In BPD, max pooling across visits, one shared factorization per physical matrix and pass-indexed causal importance are important design constraints. Preserve the known DT04 and Deuterium negative verdicts; mechanisms and gains remain hypotheses until tested.

## Corpus

10B **processed** token exposures, proposed planning lanes: custom research/code material 2.5B, general curated text 3.5B, verified task and code completions 2.0B, procedural task compositions 1.5B, failure-directed examples 0.5B. All amounts are provisional. Source licenses, privacy, canonical deduplication, task-family lineage and benchmark overlap must be audited before the final manifest. Train/validation/test/sealed split by source family or generator program.

## Evidence families

- Strict language and mathematics evaluation; independent code compilation and tests.
- Mathematics/formal proof repair, including Lean-verifiable claims.
- Mechanism discrimination with held-out causal-world families and signed-magnitude intervention controls.
- Research proposals, predicted effects, realized costs, negative results and calibrated abstention.
- Long-horizon memory with later premise revision and task-grounded halting.
- Organelle cold swap with a fixed bridge-only adaptation allowance.
- Dyadic critic success, calibration and GPU-seconds compared to same-cost independent samples.

Reject scoring protocols that credit invalid answer formats as correct. For prefix models, preserve native prompt/response masks. A compiled runner must match eager logits and selected gradients on representative seeds, and must pass a future-token leakage test.

## Planned checkpoints

- 10M tokens engineering: full step, overhead, optimizer resume and accounting measurement.
- 500M tokens: architecture and causal-trace calibration.
- 2B tokens: mechanisms and held-out prediction diagnostics.
- 5B tokens: bounded steering branch versus checkpoint-identical ordinary continuation.
- 8B tokens: held-out generalization, degeneration and graph-exchange review.
- 10B tokens: full checkpoint, data cursor, conversion/quantization compatibility, evaluation and cost ledger.

Preserve each frozen checkpoint as a uniquely hashed artifact and do not retroactively change training token counts.

## Promotion criteria

Before scoring held-out data, preregister minimum absolute task gain over a same-cost task-trained router, calibration threshold and intervention fidelity requirements. Calculate test power on independent roots and meaningful task families. Report confidence intervals, multiple-comparison handling and every failed gate. A learned critic cannot label its own success; the independent verifier remains authoritative.

## Non-claims

Current documents show no trained Aleph weights, no successful HRM dyad, no measured eight-H100 training throughput, no established Silico RU conversion, and no verified recurrence-specific causal steering gain on a new 1B model.
