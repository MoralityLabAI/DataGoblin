# MeTTa-AIRIS-01: first paired synthetic intervention study (2026-11-22)

**Code:** DataGoblin PR #29, source branch research/metta-airis-counterfactual-20261122.  
**Reproducible evidence:** GitHub Actions run [44983795848](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44983795848), artifact metta-airis-paired-counterfactual.  
**Receipt SHA-256:** 0403d61df0f344e83865479bf9fe761e7a04348938e2a2b5b68a84b46512ca09.

The result is a bounded **synthetic verifier study**. No native AIRIS or Hyperon model executes; no learned model was trained. Every exported case is marked training_ready=false, and zero new SFT presentations were created.

## Design

The earlier Pattern Miner reads 1,367 discovery roots from the existing synthetic training split. The remaining 593 internal-confirmation roots have their verifier labels masked while the motif policy is chosen. A deterministic, family-balanced subset of 96 confirmation roots is then used for all three policies: 19 each from budget, coverage, dependency and revision families, plus 20 from lineage.

Each arm acts on the same root and selected records, using one admissible public-fact change. The program and exact collection oracle are regenerated after changing the fact. A separate targeted Boolean predicate must agree with the verifier before and after. An identity sham has zero effect. Source-root hashes, per-case world digests and pairwise control identities are retained for a full replay.

## Results

| Arm | Target violation flips | Full violation signature changed | Oracle next edit changed | Oracle optimal set changed |
| --- | ---: | ---: | ---: | ---: |
| Motif-guided | **86/96** | 87/96 | 24/96 | 59/96 |
| Seeded uniform | 80/96 | 81/96 | **28/96** | 46/96 |
| Budget-first fixed heuristic | **96/96** | **96/96** | 21/96 | **72/96** |

The original headline, a +6.25-percentage-point motif advantage against uniform on targeted flips, becomes less persuasive after the fixed-budget ablation. The budget intervention is built to cross the selected-cost constraint. It therefore produces a targeted flip on all 96 cases, offering an easy proxy shortcut. The motif policy also chooses the budget intervention 70 times; uniform selects it 29 times.

The paired target-flip comparison is **14 motif-only wins, 8 uniform-only wins, and 74 ties**, with 27 roots receiving an identical mechanism in both arms. An exploratory exact two-sided paired sign/McNemar test on the 22 discordant target-flip outcomes gives p≈0.286. This sample does not support a reliable superiority claim.

For the more relevant canonical next-edit effect, uniform has 11 unique successes against seven for motif guidance, with 17 shared successes. This metric also does not establish useful training data: it records that the exact Python oracle's answer changed, without evaluating the value of training a student on the perturbation.

## Methodological boundary

- The task is a finite synthetic corpus-selection problem. A changed constraint flag is a causal effect of a deliberate edit **inside that simulated environment**.
- The policy's discovery ranking is train-only. The 96 confirmation cases are internal training-partition diagnostics, and a new steward-held pool is required for independent model generalization.
- Source rights, real MeTTa execution, typed research authorization, and the final ResearchMembrane decision remain separate.
- Raw sources from upstream Goertzel projects are neither imported nor copied.
- The stored per-root output is an audit record, not a released SFT or preference dataset.

## Next registered experiment

Change the primary target to **verified changed canonical next edits and useful counterfactual repairs**, with root-grouped splits and equal-volume data budgets. Require a fixed-budget control, a deterministic rule control, seeded uniform interventions, and a motif-guided policy. Generate new training-root lineages rather than appending extra presentations to the existing roots at the four-view cap. Preserve failures and stop labels.

Only after a fresh steward-held evaluation should an equal-budget recurrent-student experiment test action correctness, invalid admissions, premature stopping, token use, and out-of-distribution transfer. A successful intervention flip is an experiment receipt; actual training utility remains unknown.
