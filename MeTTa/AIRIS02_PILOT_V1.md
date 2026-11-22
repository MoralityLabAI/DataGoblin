# AIRIS-02 pilot v1: actual trained-model baseline, pre-validity decoder

**Date:** November 22, 2026. **Source:** PR #33, [CPU run 44992144423](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44992144423), artifact airis02-cpu-student-receipts. Research artifact from commit 832beea. No source text or public development teacher labels were uploaded.

All 4 intervention selectors trained one tiny linear predictor and one tiny weight-tied recurrent predictor on the same set of 160 fresh synthetic source roots, with one item per arm/root. Both model classes used 48 gradient updates, identical initialization seeds across intervention arms, and separate source-root splits from the original 0–511 corpus. The recurrent model was trained at four loop iterations and evaluated at 2, 4, and 8.

Independent source-root replay succeeded. The shared development distribution included 60 uniform-intervened public tasks for every trained model.

| Training source arm | Linear exact first edit | Recurrent exact first edit at depth 4 | Full solved + STOP |
| --- | ---: | ---: | ---: |
| Uniform | 28/60 | 13/60 | 0/60 |
| Motif-guided | 30/60 | 13/60 | 0/60 |
| Budget-first | 28/60 | 13/60 | 0/60 |
| MOSES-gated | 31/60 | 13/60 | 0/60 |

**Failure interpretation:** the initial dataset contains nearly all REMOVE targets, with almost no positive STOP examples. Every evaluation episode attempted an inapplicable action and terminated; in the logged pilot, this was 60/60 episodes in each model and arm. The recurrent model returned the same action pattern at all three inference depths. This pilot demonstrates the root/custody pipeline and a low-capacity training run; it provides no evidence for recurrent reasoning, adaptive stopping, or curation-task success.

## Registered corrective experiment

Introduce a structural *membership-only* action-validity mask into training and inference so that ADD can target only absent valid candidate IDs and REMOVE only selected IDs. Keep STOP legal at every state so premature STOP is measured. Give a deterministic, predeclared subset of training roots a verified fixed-point starting selection as their **one** SFT presentation per arm, enabling positive STOP supervision while keeping four total presentations per root. Give the same fraction of public development roots fixed-point prompts, and evaluate every training arm against the same shared uniform-world development allocation.

The changed sample-selection policy must have a new corpus schema and a new CI receipt; this negative v1 result remains frozen. The fresh 768–1023 public test reserve and 1024–1151 steward reserve remain unopened. Neither this pilot nor its corrective experiment establishes performance on an independent steward-held root pool.
