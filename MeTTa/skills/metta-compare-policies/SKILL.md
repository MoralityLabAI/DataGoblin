---
name: metta-compare-policies
description: Design, run, or read out matched Looped Transformer and Inkling comparisons, MeTTa rule ablations, and frozen source-root transfer evaluations.
---

# Compare matched policies

1. State the planned contrast and change one registered factor per arm, or
   declare all four arms of a paired two-factor ablation. Hold model type,
   parameterization, initialization seeds, source-root allocation, optimizer
   updates, and inference caps fixed when isolating rule or data quality.
2. Select on development only. Freeze checkpoints, STOP thresholds, and
   scoring before evaluating fresh frozen roots. Assert that roots and
   derivatives are disjoint across splits and prior holdouts. Keep raw
   frozen traces and checkpoints out of Git.
3. Grade completion and correct STOP with an independent checker. Report
   paired root-level differences and uncertainty, first edits, premature
   stops, unsafe adds, verifier calls, and actual recurrent passes. Report
   cost across all attempts and per success when defined. Distinguish model
   loop passes from controller proposals and verifier invocations.
4. For the existing Inkling variable-loop protocol, use the staged suite,
   capture, and score commands in `docs/inkling_variable_loop_gym.md` from
   the DataGoblin checkout. A dry run checks request preparation. A real
   comparison requires model responses and independently scored outcomes.

Return the contrast, code revision, model and checkpoint identities, seeds,
root counts and hashes, actual training and inference budgets, independent
results, and negative findings. Mark planned, dry-run, and executed arms
explicitly.
