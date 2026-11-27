# MM-GYM-03: verifier-budgeted recurrent candidate ranking

Public development pilot. Corpus frozen to the successful [MathCan MM02 artifact](https://github.com/patrickdugan/MathCan/actions/runs/45858715061/artifacts/15136757263), paired with [EvalMath #108](https://github.com/MoralityLabAI/EvalMath/pull/108). Scope: 585 public closed-bank problems, split 186/199/200 by database source order. All oracle labels and hashes are checked before use; only train oracle labels contribute gradients or frequency features. No sealed evaluation, generalization claim, or native Aleph/TRM implementation is implied. DataGoblin adapter is standalone under training, with no change to production routing.

Run `python training/metamath_gym/pilot.py /path/to/unzipped/mm-gym02-corpus --out receipt.json`. Optional `--neural` requires CPU PyTorch; no downloads or paid compute. Tests: `python -m unittest discover -s training/metamath_gym -p 'test_*.py' -v`.

## Fixed neural pilot before execution

A 32-wide candidate encoder with a shared recurrent residual cell reads a bank mean over locally legal candidates. Depths 1, 4, 8, paired initializations 109/113/127, 80 epochs, batch 16, AdamW lr .003, wd .01, no early stopping or selection on challenge. Identical parameter counts and optimizer updates, different recurrent compute. One CPU thread. Loss is logsumexp(all legal scores) minus logsumexp(all fully accepted scores). Each candidate is featurized from its public signature, prefix stack, local instantiated output, target, and train-only label frequency. Local applications are explicitly charged once per bank; no suffix verification during ranking. Report top-k and hypothetical suffix replay calls (oracle simulation with stop-on-hit), not live verifier calls. The frozen corpus oracle has full suffix validation. This is a compact recurrent ranker, not a transformer or native TRM. Recurrent depth effects are exploratory and public; no tuning after viewing the challenge.

## Next BPD gate

Only after a larger dependency-family grouped corpus and a sealed test, fit BPD to the shared cell and register task steering vs matched-rank/norm sham across independent roots; maintain equal local application and suffix-verifier budgets. Retain task-only arm, audit component alignment across passes, and evaluate ordinary unsteered inference. Model promotion requires an incremental benefit over symbolic matching, not merely the frequency baseline. Keep the proof certificate external to the learner.

No new mathematical theorem or ZFC lower bound follows from this exercise.
