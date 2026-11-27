# DG-AGN-TINKER-01

A small controller on the RTX 3050 host calls the Tinker-hosted Inkling 2 Mini model. No local GPU training is required. The Aleph pretrain is outside this experiment. The initial curriculum is an independently measured AGN-02 linear-student fixture; it is not evidence of frontier reasoning or improvement of the local DG checkpoint.

## Offline smoke test (no keys, no charges)

From the repository root:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_agonist_tinker_gym.py
python -m Agonist.tinker_gym fixture --out runs/tinker-fixture --groups 4
```

This constructs distinct root families for the curriculum, routing, failure diagnosis and repair adapters. Curriculum labels are measured by fitting a small student with equal optimizer steps for each possible candidate. The held-out student outcomes never appear in model inputs.

## Live clean campaign

Install the current official Tinker SDK/Cookbook on the controller (SDK 0.38 or newer for future-returning client factories; API checked against 0.40.1). Keep credentials in the environment or an OS-managed secret store. Do not put keys in the repository or command-line arguments. Start the local llama.cpp completion server bound to loopback, with the actual Aleph checkpoint and an adequate context. The client does not assume a chat template or grant tool access.

```powershell
python -m pip install tinker tinker-cookbook
python -m Agonist.tinker_live campaign --live --out runs/inky-clean --round 1 --rounds 3 --groups 16 --steps 2 --batch-size 4 --eval-groups 8 --rollout-groups 4 --curator dg --dg-url http://127.0.0.1:8080 --dg-model-id ALEPH_ACTUAL_CHECKPOINT_SHA --tinker-usd 5
```

Replace the placeholder model identity with the actual checkpoint digest. For the initial provider-only smoke test omit the two DG arguments and set --curator fixed. The amount is an example cumulative Tinker reservation ceiling, not a recommendation to spend or an actual provider balance.

Each round starts with a rollout of the current policy, records failed decisions, curates unique training roots, fits role-specific LoRAs, saves optimizer and sampler snapshots, compares each candidate to the incumbent on a separate development namespace, and rolls out again. Promotion requires a positive development score delta, parse reliability, and zero unauthorized routing releases. These are development gates, not sealed evaluation. The curriculum adapter can be tested as curator with --curator inky. The selection interface asks for one allowed action per slot and stores the full candidate mapping; invalid outputs fall back to failure-first selection with an explicit receipt. Fixed, random, and failure_first selectors are controls. The synthetic student-utility objective remains a proxy, so transfer to live LoRA selection is unproven. Clean observed failures become one independently labeled correction example per role/root; only allowlisted prior action IDs and recomputed failure codes enter its prompt. When DG is running, the specialist proposes and DG reconsiders; both grades and disagreements are retained. Sampling and SFT use effort 0.9; sampling temperature is 1.0. The default 8,192 generation tokens and 24,576 context ceiling can still truncate difficult examples. The default 5e-5 learning rate is provisional; sweep against the Cookbook model-specific get_lr starting point before scaling.

## Fable advisory study

Only run if the operator has verified that the intended advisory use is permitted by provider terms. This path is an isolated inference experiment, with no training/export/admission of Fable outputs or Fable-influenced trajectories. It can read approved clean adapter checkpoint references. The clean lane cannot read the advisory journal. Advisory roots use a separate seed namespace.

```powershell
python -m pip install anthropic
python -m Agonist.tinker_live rollout --live --mode advisory --advisory-authorized --out runs/inky-advisory --registry runs/inky-clean --round 1 --rounds 3 --groups 8 --rollout-groups 4 --attempts 3 --dg-url http://127.0.0.1:8080 --dg-model-id ALEPH_ACTUAL_CHECKPOINT_SHA --tinker-usd 1 --fable-usd 3
```

After repeated unsuccessful attempts, the advisor sees the public problem and prior action/failure codes. Its visible correction goes to DG for the final attempt. No hidden chain of thought is requested or stored. There is no automatic training on Fable advice. A separate journal means separate Tinker reservations; account-level spend controls must cover both lanes.

## Operations

Run one worker per journal. Use an existing supervised Windows scheduled task or service with restart enabled. The launcher above is finite and restartable; restarting identical arguments replays completed receipts without repeating the provider calls. Increase the cumulative caps explicitly only after checking the account. A pending receipt caused by network or process failure stops new calls until manually reconciled with the provider. Do not delete a pending ledger or blindly retry an optimizer call. The STOP file in the run directory prevents subsequent calls; it cannot cancel an already submitted request. Back up SQLite databases using the SQLite backup API or after stopping the worker, including any needed WAL state.

```powershell
python -m Agonist.tinker_live status --out runs/inky-clean
python -m Agonist.tinker_live trace --out runs/inky-clean > runs/inky-clean-traces.jsonl
python -m Agonist.tinker_live trace --mode advisory --out runs/inky-advisory > runs/inky-advisory-traces.jsonl
New-Item runs/inky-clean/STOP -ItemType File
```

Traces retain public task state, attempts, routing decisions, measured regret, snapshot identity, token counts, checkpoint paths and provider receipts. Keep the traces private; the advisory lane is never trainable. Token reservations are conservative estimates at the stated rates, not billing guarantees. Check current provider prices, set provider account caps, and account for storage and provider-side retries. The implementation requests a 45-day checkpoint TTL; archive valued checkpoints separately before expiry.

## First live acceptance

Verify model/renderer and SDK version compatibility, one forward/backward + optimizer step, state restore, sampler inference, local completion, STOP, cost ceiling and a full replay. Use a disposable development run before scheduling unattended. No live credentials are used by the tests.
