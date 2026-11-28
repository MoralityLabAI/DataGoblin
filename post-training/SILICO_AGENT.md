# Silico auto-research agent handoff

Use this as a task brief after explicit authorization of a campaign. Until then, perform zero-cost local preflight only.

You are the outer research agent for DataGoblin post-training. Work in MoralityLabAI/DataGoblin on a dedicated research branch. Read post-training/README.md, campaigns/PT-01.md, campaigns/35ru.json, the existing direct-A checkpoint receipt and training/gym interfaces. Resolve all pinned revisions. Do not expose secrets, credentials, raw private data or held-out answers.

Your research question is whether independently verified outcome metadata improves research decisions beyond imitation of the same teacher text. Start with a tiny CPU fixture; implement and test a transition adapter, split audit, outcome authority checks, and durable metering/reservation interface. Choose one RLT/Lean environment and one EvalMath research-state environment with exact evaluation. If an upstream interface is unavailable, document the blocker and run only the local fixture; never invent external evidence.

Request public research plans, derivations, predictions and revisions from the authorized teacher. Check applicable model/provider rights first. Preserve executed counterfactuals, negative outcomes, unknowns and the actual evidence. Freeze the reward and compare a common-parent SFT control against an outcome-weighted method, with a shuffled-outcome control when feasible. If the first model is weak, keep actions typed and bounded. Do not treat teacher rollouts as on-policy student data.

For a live 35-RU campaign require a campaign-specific authorization ID, a known provider budget meter, a provable upper bound for open jobs, and an independent dev protocol. If any are absent, stop before dispatch. All paid retries count; separate external API spending requires its own authorization. Record actual debits rather than assumed RU-to-GPU conversions. Disable auto-renew. Save before each paid dispatch, persist job IDs and reconcile after interruptions. Do not open sealed evaluation labels or silently replace the parent checkpoint.

Commit sanitized code, protocol, manifest hashes, aggregate findings, tests and a replay recipe to a branch; store rights-reviewed trajectories and checkpoints in approved persistent storage. Produce a short campaign report with measured cost, parent/candidate lineage, uncertainty, failed attempts, pass/fail gates, and the next proposed capped campaign. Stop after the authorized increment even if there is a promising next experiment.
