# DG-COTP-02: Research Policy Foundry

Pilot scaffold, 2026-11-28. Research-only custody. No live teacher runs or new scientific results are included in this branch.

The source paper is [CoT-Proxy](https://arxiv.org/abs/2611.33148). The pilot preserves observable decisions, compact public rationales, predictions frozen before feedback, and revised research prints. It adapts the research process to [EvalMath](https://github.com/MoralityLabAI/EvalMath).

## First wave: 12 high-difficulty roots

The frontier manifest references nine exact EvalMath Git blobs, spanning adaptive disclosure, post-selection risk, full-filtration validity, correlated proof fidelity, BPD intervention design, FRC, covert feedback, acquisition order, and prospective critique. Seven proposed faculties are represented. These are discovery tasks, not sealed evaluation roots.

Four bounded numerical fixture families supply exact observations: transcript selection risk, cost-constrained adaptive interval queries, sharp triple false-acceptance intersections, and routed probe separation. The risk fixture uses a second exhaustive oracle; the joint fixture constructs an attaining law. Adaptive and cover checks currently have one implementation each. These finite observations do not verify the novel research claims in the prints.

## Offline preflight

From the DataGoblin root (Python 3.10+; standard library only):

```sh
python -m unittest discover -s research_gym/dg_cotp_02 -p "test_*.py"
python research_gym/dg_cotp_02/gym.py prepare --out runs/dg-cotp-02-mock
python research_gym/dg_cotp_02/gym.py mock --out runs/dg-cotp-02-mock
```

The mock deliberately makes incorrect predictions and exercises the feedback transition. It is not a teacher performance experiment.

## Paid pilot: two dependent batches

Use a full local EvalMath checkout containing the pinned blobs. The following example assumes a sibling checkout named EvalMath. Inspect the generated source packets before authorizing transmission; they contain up to 18,000 characters from public pinned sources.

```sh
python -m pip install -U anthropic
python research_gym/dg_cotp_02/gym.py prepare --evalmath ../EvalMath --out runs/dg-cotp-02-pilot
```

Load ANTHROPIC_API_KEY from the existing local secret store into the process environment. Never print or commit the key. Verify Fable access, current pricing, promotional eligibility for batches and remaining promotional balance. Configure a dedicated provider-side spend limit before executing. The example cap is $10 TOTAL for both waves and must be reduced if remaining eligible credits are lower. The reserve is an estimate, not a provider-side guarantee. Keep the run directory private; no Fable print belongs in the public repository by default.

```sh
python research_gym/dg_cotp_02/fable_batch.py submit --wave 1 --out runs/dg-cotp-02-pilot --budget-usd 10 --confirm-credit-and-limit
python research_gym/dg_cotp_02/fable_batch.py collect --wave 1 --out runs/dg-cotp-02-pilot
```

Repeat collect after the provider reports the batch ended; it performs one status check and returns. Review the receipt, initial public proposals, and independent observations. If the first wave has unexpected charges, ambiguous submission or missing results, reconcile with the provider and do not continue automatically.

```sh
python research_gym/dg_cotp_02/fable_batch.py submit --wave 2 --out runs/dg-cotp-02-pilot --budget-usd 10 --confirm-credit-and-limit
python research_gym/dg_cotp_02/fable_batch.py collect --wave 2 --out runs/dg-cotp-02-pilot
```

Repeat collect until the provider reports the batch ended. Outputs: wave1/<root>/initial.json, observation.json, response.json; wave2/<root>/research_print.md and research_print.json; per-wave accounting and receipts. Incomplete or malformed responses stay in the audit trail and are excluded from the continuation wave. The program never submits a duplicate reservation automatically, including after an uncertain transport failure.

Model: claude-fable-5-2; high effort for first wave, xhigh for the second. Max billed output: 8,192 and 16,384 tokens per request. Rates used: batch $4/M input and $20/M output, checked against [Fable 5.2 pricing](https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2) on 2026-11-28. No caching, server tools or retries requested. Estimates use provider token counts plus a conservative input buffer and max billed output. 5% of the explicit cap remains uncommitted. Reconcile with provider invoices and use an independent account spending limit.

## Review and scale criteria

Review correctness, originality relative to pinned sources, experimental discriminating power, calibration, proof scope, and reproducibility separately from the exact fixture grade. Source hashes attest bytes, not truth. The output is a candidate print; novel claims need independent mathematical review or additional verification.

Next implementation: expand to 300 distinct research roots from the wider repositories, wire native EvalMath verifiers and Lean workers, build fresh family-disjoint sealed confirmation, evaluate counterfactual continuation pairs, compare model effort and cost, and integrate DataGoblin ResearchMembrane. Only after a reviewed pilot should a larger promotional campaign be authorized.

Fable outputs remain research-only. Training/export permissions must be established under the applicable terms; no SFT exporter is provided here. An independently authorized teacher can later support the CoT-Proxy arm comparison.
