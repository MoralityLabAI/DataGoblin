# CORT-01: execution consensus and correlated reasoning errors

Status: public development proxy, exploratory. Reference: [Cortheon v2](https://github.com/ashhart/Cortheon), especially ReasonForge and the AIME ablation. Clean-room implementation; no upstream code copied.

## Hypothesis
A controller that discounts duplicate methods and calibrates stopping on independent evidence can reduce false consensus at matched branch budgets. Correlated task misreadings can defeat even diverse executable methods. Do not treat an executed script or a consensus as proof.

## Design
Generate modular affine recurrence, directed reachability count and subset count tasks. Three algorithmic proposal methods are checked against separate reference specifications. A synthetic observation generator injects shared task misreading, method-correlated errors, individual errors and failures. Correctness labels are environment-owned. Policy receives only the currently acquired candidate prefix (answer, method, role, execution status), never the task truth or latent fault labels. Fit a Laplace-smoothed frequency table on separate development episodes. Stop only when two distinct executed methods agree, the bin has at least 40 observations and estimated accuracy is at least 0.90. This is a tabular proxy, not a trained TRM or RLT.

Compare single, full-budget majority, fixed execution-consensus, full-budget method-weighted vote and calibrated method-weighted early exit. Budgets 1/4/7 are maximum simulated branch acquisitions, not matched physical compute. Record realized calls, correct answers, early stops and false early stops for ordinary and shifted failure rates. Fixed policy is a sequential approximation to the upstream asynchronous race; no claim of direct reproduction. Grouped voting divides each method's vote among its conflicting answers. Role 3 is non-executable. No model, arbitrary generated code, network or checkpoint used.

## Run
From repository root:
```sh
node training/rlt_gym/cort01/run.mjs
```
The runner cross-checks all three algorithms against the reference, checks control invariants and prints the public demonstration metrics. The seed is public and the results are development-only; keep future prospective roots under independent custody. Never promote these simulated accuracies as LLM performance.

## Native follow-up (registered design, not implemented)
Use the same public observable candidate interface with frozen local endpoints. Require an externally isolated verifier service, independent reference grading and per-request server-side counters including cancelled work. Compare vanilla, matched majority, fixed Cortheon-style race, trained TRM selector, RLT repair and BPD/MMS-aware scheduler. Keep method diversity and correlation evidence as explicit features. Add independent same-budget serial controls, full-budget no-early-exit control, calibration split and OOD evaluation. Freeze resource caps, seeds and minimum effect criteria before opening a held-out cohort. Record candidate traces and cost receipts separately from training labels; apply root caps and keep evaluation roots out of exported training data.

Security: use disposable Linux container/VM; do not parse model-supplied expressions with unrestricted eval/SymPy on the host. No arbitrary code is executed by this proxy.
