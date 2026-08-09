# V10 set-metric robustness preflight

## Outcome

The v10 whole-array metrics now have a frozen, executable development
preflight. An evals-reviewer audit observed all 20 preregistered rows, accepted
all 20, and found two independent probe families in every metric-by-probe-type
cell. Both `exact_status_vector_accuracy` and
`exact_selected_set_accuracy` received `development_pass`.

This is metric-development evidence, not a model result. No v10 sorter was
trained, no architecture was selected, no v9 confirmation label was opened,
and no checkpoint, corpus, deployment, or larger-model promotion is
authorized. The audit's operational decision is `collect_holdout`.

## Frozen metric contract

Both metrics score each array as one binary unit after aligning predictions to
oracle entries by candidate ID:

- exact status-vector accuracy is one only when every aligned candidate status
  is equal;
- exact selected-set accuracy is one only when the candidate IDs whose status
  is `SELECTED` are exactly equal; and
- duplicate, missing, extra, empty, or invalid candidate assignments are
  contract violations rather than partially credited rows.

Candidate order is deliberately a nuisance variable. The selected set is
derived from the same status assignment, so a second unbound selected-set
claim cannot disagree silently with the status vector. Later cross-array
aggregation must average these array-level binary values; per-candidate
accuracy and status macro-F1 remain diagnostics and cannot substitute for the
whole-array gates.

The preregistration was staged through evals-reviewer before executing the
public fixture outcomes. Staging expanded the following matrix:

| Probe type | Independent families | Frozen expectation |
|---|---|---|
| invariance | candidate order; ID alpha-renaming | exact stability |
| sensitivity | remove a selected member; insert a nonmember | 1.0 decrease |
| monotonicity | compound removals; compound insertions | never improve |
| anti-gaming | factorized-majority decoy; duplicate identity | detect both |
| clean control | canonical exact; reordered exact | no violation |

The monotonic probes use a required delta of zero because a binary exact metric
already reaches its floor after one error. They test that compounding errors
cannot improve the metric; they do not claim graded severity resolution.

## Evidence review

The review layers remain separate:

| Layer | Status | Interpretation |
|---|---|---|
| metric robustness | development pass | 20/20 staged rows passed |
| task result | not run | no v10 model or data gym outcome exists |
| measurement reliability | complete public run | coverage 1.0; no invalid, unexpected, or missing rows |
| claim support | development evidence only | local to these public fixtures and metric implementation |
| operational decision | collect holdout | no acceptance or promotion yet |

The ten families are distinct transformation or counterexample families, but
their observations come from the repository's public deterministic fixture
implementation. The independent evals-reviewer checked schema, staging
consistency, thresholds, family counts, and outcomes; it did not independently
author the oracle cases. Consequently, this pass cannot be treated as
environment-owned confirmation under DataGoblin's authority rules.

No SAE or VPD hook was proposed. Mechanistic evidence would be corroborative
only and could not upgrade a missing or failed behavioral probe gate.

## Bound receipts

- frozen robustness specification:
  `aa76ce31a0f32a518f4b33a6d83c76b577401a9ff8761d18c2694a9ebdaa46c0`
- staged probe plan:
  `ecb8ea2e021200b30d90a6754ac08d30c1b455adcbf3c7123e63f32103e3f777`
- staging report:
  `af65a90b93f91f83cdde0dea40267895d4f7096ddb67979076bb21fe7a07d0d7`
- development observations:
  `0d714eb8ca39d070eee9e0e88d97d1c993e95efecb9dc93e4a83d7862254ec51`
- robustness audit JSON:
  `f1ceba9bfd337d630aa9ee676a83972da076dd0caa32c270e76007ea165874c6`
- robustness audit Markdown:
  `a0132c581b92859b32577e06af14d48d95a3fe7f15aa804bcdb544d6fbf4ca49`

Generated observations and review artifacts remain outside the repository;
only the frozen contract, deterministic implementation, tests, and this
receipt summary are committed.

## Next gate

An environment-owned reviewer should author and sequester a fresh holdout probe
pack covering the same five probe types, publish its contract hash, and run the
unchanged metric implementation once. Only a held-out robustness pass should
allow these metrics into a separately frozen v10 training contract.

That later model contract may compare the v9 recurrent mean-pooled head with
edge-conditioned message passing and a pair-reducer control, while testing a
genuinely set-coupled selected-set loss or constrained decoder. Architecture
selection must still use fresh development lineages, exact mean and worst-fold
whole-array gates, and sealed confirmation that remains inaccessible unless a
development primary qualifies.
