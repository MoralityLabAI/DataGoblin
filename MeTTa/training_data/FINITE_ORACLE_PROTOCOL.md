# MeTTa-06: independently computed finite semantics

This experiment uses a new, explicit Python reference evaluator for a narrow
MeTTa subset and compares its computed answers with native Hyperon 0.2.10.
It does not consult Hyperon to produce targets. Its synthetic programs are
public diagnostic examples, and there are no unreleased confirmation labels.

## Scope

The Python reference accepts ground relation facts and queries of the form
`(match &self pattern projection)`. It performs recursive structural binding
with consistency checks for repeated variables. Unbound projection variables
and unsupported source expressions are rejected. Every query result is
compared as an order-independent multiset, with alpha-renaming of variables
inside each result. Multiplicities are retained to avoid concealing native
duplicate branches.

The additional type probe specifies just the grounded `Number + Number`
type rule under `pragma! type-check auto`. It derives `BadArgType` from
the argument's position and inferred ground type. It does not certify the
general MeTTa type system, all type-error atom shapes, or runtime exceptions.

MeTTa supports much more than this subset: rewriting rules, nondeterministic
reduction, spaces, effects, metatypes, grounded operations and imports.
Those require separately defined reference semantics or cross-implementation
differential checks.

## Reproduce

```sh
python -m pip install 'hyperon==0.2.10'
python -m unittest discover -s MeTTa/training_data/tests \
  -p 'test_finite_semantic_oracle.py' -v
python MeTTa/training_data/scripts/finite_oracle_pilot.py \
  --roots 32 --timeout-sec 8 \
  --out runs/metta/finite_oracle/observations.jsonl \
  --report runs/metta/finite_oracle/report.json
```

Each of the 32 deterministic public roots includes 27 distinct ground fact
expressions and four queries. A Python unifier independently computes answers
for empty branches, repeated-variable patterns, nested projections and
nonempty branching results. Hyperon runs each source script in a temporary
directory with a timeout and restricted subprocess environment. The
separately generated type-error probe brings the total to 129 comparisons.

CI reports both agreements and discrepancies. Its passing status certifies
that the tests and receipt pipeline completed; model correctness claims
depend on the observed comparison counts, which are preserved explicitly.
Any mismatch should be examined before deciding whether to extend the
covered subset. A fresh change to the root generator changes source hashes.

## Release and evaluation limits

These tests are deterministic and publicly generated. All observations
retain `training_ready=false`, `sealed_eval_eligible=false` and
`upstream_author_targets_used=false`. Their results cannot be called a
sealed score, a trained recurrent-model result, or an independent full-language
verification.

The upstream author-asserted corpus under `MeTTa/training_data/` and its
187 native-agreed SFT candidates stay unchanged. Support for additional
source constructs will require explicit opt-in with strict rejection on
unknown syntax.

## Next empirical experiments

Grow the independently specified subset through source-level rewrite
semantics, typed error categories and side effects. Compare the same
deterministic tasks in a pinned second runtime such as PeTTa when installation,
version identity and compatibility tests are established. Keep separate
semantic assumptions and receipts for every runtime.

Construct novel source-root confirmation problems in a separate nonpublic
evaluation custodian before running Bonsai or recurrent-model experiments.
A public GitHub Actions run has transparent reproducibility and cannot
serve as an unpublished, contamination-resistant confirmation.
