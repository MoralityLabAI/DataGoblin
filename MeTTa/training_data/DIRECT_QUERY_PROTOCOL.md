# MeTTa-05: direct-query runtime replay pilot

The source-conditioned query candidates on `master` were extracted from
passing original Hyperon assertions. MeTTa-05 reconstructs the saved **source
prefix** and runs the candidate's query as a separate top-level MeTTa
evaluation using `hyperon==0.2.10`.

A direct-query observation is useful for debugging assertion semantics,
source-state replay and failed-output alignment. It comes from the **same
interpreter implementation** as the original assertion receipt. Independent
oracle verification is a later gate.

## Command

```sh
python -m pip install 'hyperon==0.2.10'
python MeTTa/training_data/scripts/direct_query_pilot.py \
  --max-packets 24 --timeout-sec 8 \
  --out runs/metta/direct_query_pilot/observations.jsonl \
  --report runs/metta/direct_query_pilot/report.json
```

The dry static tests do not require Hyperon:

```sh
python -m unittest discover -s MeTTa/training_data/tests \
  -p 'test_direct_query_pilot.py' -v
```

The runner first reruns MeTTa-04's pinned-source and source-packet digest
audit. It selects a bounded set round-robin across whole source roots and
reconstructs `source_prefix + "\n!" + query + "\n"`. Every subprocess runs
in a temporary directory with an allowlisted environment, a timeout, capped
source bytes and stdout size. The process exit, duration, hashes and output
status are recorded, including timeouts and parse errors.

A `direct_query_observed` status requires a zero exit, empty stderr, and
one output line per preceding top-level source evaluation plus one for the
new query. This check tests output alignment. It does not assert that the
new output equals the author's expected answer. Query expressions have
different result semantics under `assertEqual` and
`assertEqualToResult`, so string comparison alone would create false
verdicts.

All rows carry `semantic_agreement_verified=false`,
`independent_semantic_oracle=false`, `sealed_eval_eligible=false` and
`training_ready=false`. The workflow uploads observations as a GitHub
Actions artifact; it does not change or automatically promote the committed
candidate dataset.

## Subsequent verification

Use a separately specified evaluator or canonical semantic reference for
MeTTa expressions, including nondeterministic result ordering, alpha
equivalence, and side-effect behavior. Run the same case in both reference
and candidate interpreters, record mismatches as reproducible negative
examples and keep all derived repair traces associated with source-root and
runtime identifiers.

A genuine sealed eval must be constructed from new source roots whose
answers have not appeared in the public corpus or any learner context.
