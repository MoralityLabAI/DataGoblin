---
name: metta-verify-rules
description: Author, debug, or replay MeTTa facts, typed rules, pattern matches, and native query fixtures against independently specified outcomes.
---

# Verify MeTTa rules

1. Identify the MeTTa engine and version, allowed atom shapes, evaluation space,
   and input facts. Keep fixtures small and reset the space between cases.
2. Specify expected bindings from the public problem before running the rule.
   Include a positive query, a missing-match query, and a guard violation when
   the rule adds a guard. Add a duplicate case when multiplicity matters.
3. Check actual native results against that expectation. Preserve multiplicity
   unless the interface specifies a set, and compare order only if it is part
   of the contract. An empty result stream and a stream containing `()` have
   distinct meanings. Record interpreter errors separately from either.
4. Inspect generated rules before execution. Use a bounded fixture with
   reviewed imports and grounded operations. For a reusable rule, check a
   second fixture that changes the matched bindings or scope.

The existing DataGoblin smoke test replays one train-partition query per
curation family in Hyperon. Its receipt measures native query agreement.

```sh
python -m MeTTa.native_query_replay --roots-per-family 16 --max-train-roots-per-family 1 --queries-per-root 1
```

Return the candidate rule, the query and expected results, then observed
results if available. Include the runtime version, source or fixture
identity, independent checker, and any mismatches.
