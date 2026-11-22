# BSAR-03: matched source-claim pairs

BSAR-02's held-out Step 7 corpus is vulnerable to template memorization: a train-only question lookup obtains 164/164 correct without inspecting the source signature. This follow-up uses the same [MathCan source catalogue](../bsar02/six_round_catalog.json) while making the truth value vary **within the same theorem and question form**.

## Construction

Every retained theorem root receives a paired question about its declared name: one matching name and one decoy. The generator adds matched named-premise questions where the statement contains a named hypothesis, then matched parameter-type questions when a supported bound type is present. Each decoy is verified absent from the exact theorem signature. The pairing preserves the split ownership of the source family: Steps 2-5 train, Step 6 development, Step 7 test. Repeated helper theorem names are assigned to the latest split, inherited from BSAR-02.

The two target labels, `PRESENT` and `ABSENT`, are exactly balanced within every pair. Scoring counts both single-example accuracy and complete paired correctness. A constant question-template classifier therefore reaches **50% example accuracy and 0 complete pairs** by construction.

```sh
python -m training.rlt_gym.bsar03.matched_claims export --output-dir /tmp/bsar03
python -m training.rlt_gym.bsar03.matched_claims constant-baseline --output-dir /tmp/bsar03
python -m pytest tests/unit/test_rlt_gym_bsar03_matched.py -q
```

The public JSONL contains only question IDs, signatures, candidate claims, and provenance, without target labels. Predictions use the shape `{"id": "...", "label": "PRESENT"}` or `{"id": "...", "label": "ABSENT"}`; a model integration should preserve the pairing and compare the two answers together.

## Validation boundary

This gym tests **exact source-grounding** of Lean theorem declarations: identifying the named theorem, reading declared hypothesis names, and recognizing explicitly bound parameter types. A model can succeed through careful source parsing. These are useful precursor tasks for DataGoblin curation and lightweight recurrent retrieval policies.

Further mathematical-evaluation work should require a formal inference certificate, inspect implicit assumptions, and reject invalid theorem applications under changed hypotheses. Every genuine proof claim still needs independent Lean kernel checking. The BSA-R source and this test remain public, so future sealed confirmation must use fresh roots and unseen question families.
