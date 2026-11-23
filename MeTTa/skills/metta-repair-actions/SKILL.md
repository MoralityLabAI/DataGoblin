---
name: metta-repair-actions
description: Propose typed repair actions for DataGoblin curation states, use verifier feedback, and check STOP decisions and bounded multi-edit exchanges.
---

# Propose verified repairs

1. Read the public state, legal-action interface, verifier budget, and edit
   limit. Record the source root. Predict typed actions and filter them
   through the environment before treating edits as legal.
2. Prefer a verified edit that advances a complete repair. A locally tied
   or non-improving edit may be necessary. If the controller supports it,
   propose a bounded sequence and verify all transitions before committing
   an atomic exchange. Tie-aware supervision should allow multiple equally
   good legal first edits when the oracle agrees.
3. Use a verifier rejection as an observation for the next proposal. Keep
   rejected edits out of public state and count all verifier calls and
   extra model passes, including rejected attempts. Respect the requested
   cap. Use exact legal repair distance when an independent oracle
   provides it.
4. Propose STOP only when the terminal check succeeds. Report exhaustion,
   abstention, and premature STOP separately from successful completion.

DataGoblin fixtures for tie-aware action targets and bounded exchanges:

```sh
python -m pytest -q tests/unit/test_metta_looped_action_ties.py tests/unit/test_metta_looped_stop_exchange.py
```

Return the candidate and observed transitions, rejection reasons, final
terminal verdict, and actual calls and recurrent passes.
