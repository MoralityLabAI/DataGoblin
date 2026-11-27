# DataGoblin Research Verifiers v2.1 (`datagoblin-research-v2-1`)

Separately versioned successor to `datagoblin-research-v2` (preserved unchanged
from PR #81 / PR #93). Adds:

- Per-episode action budgets (`8` turns per episode without cross-episode starvation)
- Persistent compact research state (`compact_state`) in every turn observation
- Intermediate verified capability scores (`capability_score`) reported beside strict terminal `primary_score`
- Controller-assisted stopping, learned LDT stopping, TRM action routing, and RLT state correction
- Independent declarative reference semantics for owner, scope, temporal, and revocation persona state

## Validate

```bash
python -m pip install -e '.[dev]' -e environments/datagoblin_research_v2 -e environments/datagoblin_research_v2_1
vf-validate datagoblin-research-v2-1 --runtime.type subprocess -n 8
```
