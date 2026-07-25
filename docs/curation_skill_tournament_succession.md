# Curation-skill tournament succession

The succession tests `curate-game-agent-data` under several kinds of curation pressure and turns
only tournament-level, content-free evidence into a future LDT selection task. It does not train an
LDT or release underlying game records.

The frozen contract is
`configs/preregistrations/curation_skill_tournament_succession_v1.yaml`.

## Pressure families

Each pressure has a development tournament and a lineage-disjoint heldout tournament:

| Pressure | What it tests |
|---|---|
| `boundary_precedence` | Sealed route-only custody, cumulative hidden/private diagnostics, and private-before-unsupported routing |
| `authority_provenance` | Separation of successful field mapping from evaluation, simulated, inferred, unresolved, and verified authority |
| `balanced` | Ordinary and semantic schema drift with legitimate mapping opportunities |

Every round runs normal, compact, and coverage-pressure learned adapters against the frozen
exact-key baseline. Safety gates precede quality. Results remain per game, and each tournament
retains its Pareto front instead of emitting a universal champion.

## Taste contract

Pairwise curation preferences use this order:

1. safety qualification;
2. per-game Pareto dominance over ordinary and OOD exact mapping and mapping F1;
3. lower parameter cost only when quality is equivalent within the frozen tolerance; and
4. `TRADEOFF` when neither side dominates.

Environment-owned counterfactual controls provide explicit unsafe-high-coverage,
safe-zero-coverage, and ordinary-versus-OOD-specialist comparisons. The dataset gate requires all
three preference labels and all four rationale codes to occur; a superficially green curriculum
that never teaches tradeoffs fails.

## LDT foundation custody

Development rounds emit public training pairs with targets. Heldout rounds emit public inputs while
their targets remain in `private_steward/heldout_labels.jsonl`.

Allowed features are game, pressure family, context mode, optimization pressure, parameter bucket,
safety qualification, and ordinary/OOD exact mapping and mapping F1. Row-level records, source
values, paths, hidden reasoning, token use, source-agent identity, and sealed metrics are excluded.

This authorizes training only for the synthetic meta-selection task. It does not authorize a larger
model to train on private game corpora.

## Run

Fingerprint the installed skill before and after the evidence-supported mutation:

```powershell
python $env:USERPROFILE\.codex\skills\curate-game-agent-data\scripts\fingerprint_skill.py `
  $env:USERPROFILE\.codex\skills\curate-game-agent-data `
  --out D:\Temp\curation-succession\skill_after.json
```

Then run:

```powershell
$env:PYTHONPATH = "src"
python -m datagoblin.adapter_verifier skill-tournament-succession `
  --config configs/preregistrations/curation_skill_tournament_succession_v1.yaml `
  --skill-before-fingerprint D:\Temp\curation-succession\skill_before.json `
  --skill-after-fingerprint D:\Temp\curation-succession\skill_after.json `
  --candidate-validator $env:USERPROFILE\.codex\skills\curate-game-agent-data\scripts\validate_adapter_candidate.py `
  --out D:\Temp\curation-succession\run
```

Generated tournament runs, weights, steward labels, and LDT rows stay outside the repository.
Commit only code, contracts, tests, and durable documentation.
