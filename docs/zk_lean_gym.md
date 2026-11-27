# ZKL-08: proof-guided circuit optimization gym

Development-only CPU finite-field fixture; no model inference, training, funding or
promotion. Shapes use the same zkl-r1cs-v1 schema as SatsHedgeZero's independent
importer, with modulus 7 or 11 and a hard 50,000-assignment cap per circuit.

```sh
python -m pytest tests/test_zk_lean_gym.py
python tools/gym_ram_watchdog.py --cpu-only -- python -m datagoblin.zk_lean_gym --split development --seed 7 --per-family 2 --optimize --out runs/zkl/dev.jsonl
python tools/gym_ram_watchdog.py --cpu-only -- python -m datagoblin.zk_lean_gym --split holdout --seed 19 --per-family 2 --out runs/zkl/holdout.jsonl
```

Development families: bit, multiply, affine. Public holdout families: two-bit range,
private selector, composed multiply/add. These are topology-separated *development*
tests, not a secret evaluation; use operator-owned disjoint roots and hidden families
for promotion. Each root has clean, row deletion, duplication and coefficient mutation.
The independent exhaustive verifier enumerates all canonical assignments, compares
existential public projection with the direct spec, records a concrete soundness
counterexample or missing public tuple, and separately reports completeness.
Multiple valid private witnesses are permitted. The deterministic optimizer removes
one row only if the starting relation is sound and complete and the exhaustive
public projection remains equal. Duplicate rows suggest a generic Lean redundancy
proof; other edits have finite evidence only. No repair is called a Lean theorem.

Next actor stage: local Inky Mini or TRM/LDT proposes typed edits and proof script,
an independent solver owns the receipt, and pinned Lean replays the proof against
the exact emitted shape hash. Retain rejected actions and verifier cost. Never put
hidden evaluation seeds into adapter training. Compare matched CPU or verifier
budgets for uniform deletion, incidence heuristic, greedy exact control, learned
ranker, recurrent repair and shuffled ranker. Reward sound AND complete before
constraint savings; inconclusive is no credit. Count proof obligations discharged,
failed proof attempts, verified counterexamples, time and peak RAM. Report fresh
held-out families separately. A production promotion requires full field semantics,
artifact binding, actual Nova integration and human review.

10k-case scaling is a proposed campaign, not run here; the initial fixture uses
small fields and does not model SHA-256, full V3 clearing or cryptographic soundness.
Clean/CLAP translation is gated by same public semantic relation and explicit field
mapping, not code similarity. Companion MathCan and SatsHedgeZero branches:
research/zkl-01-09-20261127.
