# ZKL-11 gym: algebraic redundancy and abstention

These are small, all-public synthetic R1CS deletion tasks. Development families are
linear transitivity, scaled product and essential linear constraints. Held-out
topologies are quadratic sums, multiplicative implication and essential quadratic
constraints. The multiplicative implication is deliberately equivalent but outside
the constant-multiplier span (x=0 implies x*y=0). A solver must abstain there, just as
it must abstain for genuinely essential rows. Small-field exhaustive evaluation
distinguishes those cases independently. BN254 fixtures do not enumerate assignments.

Use the external checker in a reviewed SatsHedgeZero checkout, on the exact reviewed
source blobs. Both source SHA-256 pins are mandatory; choose them independently of
the proposed certificate. Example:

```sh
python -m datagoblin.zk_ideal_gym --seed 7 --split development --per-family 8 --field 7 --out runs/zkl/ideal-dev.jsonl
python -m datagoblin.zk_ideal_gym --seed 19 --split holdout --per-family 8 --field bn254 \
  --checker /path/to/SatsHedgeZero/research/zk_lean/ideal_certificate.py \
  --checker-sha256 "$PINNED_CHECKER_HASH" --audit-sha256 "$PINNED_AUDIT_HASH" \
  --out runs/zkl/ideal-bn254.jsonl
```

The gym keeps finite equivalence, exact identity receipts and proof replay distinct.
It records generated Lean source for future independent replay; no receipt is a
production promotion. No model calls, training, or optimization of deployed circuits.
The fixture family names and public seed values are development data, not hidden eval.
