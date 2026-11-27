# ZKL-12 degree-tier curriculum
Seven synthetic all-public families use topology-separated development and holdout
splits. Degree-zero transitivity and scaling, degree-one multiplication and inverse
transport, degree-two square transport, and essential rows are scored independently.
F7/F11 use complete finite assignment checks; BN254 skips enumeration. Every task
tries degree budgets 0, 1, 2 against the independently pinned SatsHedgeZero checker.
The exact identity, exhaustive relation, and Lean replay are separate evidence levels.

The scoped CI workflow checks out SatsHedgeZero at a52debae36578f5f875fe78f1b13234eb351fca0,
then runs the cross-repository test at three moduli. It covers 42 tasks and 126
degree-budget calls (two roots per family and modulus). Public seeds are development
data only. No model calls or promotion, no production constraint rewrites.

Run `python -m datagoblin.zk_poly_gym --seed 27 --split holdout --field 7 --out runs/zkl/poly.jsonl`.
To invoke the checker, pass --checker and the three independently reviewed source
SHA-256 values (--poly-sha256, --ideal-sha256, --audit-sha256). No certificate is
marked Lean-replayed here. 
