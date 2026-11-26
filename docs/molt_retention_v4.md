# MOLT-04: learned evidence retention under delayed queries

Status: development-only synthetic memory allocation with actual NumPy training.
MOLT-03's fixed two-receipt fixture motivates a harder problem: select evidence
before the final query is revealed. This gym isolates retention; the host executes
the final read and exact verification. It does not evaluate a language model's
ability to write an answer or reproduce native TRM, MoLT or BPD.

## World and independent unit

A root produces ten public evidence entries (18 in length-shift development).
Each entry has five binary metadata features: scope match, quality hint, proof
hint, novelty and the public task cue. The host draws two distinct queried
sources from a distribution proportional to the product of their positive weights.
The query is hidden until after the memory bank has been filled. The generator's
weight is

    w = 1 + 8 scope + 24 quality proof + 9 novelty (1 - proof) + 6 scope cue.

No receipt value or source index enters the learned feature vector. The generator
law is intentionally available to the exact reference baseline, whereas the fitted
heads receive only TRAIN features and sampled query-membership labels.

Each development root has a clean and a revised-source episode. The revision
selects one independent uniform source and changes its value/version after
retention; the old receipt expires. The two variants share all pre-query observations
and the same final query. All non-hindsight policies retain the same entries in
both variants. The host then allows zero or one charged fresh verifier request.
Success requires both current receipts and the correct sum mod 17, checked by
the host. An unavailable required receipt produces a failure. Retained evidence
never gains hard authority merely because a model selected it.

## Learned head and comparisons

With x in {-1,+1}^5, h[0]=0 and a shared cell,

    h[r+1] = tanh(x W_x + h[r] W_h + b)
    score = h[R] W_o + c.

The model learns future query membership with stable binary cross-entropy and
explicit backpropagation through the recurrent cell. There are 369 total parameters
at width 16; the one-pass arm has 113 active parameters because h[0] is zero.
The matched initialization includes the inactive recurrent matrix. Full BPTT,
actual SGD updates and initial/final weight digests are recorded; the tests
check derivatives against finite differences.

| Policy | Role |
| --- | --- |
| FIFO | Retain the latest sources |
| Random | Deterministic random priorities within each root |
| Scope | Rank only the scope-match feature |
| Generator rank | Privileged knowledge of the true public-feature query law |
| Hindsight | Privileged ceiling with access to the final query and revision |
| Single | One-pass learned head |
| Tied3 | Three shared recurrent passes, same data/updates as Single |
| Single reinvest | Three times the updates, matched forward cell-example visit count |
| Tied3 shuffled | Same architecture, permuted query labels within each TRAIN root |

The fixed work comparator matches cell-example visits, not FLOPs, wall time or
parameter activity. The input projection is reused across recurrent passes and
optimizer updates differ in the reinvestment arm. Report these axes separately.
All learned scores are frozen before calibration evaluation. The head is a small
tied scorer inspired by recurrent reasoners, not a complete TRM implementation.

## Exact generator-aware reference

For weights w_i>0, the probability of a pair {i,j} is w_i w_j / Z. If a bank S
has h hits on a queried pair, success with b verifier calls requires h >= 2-b.
After a uniform random revision, its conditional success at h=2-b is
1-h/n; above that threshold the episode still succeeds. In either regime the
conditional success function g(h) is nondecreasing.

If a retained source a has lower weight than an unretained source c, swapping them
changes expected success only for pairs with one of those sources and a third
source j. Their combined change is proportional to
(w_c-w_a) w_j [g(s+1)-g(s)], where s is the bank membership of j. It is nonnegative.
Repeating the exchange yields the top-k weight bank. This is an exact expected
success result for this generator, fixed capacities and the stated revision law;
it does not require the model to know the realized query. The test compares
the rank policy with exhaustive subset search on small roots using rational arithmetic.

## Running

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_molt_retention.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m datagoblin.training_gym.molt_retention_train \
  --train-roots 1024 --dev-roots 128 --steps 256 --seeds 31 37 41 43 \
  --out /tmp/molt-retention-v4.json
```

The report contains TRAIN and calibration manifest hashes, model fit receipts,
complete per-root outcomes across capacities 1/2/4/6, zero/one verifier call
and both revision variants, plus source hashes. The fixed development workflow
stores results as a temporary Actions artifact. Do not commit generated run files.
Serialized receipt payload bytes are a representation count, not total RAM.
Real checkpoints still require the operator's hard resource cap and the Research
Verifiers v2 endpoint integration. This bounded CPU fixture has no endpoint calls.

The same development worlds are crossed with all training seeds. Rows are
correlated within both world and model; report world counts and seed counts
separately. The longer-length split is development-only, with no sealed
confirmation or checkpoint ranking. Use EvalMath preflight and fresh root
populations before a causal or generalization claim.

## BPD/MMS next

Use the trained utility head as a baseline for memory-component interventions.
Fit component selection on TRAIN only, preserve full-write reconstruction and
compare targeted donor restoration against matched random, displacement and
future-readout Jacobian controls. Reevaluate ordinary inference after steering
training. The first learned retention study does not itself implement BPD.
For MMS, sweep actual byte budgets with process-tree peak RSS and bank maintenance
cost, alongside the current fixed-slot controls. The scalar receipt count should
not be interpreted as a hardware memory measurement.
