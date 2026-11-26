# MOLT-05: frozen recurrent-write restoration

Development design frozen before the fixed CPU workflow. This is an SVD analogue of the
[BPD MOLT prefix protocol](https://github.com/MoralityLabAI/BPD/blob/722613a80d80af0e80bb9f5ac1ce83eef111e4f7/mechinterp/bpd/experiments/molt_prefix_memory/PROTOCOL.md).
It uses the MOLT-04 tiny tied utility head, not a native BPD factorization or a prefix-attention model.

## Instrument

For each entry x, let f=x W_x+b, h1=tanh(f), z2=f+h1 W_h, and
y=tanh(f+tanh(z2) W_h) W_o+c. The recipient intervention location is z2,
the actual second-pass preactivation write. Decompose W_h=U diag(s) V^T
and preserve every singular component. A root-determined nonidentity cyclic
donor permutation replaces the recurrent write h1 W_h while preserving f.
The donor is another receipt in the same world, independent of the eventual query.
The intact computation is an oracle source for the restorative vector; this is
a deliberately synthetic corruption and does not establish a deployable retrieval policy.

Fit on the MOLT-04 TRAIN generator (1024 roots), fresh seeds 53,59,61,67,
256 updates. Select four singular directions using mean squared TRAIN signed
future-score Jacobian contributions to the original-minus-donor write. Freeze
model and selection before development. Evaluate fresh seed-991 worlds
(128 short and 128 long), each in the clean/revised source variants, capacity 2
and one verifier call. Scores are rounded to 10 decimals for the host ranking,
with deterministic source-index ties. Full reconstruction and full restoration
must preserve both scores within 1e-9 and the resulting retention decision.

## Sham construction

For target displacement d at recipient z, the scalar derivative is
J = diag(1-tanh(z)^2) W_h [W_o * (1-h3^2)] (column convention).
The complete future score vector has a block-diagonal Jacobian because
the scorer has no interaction between entries. For each entry construct a
distinct vector s with

    ||s|| = ||d||, J s = J d, z dot s = z dot d.

The last equality and the first imply ||z+s||=||z+d||.
Project d onto span(J,z); rotate its orthogonal complement in a deterministic
random direction perpendicular to the target residual. This matches signed
first-order future scores, recipient displacement, and endpoint state radius,
at the same allowed full 16-coordinate recipient support. It does not match
SVD component sparsity; a separate disjoint rank-four control matches the
displacement when feasible. A norm-only sham is another diagnostic.
The shams do not inspect query labels. Zero-dose entries are recorded separately;
a failed distinct match executes the unsteered baseline in that arm and remains
in the full denominator. Any active unmatched entry blocks a strict matched
claim. Do not interpret zero-dose rows as informative mechanism evidence.

| Arm | Purpose |
| --- | --- |
| Intact | Task competence |
| Reconstruction | All components, parity |
| Corrupted | Donor-permuted write, repair headroom |
| Full restore | Privileged complete restoration |
| Targeted | Four TRAIN-selected components |
| Equal rank | Disjoint random four components, norm-scaled |
| Norm sham | Random direction, equal displacement |
| Jacobian-radius sham | Distinct direction, equal signed Jd, norm and endpoint radius |

For each active intervention also apply the same restorative target vector at
the clean base and construct its sham at that base. This deliberately ungated
overshoot tests clean-state damage, not detection of a corruption. The original
intact score is available for error measurement. There is no steering during
training; all weights are frozen after fitting.

## Outcomes

The primary descriptive slice is long, revised-source worlds. The balanced
score is (verified corrupt + verified clean)/2 across the entire fixed roster.
Report targeted-minus-each-sham paired deltas, per-model deltas, actual match
counts, conditional recovery among intact-success/corrupt-failure pairs,
damage among intact successes, and mean absolute score error to the intact
model. No selecting a favorable comparator after the run. Four independently
initialized models share training and evaluation worlds; task and model
variation are crossed. No population confidence or causal promotion from
this development panel. A future registered study needs EvalMath preflight,
fresh independent roots and a meaningful measured effect floor. The earlier
native BPD cohort seals remain untouched.

The signed Jacobian controls a continuous readout, while top-k and exact
verification are discontinuous; equal Jd does not imply equal actual score
changes or ranking. If the strongest sham performs as well as the target,
do not treat a weaker sham advantage as evidence of donor-specific content.
The recipient is a preactivation, not a post-tanh state; the reported radius
and norm must be interpreted at that location. There is no measured hardware
memory savings or checkpoint deployment in this experiment.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q tests/test_molt_retention_steer*.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m datagoblin.training_gym.molt_retention_steer_run \
  --out /tmp/molt-steer-v5.json
```

The report retains all episode outcomes, selected components, model/source
hashes, per-root match audits, final score vectors, and paired contrasts.
Generated reports stay outside Git. The CI artifact is development-only and
expires after seven days. The fixture allocates no model endpoint or GPU.
