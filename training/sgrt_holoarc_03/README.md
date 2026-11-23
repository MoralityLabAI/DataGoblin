# DG-SGRT-03: sequential storyworld evidence, same-set holonomy test

**Stage:** CPU-only synthetic mechanism development, 2026-11-23. This is a
stacked follow-up to DG-SGRT-01/02 (draft DataGoblin PR #53), which reported
a negative three-seed learned candidate-ranking result. All older results
and frozen SRT/SGRT protocols remain unchanged.

## Target question

Can path-sensitive transport help a small learned recurrent policy revise
its decision when new evidence arrives **after** the preceding decision,
when candidate coverage, task families, verifier calls, and trainable
parameters are held fixed?

A synthetic world has a four-bit public state. Fictional events such as
reversing a witness flag, rotating courier seals, and closing an archive
are deterministic noncommuting transformations of that state. Each root
has two episodes with identical initial state and the same two event
packets, presented AB and BA. Their final states deliberately differ.
A public episode exposes the initial state and events in order. The
environment-owned scorer alone computes the intermediate and final
correct labels with an independent exact interpreter.

All policies receive exactly 16 legal action candidates (a forecast of
the next world state) at every step. Each policy predicts **once after
the first event**, keeps its prediction, then receives the second event
and predicts again. The simulator checks each prediction after the run;
its result is not fed into the policy. Thus all policies have identical
candidate multisets, four exact checker calls per paired root and two
recurrent visits per episode. Distinct root pairs, rather than paired
episode rows, are the unit of statistical inference.

## Controls

Five small NumPy heads have 937 trainable parameters each, the same
random initialization and root-pair ordering, the same 384 Adam updates,
and both intermediate and final supervised losses on the training split.

| Arm | Mechanism |
|---|---|
| bag | Order-agnostic sum of event embeddings, exact alias negative control |
| direct | Ordinary learned two-step tanh recurrence with identity transport |
| flat | A commuting event-specific orthogonal transport |
| curved | Noncommuting event-specific holonomy produced by closed 12D spirographic loops |
| random | Independent orthogonal conjugates of flat transport, matching each event's eigenspectrum |

For flat and curved, the per-event matrices have the same trace,
determinant, eigenangle multiset, orthogonality and Frobenius distance
from identity. Flat matrices use a shared orthogonal plane atlas and
commute across events. Curved matrices are Wilson products along closed
spirographic paths; their commutators are nonzero. Random control has
matched event spectra and can also be noncommuting, without spirographic
organization. This is a useful alternative when learned history matters
for reasons unrelated to the selected geometry.

A sixth comparison, curved_transport_swap, uses the **same trained
curved weights** and preserves the chronological event embeddings but
swaps which event's transport operates at each step. Its per-root
transport multiset is unchanged. This tests sensitivity to event-to-
transport ordering, without changing candidate pools or checker calls.
The exact ordered simulator is the seventh comparison and ceiling.

All learned modes share a 16D carried state and the same update:

    h0 = tanh(initial_table[public_initial_state])
    gt = sigmoid(gate_logits[public_event])
    ht = tanh(h(t-1) @ Wrec + event_table[event] + gt * (Qevent - I) @ h(t-1))
    logits = ht @ Wout + bias

The bag control uses commutative accumulated evidence after the
first visit, making identical second-step logits for AB and BA.
Its residual gate is inactive; direct has an identity transport and
is the ordinary recurrent comparator. Because flat/curved/random apply
a frozen geometry with additional small-matrix operations, equality
of trainable parameters does not establish equal floating-point cost.
Record actual runtime before any model-architecture promotion.

## Public split and default run

- Training: 24 unique noncommuting lineage roots in each of three
  families, 72 paired roots or 144 two-event episodes.
- Evaluation: 12 roots per family across all four families, including
  12 entirely held-out event-pair compositions; 48 pairs or 96 episodes.
- Each root is keyed by initial state and unordered event pair. Stable
  SHA-256 bucketing separates train and evaluation within the first
  three families. The held-out family uses different unordered event
  pairs whose individual operations occur during training.
- No exact target or hidden evaluator state reaches the forward pass.
- Default optimizer seed 23; 384 paired-root updates; Adam LR 0.014;
  two visits/episode; exact candidate pool size 16; four checks/root.
- Public generator/split seeds 31003 and 61003, recorded in source.

### Run

    python -m pytest tests/unit/test_sgrt_history.py -q
    ruff check src/datagoblin/training_gym/sgrt_history.py tests/unit/test_sgrt_history.py
    python -m datagoblin.training_gym.sgrt_history --seed 23 --updates 384 --train-per-family 24 --eval-per-family 12 --out runs/dg-sgrt-03-public

The workflow in .github/workflows/dg-sgrt-history.yml runs the same
commands, validates the geometry and oracle, then uploads the full
development report and scorer-owned individual rows. It needs NumPy
only, runs on the GitHub-hosted CPU, and never invokes Inky Mini, the
local GPU, paid APIs, or any trained Aleph checkpoint.

## Measurement and claim boundaries

The primary diagnostic is pair-level **both_orders_correct**: an arm
must predict the final state in both counterfactual orders for the
root to count. Report first-visit accuracy, final-visit accuracy,
order-distinguishing output count, and exact-oracle ceiling.
An order-blind bag model has an upper bound of one correct final
episode per paired root by construction.

Use paired root bootstrap contrasts against direct recurrence. Break
out the held-out mixed family. Keep every failure and zero count.
Any positive curved result must also exceed the strongest random and
flat transport controls under a matched resource budget before
claiming value from the spirographic connection.

The initial stage tests **frozen geometric priors** in a learned
recurrent head. It does not train the connection itself. The world's
event operations are known, finitely enumerable and much easier
than open-ended creative storyworld generation. The checker supplies
no online reward or corrective hint between the two decisions.
A prospective interactive follow-up can provide an action-dependent
second observation under controlled oracle custody after this baseline.

## Stop / promotion rules

Keep DG-SGRT-03 exploratory until paired root intervals and multiple
independent initialization seeds exist. A single public development
run cannot change the current HOLD decision on Aleph integration.
If curved transport loses to direct or random, retain that null.
If it wins, register independent family-disjoint sealed runs and a
matched-compute adaptive comparison before any real host sidecar.

Proposer autonomy stops at a prediction. Only the fictional simulator
may change the world. No API credentials, actual trading, warfare,
deployment authority, model downloads, or production side effects.
