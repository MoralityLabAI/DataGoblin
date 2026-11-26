# MOLT-11 isolated response memory accounting

Companion to [LoopedTransformers #409](https://github.com/MoralityLabAI/LoopedTransformers/pull/409).
The planner consumes a completed engineering report, validates source identifiers
and internal archive/phase/work/parity receipts, charges common tensors,
stored targets, head/Gram costs or the separate recompute teacher, and compares
the observed setup/training RSS against a proposed budget.

The repeat forecast uses the measured target-preparation phase once and measured
training phase elapsed time times a declared number of identical repeat runs.
Common setup and the cross-arm parity audit are reported separately. This assumes
the same immutable teacher and workload and does not predict a changed model's
runtime. A single fixed-order timing panel is descriptive. The RSS cap comparison
is post hoc, not a resource reservation or enforced process-tree bound. Actual
source content and training equivalence are not independently certified by this
internal accounting pass.

    python -m pytest -q tests/test_molt_isolated_response_accounting.py
    python -m datagoblin.training_gym.molt_isolated_response_accounting \
      --report /tmp/molt11/isolation/report.json --payload-cap-bytes 20971520 \
      --rss-cap-bytes 2147483648 --reuse-runs 4 --out /tmp/isolated-plan.json

A failed competence gate remains visible. Passing the parity study supplies no
training-effect or promotion authority. No protected split or paid compute.
