# MOLT-09 native observer accounting

Companion to [LoopedTransformers #407](https://github.com/MoralityLabAI/LoopedTransformers/pull/407).
Recomputes map admission and payload accounting from native teacher swap receipts.
The baseline uses exact native output-span coordinates. A map must pay for
its basis/spectrum and the fallback tag. Common physical basis and teacher
response target bytes are charged separately, once per declared scope.
A duplicate displacement cannot support the amortization decision.

The importer checks complete paired donor identities within the report,
parity gates, eligibility arithmetic, and cost summaries. The report is
an engineering receipt, not independent evidence of its source identity,
numerical certificate, memory cap, or actual runtime. No promotion.

    python -m pytest -q tests/test_molt_native_observer_accounting.py
    python -m datagoblin.training_gym.molt_native_observer_accounting --report /tmp/molt09/replay/receipt.json --out /tmp/native-accounting.json

The unit fixture deliberately shows a 132 B saving against 1,024 B of
native coordinates, with 352,000 B of teacher response targets and
8,192 B of common chart data. This is a synthetic arithmetic fixture,
not the measured native run.
