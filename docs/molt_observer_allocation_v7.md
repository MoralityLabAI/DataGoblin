# MOLT-07: forecast allocation of observer maps

Development planner for the [LoopedTransformers observer-memory cache](https://github.com/MoralityLabAI/LoopedTransformers/pull/403). Reuses its payload contract (32-byte binding key, 12-byte header, float64 basis and spectrum). A selected map occupies persistent serialized storage; the budget excludes weights, temporary SVD tensors, allocator overhead, coefficient archives and process-tree memory.

The operator supplies immutable binding digests, monotone upper bounds on singular tails, maximum intervention radius, optional uniform Hessian bound, numerical rounding allowance and error budget. Values are nonnegative integers in nano-units. The linear mode uses a conservative ceiling of radius times singular tail. Conditional nonlinear mode also charges the uniform Hessian bound times radius squared; without that bound approximate ranks are unavailable in this mode. Full rank has zero mathematical projection loss, subject to the supplied numerical allowance. The planner cannot certify any supplied bound or checkpoint.

For each admissible rank, forecast value is (requests - 1) times (cold build microseconds - hit microseconds) minus encoding microseconds. The model assumes one construction and subsequent hits for a resident map. It rejects nonpositive options. A sparse exact multiple-choice knapsack selects at most one rank per binding, maximizing forecast saved time under the shared byte cap. Ties minimize bytes and then use canonical binding/rank order. No-cache remains available. This is a forecast of a fixed resident set, not a general LRU trace optimizer; the measured reuse and time receipts must come from separate evaluation.

```sh
python -m pytest -q tests/test_molt_observer_allocation.py
python -m datagoblin.training_gym.molt_observer_allocation --offers /private/offers.json --out /private/plan.json
```

Input is a JSON object with schema molt07-map-offers-v1, mode linear or conditional_nonlinear, cap_bytes, and offers (each with every MapOffer field). Ranks index the tail array at k; the dimension+1 entries include the rank-zero tail and a terminal zero. hit_us and encode_us arrays are indexed by k-1. No model endpoints, training, or hard deductions are executed. Do not treat provider forecasts as actual FLOPs, wall-time or RSS. Confirmation, promotion, and certificates remain forbidden.
