# DISTILL-18: native MoLT exact selection compactification

Opt-in DataGoblin port of principles established by [DISTILL-15](https://github.com/MoralityLabAI/Compactification-Theory/pull/319) and [DISTILL-17](https://github.com/MoralityLabAI/Compactification-Theory/pull/321). It does not import the synthetic Distill model or change existing default policies, training, provider routes or host verification.

## Exact scope

MoLT-GOBLIN-02 uses positive integer costs and utilities with an exact knapsack objective and lexicographically smallest sorted ID tuple on ties. The canonical selection key is the sorted tuple of eligible affordable (ID,cost,utility) records and the capacity saturated at their total cost. Removing a forbidden or individually unaffordable record cannot change a feasible choice; saturating capacity preserves the feasible set. Equal keys thus induce identical feasible ID sets, utilities and the same deterministic optimum. For positive utilities, all items must be selected if all fit. If the two cheapest costs exceed capacity, a feasible nonempty solution has one item. Otherwise execute the same sparse DP as the native reference, preserving its tie rule. Input limits and positive integer checks are explicit.

A lazy LRU holds only the selected ID tuple, never receipt IDs, frame IDs, permissions or verified claims. Every submission attaches current public frame and receipts. A cache hit cannot make a stale receipt current; the existing Episode independently checks admissibility, issued receipts, source versions, and exact optimum. The policy inherits the selective-refresh observation transitions. Memory is reset per root, stream and arm in this panel.

## Registered development campaign

64 new TRAIN and 64 calibration roots, fixed independent seeds, both native relevant/unrelated revisions. Two workloads: one pair per root (distinct variants) and eight repetitions of that pair (explicit repeat stress, not new independent roots). Four arms: unmodified reference, compiled without persistence, bounded lazy two- and eight-entry LRU. Each episode is a fresh native host with fresh public observation processing. Compare entire action and observation traces to reference and require every episode to pass host verification and optimality. No hidden evaluation roots or data exports. No prefilled bank.

Primary endpoints are trace/host parity and descriptive DP visits, fast-path frequency, cache hits, misses, writes, evictions, and submission process CPU time. DP visits are only a count of the fallback algorithm's inner iterations, not total work. Actual submission timing includes normalization, hashing, selection, eviction and fresh receipt binding, but excludes other episode transitions and offline reference instrumentation. Report both repeat workloads separately; do not infer on-policy recurrence or hardware-level speedup from repeated replay. Full per-root receipts remain in Actions artifacts. No training admission or promotion. No model or paid API calls.

Run locally:
```sh
python -m pytest -q tests/test_molt_compact_policy.py tests/test_molt_memory_panel.py
python -m datagoblin.training_gym.molt_compact_policy --roots 64 --out /tmp/distill18-native-molt.json
```
