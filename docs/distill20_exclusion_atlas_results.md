# DISTILL-20 hosted exact exclusion atlas result

Tested commit f9e22ccbd28820d979f71d7b6c046b27cde4fc8e. [Dedicated run 46103155314](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46103155314) passed focused pytest, Ruff, complete campaign and [artifact upload 15245688426](https://github.com/MoralityLabAI/DataGoblin/actions/runs/46103155314/artifacts/15245688426). Raw rows digest 089e895ee29e3d5c9507b2fc8412ba268aad3efcb49d50b0299fc9f226e19de5. All 576 native episodes preserved reference traces and passed host optimality and verification. All 26,496 public solver requests preserved reference action equality, feasibility and current receipt binding. Public horizons share underlying roots, so cell wins are not independent across horizons.

| Public workload | Eager wins vs packed dense | Delayed wins vs packed dense |
|---|---:|---:|
| stable/1 | 0/96 | 31/96 |
| stable/4 | 71/96 | 0/96 |
| stable/16 | 96/96 | 96/96 |
| stable/32 | 96/96 | 96/96 |
| churn/16 | 24/96 | 0/96 |

32 stable requests, eager atlas speedup over packed dense, including preparation (tight/medium): width 16 2.75/5.39x, 32 6.18/7.72x, 64 9.29/8.57x, 128 10.59/8.57x. At width 128 medium, reference 15.493 s, packed dense 2.616 s, eager 0.305 s across 384 requests; reference/eager 50.76x. At 128 tight, packed dense 0.725 s vs eager 0.06845 s. Peak temporary cell count 132,225, no build exceeded 150,000-cell cap in this roster. This is a logical table-cell bound, not process RSS. No eager atlas wins native cold aggregate cells; delayed performs no builds in native episodes. Eager timing in a few individual cold roots is noisy. These are descriptive CPU timings and wider public solver replays, not native deployment results.

Stable long scopes justify optional atlas compilation. Churn and uncertain horizons require cheaper admission. Existing five-candidate default unchanged. DISTILL-21 was registered independently before these results were retrieved; its critical-only witness approach and frozen thresholds remain unchanged. No provider calls, protected data, training admission or model updates.
