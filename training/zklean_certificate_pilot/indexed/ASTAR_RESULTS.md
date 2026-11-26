# ZKL-10: exact shared-proof A* with ordered-height lower bound

Exploratory follow-up to ZKL-09 on the same public pilot seeds, 2026-11-25. No trained model or paid GPU use.

For normalized atom a, let H(a) be the minimum dependency height of a proof tree, including a itself: H(a) = min over rules r with head a of 1 + max over premises p of H(p), where max(empty)=0. For obligation set M, h(M) = max over a in M of H(a) + |{b in M : b>a}|. Every derivation of a includes a dependency path at least H(a) long; because all edges strictly decrease the normalized index, no atom on that path can coincide with any outstanding larger atom. Thus h is admissible. A* with reopening and a goal-pop stopping rule finds the minimum unit-cost shared DAG when it terminates within the cap. This is restricted to the generated acyclic grammar (all atoms reachable); no claim for general Lean proof search.

| split | exact DP complete | A* complete | rescued | matched distinct DP states | matched A* discovered states | A* max discovered |
|---|---:|---:|---:|---:|---:|---:|
| small | 128 | 128 | 0 | 140061 | 3422 | 122 |
| size | 128 | 128 | 0 | 417448 | 10205 | 1321 |
| grammar | 117 | 128 | 11 | 2875721 | 83548 | 76080 |

Cap: 100000 distinct states per case; A* counts discovered states (including the frontier), not only expanded states. All 384 certificates replayed; all 373 uncapped DP costs match exactly; the 11 remaining A* optima follow from admissibility, not independent DP equality. Total A* rows 710/1222/983. This follow-up uses already examined seeds and is diagnostic, not sealed or preregistered confirmation. The paired size shift state-count reduction is 40.9x; runtime/memory speedup is not inferred from state count alone. The hardest A* instance discovered 76080 states, so some exponential search risk remains.

Run `python astar.py` and `python -m pytest -q test_weighted.py test_dag.py test_astar.py` in the indexed directory. Local tests: 64 passed (30 earlier weighted tests, 17 DAG tests, 17 A* tests). The A* tests independently compare 16 tiny instances against forward-state Dijkstra and exercise the state cap.

Next gate: a learned ranking arm must retain this lower bound and verifier, compare to A* on fresh generator families with identical search accounting, and report solve coverage, certified cost, state count, wall time and peak memory. A trained ranking should not replace the admissible bound unless it independently proves admissibility. Use the measured Nova step constraint count only for the particular circuit it describes.
