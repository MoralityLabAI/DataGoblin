# ZKL-09: minimum shared proof DAGs

Exploratory CPU result, 2026-11-25. No training, no paid GPU use. Original indexed grammar from ZKL-07. Step count is a proxy for uniform folding expense; the indexed rules would still require their own circuit for dependent typing.

Choose a set M of outstanding normalized atoms. Expand the highest-index atom h in M with a ground rule r. Minimize c(r)+D((M - {h}) union premises(r)), with D(empty)=0. All premises have smaller indices. Every reachable goal derivation can share identical normalized atoms; positive costs imply one derivation per atom suffices. Expanding the highest atom first makes previous choices irrelevant to remaining subproblems. The algorithm is exact for this grammar and objective, subject to a 100,000-state cap; exponential state growth remains possible. The proof certificate is checked independently (rule, dependent binder, reductions, backwards references, goal, absence of duplicate atoms and unused rows). Independent forward-state Dijkstra agrees on both objectives for 16 small exercises.

## Run

From this directory: `python dag.py` and `python -m pytest -q test_weighted.py test_dag.py`. Local focused run: 47 passed. Public pilot seeds 40000-40127 (limit 12), 50000-50127 (limit 24), 60000-60127 (limit 24 shifted grammar). Same cases for all arms, all 11 capped cases excluded from paired aggregate; no generalization claim for excluded cases. No generated runs, proof corpus, checkpoints, or private evaluation seeds committed.

| split | completed | capped | max states | unit tree + share rows | weighted tree + share rows | exact step rows | exact proxy rows |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 128 | 0 | 3503 | 760 | 814 | 710 | 795 |
| size | 128 | 0 | 16166 | 1302 | 1558 | 1222 | 1528 |
| grammar | 117 | 11 | 94429 | 1002 | 1121 | 923 | 1089 |

Proxy sum (structural + iota + 2 reads) in the same paired sample:

| split | unit tree shared | weighted tree shared | exact steps | exact proxy |
|---|---:|---:|---:|---:|
| small | 10071 | 7865 | 10090 | 7543 |
| size | 28182 | 21130 | 29629 | 19660 |
| grammar | 19404 | 14724 | 19169 | 13580 |

On size-shift exact DAG steps save 21.6% vs shared weighted-tree selection, while increasing the earlier proxy by 40.2%. Cost optimization must target the actual circuit representation. Exact proxy and exact step objectives differ; there is no measured Nova performance claim. Next train a policy to choose expansion rules from the obligation mask, compare to the exact solver on uncapped cases and a bounded beam/A* control on capped cases, and separate generator families and evaluation custody before promotion.

The ZKL-08 actual Nova candidate is in zk-rusto-no-bitcoino PR #34.
