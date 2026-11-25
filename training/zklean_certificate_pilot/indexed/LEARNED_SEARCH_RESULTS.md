# ZKL-12/13/14 — learned shared-proof search gate

Status: **FAIL learned search efficiency**. Retain ZKL-10 ordered-height A* as the reference. CPU-only synthetic indexed-family exercises; this is not a general Lean proof-search result. Every policy uses the same admissible primary priority g+h, reopens improved paths and stops when the empty obligation mask is popped. Learned estimates are secondary keys only. Cap: 100,000 discovered states per solve. Every returned proof was replayed using explicit indexed rule instantiation and iota-normalization. All arms solved all cases and agreed on optimum cost.

## ZKL-12: exact residual-cost learning

80 training environments (seeds 70000–70079, limit 12), 2,000 exact DP labels, 193-parameter MLP, seeds 181/191/193, 40 epochs. Three fresh test distributions, 48 each (81000–81047 limit 12; 82000–82047 limit 24; 83000–83047 limit 24, shifted grammar with 18 sampled rules, 80% binary and offsets up to seven).

| split | depth tie | tree-cost tie | MLP 181 | MLP 191 | MLP 193 |
|---|---:|---:|---:|---:|---:|
| same | 328 | 316 | 344 | 474 | 367 |
| size | 786 | 767 | 934 | 1204 | 1046 |
| new grammar | 5752 | 5735 | 5907 | 6280 | 6327 |

Values are expanded states. All neural estimators also incurred inference overhead. Training MAE 0.554/0.558/0.545. Approximately 3.06 seconds including labels on the local CPU sandbox.

## ZKL-13: mandatory facts and disjoint premise lower bounds

For each atom a, let M(a) = {a} union the intersection over its rules of the union of the mandatory sets of that rule's premises. By induction in the strictly ordered grammar, every proof of a contains M(a). For an obligation set S, U = union of M(a) for a in S. Ordered-height applied to U is admissible. For a in U, each rule requires at least k(a) = min over its rules of the number of distinct premises outside U. Take the union V(a) of all these outside premises. Pairwise disjoint V(a) allow additive counting of k(a); a greedy disjoint packing yields a lower bound |U| + sum packed k(a). Take the maximum of this and ordered height.

| split | height | mandatory | packing |
|---|---:|---:|---:|
| same | 328 | 325 | 320 |
| size | 786 | 785 | 785 |
| new grammar | 5752 | 5733 | 5685 |
| fresh larger (84000–84047, limit 32) | 8372 | 8365 | 8349 |

All 192 cases solved by all arms. Packing reduces expansions 0.13–2.44% with no wall-time improvement. First three splits diagnostic reuse; fourth fresh. Exhaustive forward Dijkstra on 12 tiny environments, all 256 obligation masks per environment, confirms admissibility and cost agreement. 12 focused tests pass.

## ZKL-14: recurrent residual slack

120 training environments (71000–71119), 2,849 exact labels on reachable A* frontiers. Predict D(S)-h(S) with a nonnegative softplus secondary score. 193-parameter MLP and 147-parameter tanh RNN, hidden width eight, six local atom features and ten global features, seeds 211/223/227, 32 epochs. Fresh 48-root splits 85000–85047 (12), 86000–86047 (24), 87000–87047 (32 with changed grammar).

| split | depth | tree | MLP 211 | MLP 223 | MLP 227 | RNN 211 | RNN 223 | RNN 227 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| same | 821 | 814 | 905 | 869 | 887 | 885 | 856 | 884 |
| size | 1330 | 1333 | 1600 | 1433 | 1440 | 1492 | 1488 | 1548 |
| new grammar | 14788 | 14770 | 15686 | 15632 | 14848 | 14853 | 14823 | 14846 |

All eight policies solve all 144 fresh cases at the same cost. Training MAE RNN 0.248/0.253/0.249, MLP 0.278/0.277/0.274. Approximately 5.43 seconds training and label generation. On the largest split, baseline search 0.333 s, recurrent inference-inclusive search 1.77–1.86 s, excluding graph construction and verification. Absolute timings are platform-dependent.

Decision: no promotion, no paid GPU spend. Public exploratory seed ranges; no claim of preregistered sealed confirmation. The next useful gym needs a substantially richer action space or expensive verifier, where symbolic heuristics leave meaningful room for improvement. Do not weaken exactness to manufacture an advantage.

The local source/JSON evidence bundle is supplied separately in the conversation as zkl12_14_search_gate.zip with SHA256SUMS. This repository record is a results report; the new Python scripts are not included in this commit.
