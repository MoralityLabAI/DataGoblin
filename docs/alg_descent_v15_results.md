# ALG-DESCENT-15: indexed cut boundaries and cross-state transfer

Prospective protocol: [85cc2cc](https://github.com/MoralityLabAI/DataGoblin/commit/85cc2cc3b011ea9804cd599348bea45128cdc5f2). Synthetic weighted acyclic Horn proof planning, no training or cryptographic discovery. Fresh 12 IID roots 1010000..1010011 and 24 OOD roots 1020000..1020023. Four arms, seven randomized-order repetitions, 1,008 executions. All use V14 incumbent, frontier, exact state dominance, and lazy refinement schedule. Offline Bellman optimum evaluated after search. Exact index precompiles head and premise incidence bitsets, memoizes boundary sets per root/zone, and updates only affected head descendants in topological order. Transfer additionally considers up to eight recently created distinct zones from other states, using a zone only if disjoint from current known facts and its current positive residual minimum strictly exceeds the current threshold cut charge. No transfer of bound scalars. Preserve exact cut order for indexed baseline. No physical memory cap.

## OOD results

| Arm | Capped expansions | Distance scans | Boundary scans | Minimum scans | Total scans | Median time (root-average ms) | Certified at 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmax | 28.125 | 2292.292 | 0 | 0 | 2292.292 | 1.605027 | 23/24 |
| lazy scan | 21.292 | 5750.542 | 4903.500 | 744.458 | 11398.500 | 3.676570 | 24/24 |
| indexed | 21.292 | 4366.083 | 0 | 744.458 | 5110.542 | 3.288541 | 24/24 |
| transfer | 19.167 | 3853.292 | 0 | 3156.125 | 7009.417 | 3.158889 | 24/24 |

All 144 root/arm searches returned verified incumbents; 143 certified optimality. Unfinished hmax root 1020011 interval [32,34], optimum 34. All IID certify. Counts capped at 128 expansions, not unconditional time to certification. No refinement cap hit. Indexed exactly matches scan-lazy expansion/refinement traces, bound certificates, incumbent proofs and snapshot lower/upper bounds on all 36 roots. Primary gate PASS: scans decrease 55.1648%, 24/24 OOD wins, paired mean difference -6287.96 with descriptive bootstrap 95% interval [-9681.79,-3741.29]; median time decreases 10.5541%, 20 wins/4 losses, mean difference -.388029 ms, interval [-.696432,-.169477]. Counts exclude one-time index construction and bitset unions, timing includes them. Indexed OOD 2513 hits and 809 misses in root-local boundary table, mean 33.71 entries, 222.92 rule IDs and 503.33 incidence unions. Counters do not imply physical memory or equivalent operation costs.

Secondary gate vs hmax FAIL: 24.2963% fewer capped expansions and one more certificate, but 104.89% higher elapsed time. Transfer vs indexed FAIL: exactly 460 vs 511 total OOD expansions, 9.98043% reduction (below preregistered 10%); 16 wins/4 ties/4 losses, paired difference -2.125, interval [-4.75,1.0417]. Mean time decrease 3.9425%, 12 wins/12 losses, difference -.129651 ms, interval [-.524874,.291514]. Scans increase 37.1560%; transfer selects 581 nonthreshold cuts in OOD and tests on average 414.42 candidates/root. This is a suggestive reduction in expansions, not an accepted performance result. No threshold retuning.

Normalized certificate gaps at budgets 0/4/16/32/64/128:
- hmax: .405076/.242629/.082179/.029949/.006886/.002451
- scan-lazy and indexed: .405076/.222993/.059160/.015478/.002451/0
- transfer: .405076/.197607/.049095/.010096/.007353/0

Mean peak OPEN 17.04 hmax, 15.92 scan/indexed, 14.88 transfer (logical counts). Timings instrumented Python, seven per-root randomized arm repetitions; graph, exact offline reference and independent audit excluded. No native BPD/MMS, physical RAM or general speed claim.

## Audit

36 roots; 53,692 independently enumerated Bellman states; 8,003 bound certificates, 10,312 cuts (verified by deletion/unreachability), 18,315 bound prefixes, 12,676 priority events (2,364 expansions and 10,312 refinements), 864 snapshots, 26,784 compared fields, 5,273 incumbent proof steps; all pass. Independently reconstruct all 3,200 transfer-policy selection rounds and counters using sets and full residual distances. Directly compare implemented incremental update against independently calculated full distance vectors for 6,756 cut rounds (114,406 coordinates), all match. Audit uses same public graph generator and production update function only for that direct comparison; reference distances, priorities, certificate validation and proof replay are independent. Timing not independently measured. All non-timing outputs identical across repetitions.

## Reproduction

Run with assertions enabled and Python/PyTorch (V11 import dependency):
    SRC=src/datagoblin/research/alg_descent_v15
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/experiment.py"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/audit.py"

Archive also supports running experiment.py/audit.py directly in extracted directory with V11/V12/V13 dependencies bundled. Source blobs match executed files:
- indexed.py ac3c7dd0cc2151c675b2526b2d0deeea972c0d3f
- experiment.py 71bd391b52cd924ad622d77abfa3ede58be731de
- audit.py 21c5491bcce70661664a9265c23114e9012fb6a1
Saved receipt SHA256 1b1166d98cbc541dd92c0c3cdb0886db01389296a24cd5dfc98212835b0acdf9; audit 2014038d5ee768fe1647a94cf416fa8b86e4596deb8a9c47fb128c2d93af1a2e. Timing-containing receipt run-specific.

Next: exact state-relative cut index in a compact native implementation, matching hmax and indexed search traces, with actual end-to-end memory and latency. Before further BPD, test whether selecting from reusable cuts can beat the indexed symbolic policy under matched total work; greedy maximum current charge has no optimality guarantee for cost-partition strength. No Lean verification.