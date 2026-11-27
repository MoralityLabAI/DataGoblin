# ALG-DESCENT-13: admissible mandatory facts and goal-threshold cost partitions (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/d11b750c59596bbb5140291dabd779fd56118b9a). Synthetic weighted acyclic Horn proof planning only. No neural training, private proof witness, cryptographic task or external novelty claim. V11 graph generator, V12 immutable proof paths, scope hash, exact rule-ID proof verification and direct FF completion. Fresh 12 IID roots (810000..810011), 24 OOD (820000..820023; n=16/17/18; family rotated by block). Four bounds, common best-first ordering by g+b, ties smaller b then g then insertion order. All actions add one fact with positive cost; cheapest applicable rule for a head dominates other applicable same-head rules. Keep best-g, reopen on improvement, prune only using verified incumbent and admissible bound. All unresolved OPEN retained. Exact Bellman reference computed after each root's search.

## Bounds

A: topological hmax. B: maximum of hmax and mandatory-fact charge, from intersection over alternative rules of unions of prerequisite mandatory sets, plus head. C: residual goal-threshold cuts. With current residual costs, H=hmax(goal); Z is the ancestor facts with hmax >=H. For H>0, every proof enters Z using a rule whose head is in Z and premises outside. All such rules have positive residual costs. Charge the smallest residual cost of this cut and subtract it from each member. Repeat until H=0; take maximum of accumulated charges plus residual hmax across prefixes. D: same cut sequence but combine residual hmax with residual mandatory-fact cost at every prefix. At least one positive residual rule becomes zero per iteration. See MathCan HORN_COST_PARTITION.md for paper proof. No claim of consistency; search permits reopening.

## OOD results

All 144 searches return verified incumbents. 142 certify optimality by 128 expansions; the two unfinished searches (hmax and mandatory on root 820023) have interval [27,28] and optimum 28. Capped expansion means include the 128 expansions spent by each unfinished baseline; they are not unconditional time-to-certification means.

| Bound | Gap 0 | Gap 4 | Gap 16 | Gap 32 | Gap 64 | Gap 128 | Certified 16 | Certified 128 | Capped expansions | Mean elapsed ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hmax | .34172 | .16323 | .03993 | .01671 | .00595 | .00149 | 15/24 | 23/24 | 19.6667 | 1.3552 |
| mandatory | .33851 | .16323 | .03993 | .01671 | .00595 | .00149 | 15/24 | 23/24 | 19.6250 | 1.7488 |
| cut | .27270 | .11171 | .02134 | .01059 | .00298 | 0 | 21/24 | 24/24 | 13.1667 | 4.4766 |
| combined | .26950 | .11080 | .02134 | .01059 | .00298 | 0 | 21/24 | 24/24 | 13.0417 | 5.3530 |

Gap=(U-L)/offline optimum. All 12 IID roots certify under every arm. Combined minus hmax capped expansions -6.625, wins/ties/losses 20/4/0, descriptive paired root-bootstrap 95% [-9.875,-3.9167]. Reduction 33.69%; on the 23 jointly certified roots mean 9.3478 vs 14.9565. Cut minus hmax -6.5, 20/4/0, interval [-9.6667,-3.875]; jointly certified mean 9.4783 vs 14.9565. Both reduce mean gap16 by .01858845 (46.55%); 8/16/0 wins/ties/losses, interval [-.0301824,-.0084342]. Registered expansion gate for combined passes (>=25% capped reduction, >=16/24 wins, no fewer certifications). Gap gate lacks 16 wins. This is a small public development cohort.

Mean OOD bound calls 64.75/64.75/46.0417/46.125; bound rule inspections 1607/3457.58/17258.96/19549.96. Bound computation ms .7753/1.1748/4.0172/4.8283. Cut rounds 241.71/241.38 per root in C/D. Mean transition rule probes 695.21/693.71/466.75/462.04. Mean peak OPEN 13.125/13.1667/8.0/7.6667; mean peak best table 64.7917/64.7917/46.0833/46.1667. Separate counters have different unit costs. Bound scans count inspected rules in distance recurrence, mandatory set/cost and cut enumeration/minimum; not every primitive operation. Single randomized arm-order timing includes bound construction, certificate recording, proof search, FF, checks, instrumentation; excludes graph construction, offline reference and independent audit. Combined is slower than hmax. No physical RAM or wall-clock advantage. Pure cut is the practical next candidate.

## Independent audit

36 roots; 45,369 exact Bellman states. 6,199 bound certificates compared to exact continuation values, 13,018 cut charges independently checked by deleting the cut and testing goal reachability; nonnegative residual costs, exact threshold cut membership and residual hmax independently recomputed. 3,096 mandatory sets verified by disabling each fact's derivations. Every expanded state independently replayed including exact priority order (1,827 events); 864 snapshots, 7,776 receipt fields and 4,785 incumbent proof steps (repeated across snapshots). All assertions pass. The audit uses the same public graph generator but separate set-based successor enumeration, Bellman recurrence, reachability and proof replay. It does not verify physical allocation or timing. No Lean check, no native BPD/MMS.

## Reproduction

From repository root:
    SRC=src/datagoblin/research/alg_descent_v13
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/experiment.py"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/audit.py"

Requires Python with V11/V12 dependencies (PyTorch imported by V11 module; V13 trains no model). Run with assertions enabled. Replay ZIP includes the pinned V11/V12 dependency files and saved receipts. Source blob hashes verified against executed local files:
- bounds.py 50a07f34754ff76690bbe4f55250ae627ee9b822
- experiment.py 2b4c185dbf467d2d310b7659b4eeaac115900513
- audit.py 5057539f8b602ba1aeca35bf7639c20c84e01198

Local SHA256:
- bounds.py 0c4da3953e3887a689672c211df3929c6614c9634a904faf18f0d23b060847f0
- experiment.py 09e07ab8880d082329aaeff1feb3c1d10ad8d010e823c18e746f9b873340b9d3
- audit.py b329a3d91f5d860863309ebb8d752c42c33bd88ac91db6b84a60dca1207c1e63
- receipt.json e3907f42efdbc8d8e6f73685c459f8d8d9c55e04bac93a27f76a9fb802e39d7d
- audit.json eb7955d57d6041b5352c4a1d5ac1f098571702fafb91f143fdff07daa51728a9

Next: lazy certification by initially evaluating hmax for every generated state and computing the stronger cut bound only when a state threatens the frontier lower bound. Retain every unresolved state with its current certified bound, rekey on refinement, and independently audit all refinements and interruption snapshots. Count bound work and latency, not merely expansion savings. Compare eager/lazy cut and hmax on fresh roots. Native implementation and optional verified cross-state cut reuse can follow. No neural intervention until there is measured headroom beyond the strongest symbolic certificate architecture.
