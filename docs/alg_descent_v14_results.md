# ALG-DESCENT-14: lazy certified residual-cut refinement (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/a361d4108565640040bf9528981ab8969b8ad82d). Synthetic weighted acyclic Horn proof planning, no learned weights, private witness, cryptographic task or external novelty claim. Fresh 12 IID roots 910000..910011 and 24 OOD roots 920000..920023. Four policies (hmax, eager full cuts, lazy full cuts, lazy one-round cuts). All use V11 generator, V12 immutable proof paths and verified FF incumbent/completions, V13 residual cut bound. Cheapest legal rule per head, best-g dominance/reopen, independent lower bound, exact proof replay and complete unresolved frontier. Priority (g+b,b,g,serial); lazy rekeys preserve serial. Eager computes all rounds on admission including states that could be pruned cheaply. Lazy retains partial residual costs/distances and computes only when lowest frontier state is selected. Incremental arm raises bound by one cut prefix and rekeys or prunes; expansion requires completed cut bound. Snapshots stop at 0/4/16/32/64/128 expansions, selected-refinement cap 4096, never discard unresolved state. No cap hit (max OOD lazy full 138, step 545). All 144 root/arm cases executed seven times in randomized arm order; stable non-timing receipts identical. Offline reference after search.

## OOD results

| Arm | Mean capped expansions | Bound scans | Cut rounds | Full bounds | Coarse calls | Mean per-root median ms | Optimal certificates at 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| hmax | 30.333 | 2848.833 | 0 | 0 | 111.458 | 1.96484 | 23/24 |
| eager | 18.750 | 27825.708 | 373.167 | 77.500 | 77.500 | 6.81722 | 24/24 |
| lazy full | 18.750 | 12510.292 | 149.417 | 28.083 | 77.500 | 3.64424 | 24/24 |
| lazy step | 18.750 | 11106.167 | 128.750 | 18.708 | 77.500 | 3.60625 | 24/24 |

PASS primary lazy-step gate: 60.09% fewer bound rule scans (24/24 strict wins; paired mean difference -16719.54, descriptive 95% root bootstrap [-26849.46,-8881.08]); 47.10% lower root-average median runtime (23 wins, one loss; mean difference -3.211 ms, interval [-5.232,-1.674]); same expansions and certifications. Lazy full also saves 55.04% scans and 46.55% time. Incremental saves 11.22% scans versus lazy full but their measured times are close; one-round scheduling overhead is real. Against hmax, lazy step reduces capped expansions 38.19%, certifies one additional root but is slower, so joint secondary gate fails. Unfinished hmax root 920014 interval [25,26], optimum 26. All IID certify. All 144 runs have verified incumbent; 143 certify optimality. Capped counts include unfinished baseline, not mean unconditional time to certification. Mean peak OPEN 18.833 hmax, 10.75 eager, 15.292 lazy full, 15.708 lazy step. Partial residual state retained, no byte cap or physical RAM claim.

Mean normalized certificate width at expansion budgets:

| Arm | 0 | 4 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| hmax | .368388 | .189217 | .080757 | .034593 | .008992 | .001603 |
| eager | .287524 | .127105 | .043962 | .012522 | .003917 | 0 |
| lazy full | .368388 | .149521 | .064292 | .015126 | .003917 | 0 |
| lazy step | .368388 | .149521 | .064292 | .015126 | .003917 | 0 |

Deferred refinement weakens early-budget bounds; at 16 expansions all three cut arms certify 14/24 vs hmax 10/24. Post-hoc comparison: both lazy arms reproduce the exact eager expansion trace and incumbent cost history on all 36 roots. This is consistent with the conditional eager/lazy ordering equivalence in MathCan HORN_LAZY_REFINEMENT.md. Timing includes Python instrumentation and certificate generation; excludes graph construction, offline oracle and independent audit. Seven within-root repetitions, root-average medians, not a general speed guarantee. Bound scans count rule inspections, not all primitive costs. One round may add no immediate lower-bound improvement.

## Audit and replay

Independent set-based audit of all 36 roots: 44,213 exact Bellman states; 8,826 bound certificates, 16,755 charged cuts, 25,581 admissible bound prefixes (including coarse); 6,358 priority events (2,270 expansions, 4,088 refinements); 864 snapshots, 18,144 compared receipt fields, 4,925 incumbent proof steps (repeated snapshots). All pass. Every cut independently checked by deletion/unreachability, threshold membership, residual nonnegativity, delta, exact lower bound and offline optimum. Reconstruct admission, rekey, best-g, reopening, pruning, priorities, proofs and frontier. Timing and peak heap allocation not independently measured. Shared graph generator only. Source Git blobs verified: experiment.py 4a0a840094d708cc40f55ac2ad526c5811c8fc90; audit.py 6c3e0201dd5d61bc15237cbe2db7fda3636f7a23. Local SHA256 receipt 985c5a4be22b54c3764aa260a52f9676a56d60f792a21d93c504a0121c7ec0b5; audit 3be2b7520f16121b33eb03eb562afb8de18662c979614264d35da738cb592373. Raw archive contains source, pinned V11/V12/V13 dependencies, receipts, logs, hashes.

From experiment directory with Python/PyTorch (V11 import dependency), assertions enabled:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python experiment.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python audit.py

Next: test safe cross-state cut-prefix reuse and compact native residual storage, compare strongest optimized hmax and lazy step at equal actual work/time budgets. For learning, use marginal value of a cut round and verified incumbent improvement, preserving admissible pruning and frontier coverage. No native BPD/MMS, Lean check, equal-memory claim or cryptographic discovery.
