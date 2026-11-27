# ALG-DESCENT-12: certified anytime Horn planning (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/91f863e0a0cda3bf7a76a947fbef939818de6cf0). Synthetic public weighted acyclic Horn proof planning. Frozen V11 checkpoints, no retraining or teacher values at inference. Fresh 12 IID and 24 OOD roots. All six priority policies use a verified constructive FF incumbent, exact proved-set dominance, immutable proof paths, reopen on cheaper arrival, and independent admissible hmax pruning. Snapshot budgets 0/4/16/32/64/128. Four symbolic policies plus three seeds each for MLP and tied ordering, 360 searches. Every returned incumbent is verified; learned priority never supplies a lower bound.

## OOD results

Normalized certified interval width (U-L)/offline optimum, root-averaged:

| Ordering | 0 | 4 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| hmax A* | .372635 | .169327 | .040661 | .011834 | .001488 | 0 |
| hmax + FF tie | .372635 | .160994 | .040661 | .011834 | .001488 | 0 |
| FF | .372635 | .206415 | .099105 | .041207 | .005952 | 0 |
| weighted FF | .372635 | .212470 | .114229 | .062585 | .017770 | 0 |
| MLP | .372635 | .179025 | .073892 | .022090 | .002976 | 0 |
| tied | .372635 | .180074 | .073269 | .021363 | .002976 | 0 |

At budget 16 hmax and hmax+FF tie for strongest symbolic ordering. Both learned models fail the registered gate: zero root wins, 16 ties, 8 losses; MLP-minus-hmax mean gap .033231, descriptive paired bootstrap 95% [.012347,.056736]; tied-minus-hmax .032608 [.012434,.054884]. All three have U/opt=1 at 16, so the residual interval difference is lower-bound certification. At 4 expansions MLP and tied have lower mean incumbent ratio (1.01054/1.01077) than hmax (1.03290), a descriptive early-incumbent observation. All 360 searches certify the exact optimum by 128. OOD certification expansions average 17.75/17.75/19.333/21.125/18.0/18.0 for hmax, hmax+FF, FF, wFF, MLP, tied. End-of-run mean elapsed milliseconds 1.126/1.204/1.195/1.237/2.886/3.331 respectively; single randomized-order run per root/arm, includes search and instrumentation, excludes graph construction/oracle and independent replay. No neural speedup.

OOD mean peak OPEN: 11.5 hmax, 9.625 tied; mean peak best-state table 69.17/69.5, respectively. Not an equal-memory experiment: queue, traces, heuristic caches and Python object overhead also consume memory. No eviction implemented. Counters distinguish transition probes and heuristic/inference time; no equivalence of a rule probe and neural operation asserted.

## Independent audit and retrospective contour diagnostic

Independent set-based replay recomputes legal transitions, dominance, the FF incumbent construction, hmax lower bounds, scope and exact optimum. 48,425 exact reference states, 5,230 expansion events, 2,160 snapshots, 15,120 compared fields, 12,467 incumbent proof steps (repeated across snapshots); all match. No priority/inference replay. A second retrospective diagnostic checked 181,562 deduplicated legal successor edges for hmax consistency, all 360 traces for coverage at optimal g of every strict-contour state, and 180 malformed proof fixtures (all rejected). These are development checks on spent roots. OOD mandatory strict-contour mean 17.6667 states; A* mean extra .0833, equals mandatory count on 22/24 roots, zero expansions above optimum f. This identifies a near-tight expansion lower bound for this particular certification architecture, not general search complexity. See MathCan companion HORN_STRICT_CONTOUR.md.

## Reproduce

Run V11 to obtain checkpoints, or use the replay ZIP with frozen models.pt; pass its path to anytime.py. From the experiment directory:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python anytime.py models.pt
    python audit.py
    python contour.py
The scripts import V11 proof_planning.py, supplied in the archive and available in the sibling repository directory. Python/PyTorch CPU, one Torch thread. Do not disable assertions with python -O; this is a research verifier, not a hardened service.

Verified source Git blobs: anytime.py 3a58a1924321bf47b9ee44e2e47a4fe8c4085a6a; audit.py f034de8c5e542074eec7b670f85ba59ec5abb326; contour.py d0a0bc4c0812a76f52ba8abce03acab4d5ba2129.
SHA256 frozen checkpoint fc394c5fda4231ab9a3c26325ab87268870ae8f461f1c59a91ea719cd249ede0; receipt 7eaaac874fd785a64651ff4c02d358434baee0f8250fb825ab64ef62684ead57; audit 403c5c48fa4be4b9cbf85f054e53d00e906b6efec440c284ead148ef893667ef; contour f35b5ef7eee8f102fc6e43dfe28ed57dcf0778a17a99f65bc2f00d53f8d2d7bc.

Decision: PASS correctness of certified intervals, FAIL learned ordering improvement. Next experiment should improve admissible bounds (mandatory-fact/cost-partition lower bounds) and count strict contour before optimizing priorities. Keep learned incumbent search separate from certification. No native BPD/MMS, Lean verification, cryptographic discovery, physical memory reduction, sealed test or general proof-search speedup claim.