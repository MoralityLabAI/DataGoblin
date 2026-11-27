# ALG-DESCENT-10: exact bound-certificate routing (2026-11-27)

Prospective protocol: [97283d6](https://github.com/MoralityLabAI/DataGoblin/commit/97283d6acb673987264190bb06c809c924a56b94). Synthetic public-witness polynomial routing only; no factoring or DLP discovery.

## Design

Fresh train 32 roots (510000..510015, 511000..511015), IID 16 (520000..520007,521000..521007), OOD 24 (530000..530011,531000..531011). V7 family generator and 20 public features preserved. At each eligible nonterminal state add at most one product-valid but unbound equal-degree swap decoy, preserving exact product, all fingerprints, features, structural score and bucket. Deterministically shuffle. Every bank retains at least one bound admissible fallback. Labels, Bellman oracle, attempted verification and closure-cache admission all use the conjunction of exact product and indexed-child binding. Registry is trusted and available to the direct binding control; learned policies see only the original public feature vectors and simulated/actual rejection histories. This difference is deliberate. The generator's expected children are computed from known public witness factors; this is a routing task, not an unknown-factor problem.

32 train roots, 29,372 candidate/history examples, deterministic 20,000-example subset. 16 IID roots (23,058 examples), 24 OOD roots (251,718 examples). Three seeds 19/41/67 for logistic (42 parameters) and exactly parameter-matched one/three-pass models (409 each); 8 epochs AdamW .004. No BPD interventions. Exact prefilter compares every candidate with the trusted registry and still runs exact product and binding checks on the selected certificate. Product work is coefficient multiply-adds, charged for every product attempt and one per cache hit. Binding calls, coefficient equality tests and elapsed time separately recorded. Binding comparisons intentionally count every tested coefficient without early exit. Work cap 3*n^2, product calls 8*n. Other work excluded from coefficient meter; route times exclude prebuilt bank and prior generation. Single randomized arm order per root, descriptive timings. Privileged offline oracle minimizes successful work on full admissible banks with precomputed labels; excludes oracle construction time.

First local execution timed out during OOD evaluation before complete receipt; nine models and IID completed. Re-executed identical public development cohort with equivalent vectorized history tensor construction and independent integer NumPy convolution audit to reduce runtime; no seed, arm, budget or gate change. Original partial log preserved. The successful run completed all prescribed arms.

## Results

OOD no-cache root-average (model seeds averaged within roots):

| Arm | Closed | Mean product work | Mean warmed ms | OOD log loss | Brier |
|---|---:|---:|---:|---:|---:|
| structural | 24/24 | 1342.7083 | .55360 | — | — |
| bucket | 24/24 | 1091.9167 | 1.54670 | .45997 | .14478 |
| logistic | 72/72 | 1060.1944 | 2.62603 | .44671 | .14191 |
| one-pass | 72/72 | 1057.3056 | 3.20843 | .44401 | .14150 |
| three-pass | 72/72 | 1068.1111 | 3.85475 | .44922 | .14207 |
| exact binding | 24/24 | 959.75 | .63557 | — | — |
| offline DP | 24/24 | 955.50 | .38358 | — | — |

Binding prefilter performs 112.25 binding checks and 1320.42 coefficient comparisons per OOD root (mean .28213 ms measured binding time); all selected actions accepted, zero failed product attempts. Best learned arm one-pass saves 3.17% work vs bucket, fails 5% gate and wins only 6/24 roots vs logistic (3 ties, 15 losses; mean -2.89, paired root-bootstrap 95% [-25.67,+11.26]). Three-pass vs one-pass +10.81 work, 4 wins/3 ties/17 losses, interval [+4.31,+18.83]; loss +.00521, interval [-.00583,+.02078]. Exact binding vs one-pass -97.56 work, wins all 24, interval [-132.36,-70.07]. Exact binding is .45% above privileged oracle work. Direct binding uses stronger information; no learned superiority claim. One IID structural route exhausted budget; 599/600 total closures, no learned/control binding failures. OOD second family log loss bucket .41661, logistic .40759, one .41255, three .42327; two-child .50333/.48582/.47547/.47518 respectively. Public development roots not sealed.

## Feature collision

Audited 144,091 eligible candidate identities and 20,722 Bellman states across 72 train/eval roots. OOD 116,014 candidates: 12,684 product-invalid, 13,533 product-valid but unbound (includes 13,256 injected swapped twins), 89,797 accepted, zero bound/product-invalid. IID 11,841 eligible, 806 product-invalid, 1,342 unbound, 1,342 twins. Train 16,236 eligible, 889 product-invalid, 1,737 unbound, 1,737 twins. Total 16,335 injected twins. The original features are identical on each pair with no history, so any deterministic feature-only probability q has balanced pair loss -log(q(1-q))/2 >= log 2. Independent of model depth, this is a conditional information limit for this constructed paired subset, not a universal lower bound on all candidates or on policies that observe candidate identity, exact registry, or different histories. The measured OOD no-history twin losses averaged over seeds: bucket 1.06967, logistic 1.08508, one-pass 1.03333, three-pass 1.06504. Product commutativity does not ensure indexed-child identity.

## Cache contract

128 logical-byte LRU on binding control: OOD no cache 959.75 work, interval key 959.75 (0 hits), exact polynomial key 870.9167 (99 hits, peak 128 B, 9.26% work saving). Mean warmed ms .63557/.64680/.46173, respectively; host-specific. All close. Cache inserts only after accepted bound product and full recursive child closure; root/field/version scoped. Polynomial key establishes existence of closure for that polynomial, not a particular indexed witness or cost-to-go. Interval key preserves witness identity. No physical-memory or native MMS claim. Memory accounting uses len(poly)+16 or 4+16 logical bytes, excludes actual Python metadata, registry and bank.

## Audit and reproduction

From repository root:
    SRC=src/datagoblin/research/alg_descent_v10
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/experiment.py"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python "$SRC/audit_receipt.py"
    python -m pytest -q "$SRC/test_v10.py"

Run the separate existing certificate-contract regression too. Local 7 tests passed (4 new + 3 existing). Independent convolution and bottom-up DP audit checked all 144,091 identities and 20,722 states. Independent saved trace traversal replay checks arithmetic, binding, cache admission/hits, closure and counters for all 600 routes: 6,600 fields reproduced. It does not independently reproduce model inference, candidate selection or the generator. Source hashes matched local executed git blobs: experiment.py dad9f4bc567b41b9c30f5ab310afec501390a368, audit_receipt.py e71f6dfd0aab0e334782731e2f26a4c28ccf5ad4, test_v10.py 80729de23ec8d2378db15e1fa756056ee2688e7d. Raw replay archive includes dependencies/checkpoints/receipts and logs. Receipt SHA256 f70ed3601b3ce4ed98e8e7c8a983b040e1660994292283b5951d2f3dabda7485; models SHA256 f702216cf739b2a2cf7e05f29744a99b7f59f67320a3b786393ae42328445cc9.

## Decision

PASS acceptance alignment and arithmetic/trace audit; FAIL learned routing improvement gate. Do not run more BPD decomposition on this feature-deficient public-witness surface. An exact binding prefilter is a strong available control; a learned router should target resource allocation or proposal selection after exact checks. For a meaningful discovery gym, design a separate public synthetic proof-planning problem where candidates are untrusted, exact verification certifies statements, and the solution witness is withheld from proposal features; use exact small-instance cost-to-go and matched search/candidate budgets. Keep the verifier and cache contract separate from the proposer. The existing native C++ cache engine has not been migrated to the new V10 contract; neither native BPD/MMS nor Lean verification was performed.