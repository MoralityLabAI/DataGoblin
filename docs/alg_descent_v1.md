# ALG-DESCENT-01: certified polynomial routing proxy

Motivated by [Cogentic's pinned theorem statement](https://github.com/google-research/google-research/blob/4a7bc8ab0bc770e71a259b7966c500496b99010a/cogentic_research_papers/discrete_logarithm/README.md). This is an independent synthetic certificate-routing environment, not a reproduction of that paper or a factoring solver.

Run: python src/datagoblin/research/alg_descent.py > receipt.json
The stdlib script includes exact product checking and negative fixtures. The public ordered witness consists of linear factors. All policies have identical factors. Monic coefficient arrays are little endian and checked for canonical field residues, correct degree sum, strict shrinkage and exact product. Terminals have degree one. Allowed policy arities are 2,3,4,8; the verifier accepts any 2–8 children. The cost meter counts only schoolbook coefficient multiply-add operations on successful certificates; invalid fixtures separately demonstrate rejection. Proposal construction, DP computation and memory are excluded.

Development run: five primes, degrees 8/16/32/64/128, four deterministic seeds per cell, 100 roots, 300 rollouts. Costs are identical within degree by construction; seed replicates test algebraic correctness rather than statistical variation. No model trained.

| Degree | Binary cost | Max arity cost | DP cost | Depth binary/wide/DP |
|---:|---:|---:|---:|---|
| 8 | 78 | 72 | 70 | 3/1/2 |
| 16 | 246 | 240 | 220 | 4/2/2 |
| 32 | 798 | 760 | 746 | 5/2/3 |
| 64 | 2718 | 2664 | 2580 | 6/2/3 |
| 128 | 9726 | 9672 | 9450 | 7/3/4 |

Locally executed source SHA256 6510d6cbe9b4b3a77b571c8204e99ec3ce744771ce23a0cea5590a430ceb7a17. Stdout JSON SHA256 5ef7b4e3fcdac30cca257c5bf7ecc321a36d3d6b08447ac8dec0964872e5de09. Python's seeded generator is used; reproduce with the same Python implementation/version if comparing byte hashes. DP optimality is restricted to the listed balanced contiguous actions and additive meter.

Next: build a heterogeneous public candidate bank with exact identities and invalid/costly decoys, record proposal and verification costs for all attempts, add terminal/closure receipts and a fixed budget API, and test learned recurrent routing against exact small-instance DP and fixed baselines under matched budgets. BPD/MMS ablations should use identical candidate banks and explicit RAM caps. Degree-only features already suffice in v1, so v1 is a sanity check and has no evidence of model learning or algebraic discovery. Do not use any private factors as model features in a future discovery task.
