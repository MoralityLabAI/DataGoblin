# ALEPH-STATE-08 | counterexample-guided mathematical repair

2026-11-25. CPU-only public diagnostic. Keep A and both paid slots on hold. No neural training, native Aleph checkpoint changes, or learned LDT promotion. Follow-up to STATE-07.

## Design

Use STATE-07's 24,224 distinct base functions on (Z/8Z)^2 with scalar output mod 8. Expand by affine + d*indicator(x+y>=8) and affine + d*indicator(x=y), d=1..7, deduplicated against the base and each other: 7,168 new distinct functions. Sample 48 each from base, new threshold, new diagonal, base plus one output defect, base plus two output defects, and an alien periodic template not in the expanded grammar and at Hamming distance >2 from all its members. Six families, 288 unique truth tables; exclude prior STATE-06/07 closed-panel IDs for base truths and STATE-07 open-panel semantics for all truths. Seed 2026112514. Panel hash 49485351e9bcf3ec8b8ab0428c9c060f8d6aee6994000f15c7b0de3427553159.

Policy arms: frozen base, sparse repair of at most two coordinates of a base candidate, structured expansion, combined repair; all four use the sample32 initial witness policy. Combined exact entropy and random initial witness arms are controls. Raw table acquisition is an additional control. Eight initial distinct witness queries; initial acquisition only uses exact-consistent base candidates, random after singleton/empty. Sample32 draws from the class-balanced posterior with replacement when >32 survivors, exact otherwise. No cached common first query in this study. Subsequently pick the shortest consistent candidate, verify against the truth at unobserved coordinates in a fixed random order until the first counterexample, cache every inspected coordinate (including matching outputs), and re-propose; if no candidate exists, abstain. Snapshots at 8/16/32/64 reads are cumulative, not separate independent trials. A scan session carries over budget checkpoints. At final read a new candidate consistent with all observed coordinates is certified even if there is no further scan needed. A no-candidate episode can stop early. Raw table arm gathers all 64 labels. Oracle reads, separate query and scan calls, comparisons for acquisition, candidate-coordinate comparisons for mismatch updates, counterexamples and proposed lengths are counted. Argmin and miscellaneous CPU costs not included in candidate cells; CPU time separately measured. No held truth enters proposals other than observed coordinates; the offline exactness score compares complete truth tables. Independent exhaustive truth-table equivalence is used as the finite certification criterion.

## Mathematical notes

Let D be the observed coordinate set, H the base grammar, E its extension, and d_D(f,h) the number of observed disagreements. Then the sparse k-patch envelope is H^(k)={g: min_h in H d_X(g,h)<=k}. A partial observation admits a completion in H^(k) iff min_h d_D(f,h)<=k: necessity by restriction, sufficiency by patching each observed disagreement and retaining h on the rest of the domain. Thus min_h d_D is a monotone lower bound on the full repair distance as D grows. When it exceeds k, the entire sparse envelope is rejected without enumerating possible patches. Similarly E is rejected when its minimum observed mismatch count is positive. The combined envelope is H^(2) union E. Its rejection is sound under noiseless observations; it is not a rejection of every possible mathematical hypothesis.

Candidate code length uses two tag bits, 15-bit base index, or four grammar revision bits plus a 13-bit extension index, or base index plus two bits for patch count and nine bits per patch (six coordinate, three output). Raw table is 194 bits including the tag. Minimum length is within this chosen coding and proposal family, excluding shared dictionary cost; it is not Kolmogorov complexity or an optimal code. Extension templates are supplied by design, not autonomously discovered. At a fixed partial D, choose the minimum observed mismatch base candidate and its minimum necessary patches; no exhaustive enumeration of unseen arbitrary patches is required.

Unconditional certification for arbitrary black-box functions on a finite domain cannot be guaranteed with fewer than |X| distinct noiseless point observations: if a point remains unseen, modify its output, producing an indistinguishable alternative. An equivalence oracle returning a counterexample or a valid proof could alter the cost model, but our pointwise verifier counts every checked point. The full certificate here is a table agreement receipt. Any intermediate exact reconstruction is an offline score and not a certificate. A false conjecture consistent with D is possible even with a unique candidate. Every accepted final candidate has all 64 distinct coordinates observed.

## Executed results

|Policy|Exact @8|Exact @16|Exact @32|Certified @64|Counterexamples total|Scan sessions total|
|---|---:|---:|---:|---:|---:|---:|
|Frozen sample32|48|48|48|48|110|158|
|Patch sample32|52|64|80|144|450|593|
|Expand sample32|118|141|144|144|111|255|
|Combined sample32|122|157|176|240|249|488|
|Combined exact|130|160|176|240|240|479|
|Combined random|109|158|180|240|284|523|
|Raw table|0|0|0|288|0|0|

All denominators 288. All successful compact candidates are certified only at 64 distinct reads. The combined sample32 arm recovers 48/48 in each representable family and abstains on 48/48 alien. It has 165,87,64,0 incorrect unverified proposals at budgets 8,16,32,64. Zero false certified proposals by construction and checked by assertion. The sample32 arm spends 15,917 oracle reads (2,304 direct queries, 13,613 scan checks); all 240 successes consume 64 reads each, while alien episodes consume 557 total before abstention. Frozen/patch/expanded solve only their representable subsets; raw table solves all 288 at 18,432 reads. Do not interpret fewer aggregate oracle reads as a gain on matched successful coverage; abstention accounts for the savings. On successful combined sample32 cases, mean code length 24 bits (base 17, threshold 19, diagonal 19, sparse1 28, sparse2 37); raw 194, ratio 8.08x conditional on the shared dictionaries and successful cases. A raw table fallback would retain full coverage, but was not evaluated as an additional adaptive arm.

At 64-budget sample32 vs exact vs random combined arms: acquisition cells 2,455,049 / 509,547,403 / 0; filtering cells 499,666,464 / 499,101,408 / 499,917,600. This counts mismatch updates for every grammar member per observation (including eliminated candidates), unlike STATE-07 survivor filtering. Exact recomputes the initial score on each case; amortizing its first score across all cases removes much of the acquisition overhead. Candidate counts are not FLOPs. Random initial acquisition achieves four more exact proposals at 32 observations than sample32 and matches certification at 64; sample32 reduces scan sessions by 35 vs random. No clear general acquisition winner here; if scan calls are costly relative to candidate comparisons, sample32 is a plausible default. Measure under explicit cost ratios.

CPU process seconds per arm (excluding panel construction/imports): .454, .685, .735, .952, 4.596, .797, .034; total 8.25295. Zero paid credits. No confidence intervals: one fixed diagnostic panel, repeated policies on the same functions, no independently sealed registration.

## Next

STATE-09: transfer from independently verified operators to a new composed task family, e.g. multi-register maps or dynamic graph transitions, and compare an exact symbolic executor, learned proposal, compact repaired representation, and raw table fallback under matched verifier budgets. Measure the total storage including grammar description, query costs, and downstream composition. For Lean, replace the pointwise equivalence check with kernel-checked proof obligations; no extrapolation from sparse finite tables to proof soundness. Historical 14M checkpoint files remain unavailable; do not authorize 1B training.

## Replay

Run python state08.py from this directory (requires state07.py, numpy 2.3.5 and its torch imports). Source SHA256 be2ba902660add1d8c26a9eb12081149ffc59f0a6b1ed2ff112fb50428aa432c; committed blob f50217b43adfa52130d1ac266f83327c39c6d8db matches executed file. Local full JSON SHA256 c90940cf335c2ff9d14688bda013f23c8ac0d19c8f477d111e4c6ae8c2225506. Includes every policy/family/operator/budget outcome and all measured costs. Timing-dependent receipt hash changes on rerun.
