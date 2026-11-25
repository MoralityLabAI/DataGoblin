# ALEPH-STATE-14 | Charged diagnostic probes and adaptive proof routing

2026-11-25. CPU-only continuation of STATE-13. Keep direct Aleph A and both paid slots frozen. No native model training. Lean is unavailable; the certificates are checked only in the restricted Nat/List rewrite grammar and by denotational normalization.

## Design

Regenerate STATE-13 and exclude its exact equations and reversals. Regenerate its held templates at seeds 2026112520, 2026112521 (training) and 2026112522 (test). Each generation inherits the STATE-11/12 exclusions. Add a global transformation family: distributive expansion of products of two sums in training and three sums in test, and distribution of composed list maps/reversal across four blocks in training and five in test. Seed 2026112620 for global generation. Reject exact pair and reverse-pair overlaps; one held positive is removed as a duplicate. Semantic overlap and shared primitives are possible. Public, diagnostic split, not sealed mathlib theorem families.

Training: 256 positive and 88 negative equations. Held: 127 positives and 44 negatives, across long rewrites, deep contexts, independent holes and global transforms. All policies use the same denotational equality gate, which rejects the 44 synthetic negatives. Successful certificates are replayed against the rewrite rules and each intermediate expression's denotation. The gate is not a general Lean theorem prover.

Every policy except the global-search-first control attempts the STATE-13 congruence specialist first, with its original bounded search cap. The shared global-search trajectory provides snapshots at four and 64 expansions; when a probe succeeds early, stop. Continuing the same deterministic trajectory charges only the remaining work, rather than restarting it. The normalization probe runs at most eight explicit rewrite steps per endpoint; if full normalization is subsequently chosen, it restarts from the original terms and the entire restart is charged. Full normalization is capped at 512 steps per endpoint. All recorded attempts are evaluated offline and routed using only the information available at that point. Search and normalization may both fail; positive failures incur a dominant penalty.

Work is the STATE-13 proxy (schema match counts, visits, expansions, edges and normalization steps). Cost is work + 32 times successful certificate length. All attempted probes and failures are included. The proxy excludes some semantic, matching, feature and verification operations; it is not a Lean kernel-call or FLOP count. In the search snapshot timing, final certificate verification follows the timestamp. Report solver process-time sums as descriptive offline accounting rather than controlled deployment measurements.

Static controls: decomposition/search/normalization; decomposition/normalization/search; global search/normalization; dual probes followed by search or normalization. Symbolic single-probe routing chooses normalization if the fraction of initial syntactic goal distance eliminated by the search probe is at most a train-selected threshold. Dual-probe routing additionally bounds observed normalization growth. Both thresholds are chosen from fixed grids on training only: distance progress .75, growth 100 for dual, and .75 for single. The MLP takes 33 features (26 prior structural features and seven probe observations); three 1,154-parameter fits, seeds 337/347/349, 400 full-batch AdamW steps, LR .003, regress log-relative costs of the two continuations. Training uses only the 72 residual positive tasks where decomposition and both probes failed. Held residual count 44. Neither learned nor symbolic routing sees held full-attempt costs. A retrospective oracle selects among seven static/probed choices, using held outcomes; it is a headroom diagnostic.

## Primary results

All 11 policy rows (including oracle) solve 127/127 positives, with 0/44 false accepts.

| Policy | Work | Steps | Cost |
|---|---:|---:|---:|
| Decomposition then search |929604|2685|1015524|
| Decomposition then normalization |271327|3699|389695|
| Global search first |992624|2745|1080464|
| Dual probes, continue search |956916|2695|1043156|
| Dual probes, normalize first |356456|3694|474664|
| Dual symbolic adaptive |367431|3580|481991|
| Single symbolic adaptive |334492|3578|448988|
| Learned 337 |390071|3507|502295|
| Learned 347 |419364|2953|513860|
| Learned 349 |325629|3029|422557|
| Retrospective oracle |231163|2953|325659|

The static normalization-first continuation is also the strongest of the five checked fixed controls on training (509914, vs 1194231 for decomposition/search, 1348067 global search, 1169171 dual-continue, 616342 dual-normal), all with full training coverage. On held, it reduces weighted cost 61.63% relative to STATE-13's ordering applied with the new common caps. This is a distribution-specific solver-order result. Decomposition solves 72/127 directly; normalization solves all remaining 55; global search solves 19/55. After decomposition and the two probes, 44 cases remain. Probe overhead exceeds the improvement on this panel. Best learned arm costs 8.43% more at weight 32; no promotion. Oracle is 16.43% cheaper than the static winner, much less headroom than the preceding study.

| Held family | Positives | Decomposition/search cost | Decomposition/normalization cost |
|---|---:|---:|---:|
| Long |31|270373|225335|
| Deep contexts |32|10860|10860|
| Two holes |32|11957|11957|
| Global |32|722334|141543|

Post-hoc sensitivity (fixed policies, no retraining): at step weights 0,32,128,512, static normalization-first costs 271327,389695,744799,2165215. Learned 349 costs 325629,422557,713341,1876477. At 128 and 512 it beats this static control by 4.22% and 13.33% respectively, but the weight and seed are selected retrospectively and actual verification time is not measured. The break-even weight is (325629-271327)/(3699-3029) = 81.0478. Freeze a genuinely measured certificate-check cost and fresh panel before testing this hypothesis. No claim of statistical significance, generalized learned allocation or actual Lean speedup.

## Mathematical interpretation

Let A be a choice of continuation, Z the observations from a bounded probe, P its incremental work cost, and C the remaining solver cost including verification. The expected value of perfect optimal choice after observing Z, relative to a fixed optimal continuation, is
    min_a E[C(a)] - E_Z[min_a E[C(a)|Z]].
A probe is worthwhile only if this reduction exceeds its incremental cost in an accounting model in which the probe's residual state can be reused; if the continuation restarts, charge the lost work as well. The conditional-min expression assumes the same continuation options and remaining-cost convention; when probing itself changes available states, compare full policy costs from the same initial state. A model's prediction error adds decision regret. STATE-14 is an empirical counterexample to the idea that extra cheap observations automatically improve total routing cost.

The positive result is a symbolic routing correction for global transformations. The next useful research gate is measured Lean replay, followed by a prospectively frozen theorem-family/cost cohort. Only if real proof checking is expensive relative to solver work should we prioritize learned routing with certificate-length sensitivity. No new architecture or paid training authorization.

## Receipts and replay

Run from training/aleph_next/state01:
    python state14.py --out /tmp/state14

Requires the preceding STATE-11 to STATE-13 modules, CPU PyTorch 2.10.0+cpu and NumPy 2.3.5 (as available in the sandbox). The run used 6.28081985 CPU process seconds including panel generation, attempts, training and archive compression, excluding final output writes; three model fits used .1803/.1882/.1917 seconds. A separate replay checked all 332 stored held successful primitive certificates (7,216 steps) after decompressing JSON and restoring tuples. The replay is not timed in that total. Zero paid compute. Source SHA256 135333addeba20023fb5082cc6cb3309480e74921ec173c54b5f4acc96143f50, exact committed blob cf74bbae5b39b3125ced858fe687d8d3fad46550. Local full receipt SHA256 f460f7f09c3387b1fcb653cb633a18f6762f25c9ad419f52984ab2a7110af54a; gzip archive SHA256 ae83d64ed4257001543093e7d1ea4caaab16c620276d3fab8f6f13e17005ca90. Timing-dependent receipt hashes change on rerun.

No kernel-verified labels admitted. Historical 14M checkpoints remain unavailable.
