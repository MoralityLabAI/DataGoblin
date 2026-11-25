# Direct A now has its first 10M positions

**The five-lane corpus gate passes for the first checkpoint. Hold the accelerator launch.** A bounded CPU replay consumed exactly 10,000,000 nonpadding positions from 29,806 available single-presentation training examples. It produced 4,770,199 supervised next-token labels. A fresh extraction reproduced all 39 update groups, the restored suffix and the terminal cursor. No model weights were trained. Accelerator credits used: zero.

| Production lane | Previously admitted | Newly admitted | Used through 10M | Supervised labels used | Left in lane |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom research | 407,691 | 2,092,917 | 2,500,608 | 433,766 | 0 |
| General text | 2,658,936 | 843,228 | 3,500,032 | 3,496,420 | 2,132 |
| Verified | 559,168 | 1,441,480 | 2,000,512 | 497,614 | 136 |
| Procedural | 348,586 | 1,150,579 | 1,499,136 | 307,270 | 29 |
| Failure-directed | 240,373 | 259,402 | 499,712 | 35,129 | 63 |
| **Total** | **4,214,754** | **5,787,606** | **10,000,000** | **4,770,199** | **2,360** |

The quotas come from the frozen pack scheduler, including the partial update at 10M. The new supply contains 22,000 examples. The combined train-only release comprises 27,894 checked exercises and 1,912 attributed text sections or passages. Of 29,806 available roots the replay began 29,803. Independent checks recovered every consumed ID and shifted label from the published source rows. The remaining positions are the unused tails of the frozen lanes. [Machine receipt](MILESTONE10M.json) · [Exact reproduction commands](MILESTONE10M_REPLAY.md)

## What fills the custom-research gap

| New research gym | Cases | Positions | Observable decision or correction |
| --- | ---: | ---: | --- |
| Counterfactual sensor selection | 1,582 | 505,471 | Buy the cheapest observation that distinguishes two states, or report that no sensor can distinguish them. |
| Euclidean versus binary GCD | 1,511 | 269,226 | Compute both algorithms’ operations and select the cheaper one under the supplied tariffs. |
| Minimax witness purchase | 1,582 | 383,033 | Stop or buy a probe under the exact worst-case cost over surviving hypotheses. |
| Native DataGoblin experiment-plan audit | 1,431 | 544,282 | Check a proposed plan in the existing RLT gym and supply a valid correction when needed. |
| Boolean equivalence and counterexample search | 1,576 | 390,905 | Give the least distinguishing input or certify equivalence, then compare the scan and algebra costs. |

The native RLT gym checked all 1,431 experiment plans. It accepted 301 proposals and the examples supply corrections for 1,130. The existing finite-automaton witness solver independently checked all 1,576 Boolean cases, including 526 equivalent program pairs. In the sensor gym, 51 cases correctly returned that no observation could distinguish the pair. These stopping, equivalence and abstention labels are part of the training signal. Operation tariffs are specified by each task; they do not measure hardware latency. [Gym generator](milestone_research_tasks.py) · [Independent solvers](milestone_research_check.py)

The research intake rejected 226 duplicate candidates and checked the 7,682 admitted lineages against the starter training corpus. To meet the exact final 7,970-position tail, it selected 29 of 31 complete, verified examples by token length. The other checked lanes stopped at their first complete example past quota. No row was repeated or lengthened to reach a quota. [Research admissions](milestone_research.py)

## Other supply and evaluation custody

The procedural increment has 6,142 state traces and 2,622 operator-identification examples. The verified increment has 3,097 Horn proofs of depth 5–18. The failure-directed increment has 1,065 faulty machines with shortest counterexamples of length 2–7, and 331 genuinely equivalent relabelings. The starter retains its kernel-checked Lean exercises. Across these three new lanes, the intake rejected 5,407 semantic duplicates and independently recomputed every admitted response. [Checked corpus builder](milestone_checked.py)

The text increment has 236 complete sections and 825 contiguous passages from 60 books across humanities, social science, physics, business, earth science and workforce education. Every passage retains source offsets, license, author, title and link. An independent audit recovered the original bytes and complete token arrays for all 1,061 sources. The whole-section trial supplied only 121,364 positions. The subsequent intake preserved intact passages and removed source components shared with prior roles. Per-section CC BY 4.0 metadata is the basis for admission, with separate publisher spot checks in each of the six subjects. These English textbooks broaden the training mix within a single upstream collection. [Prose intake](milestone_text.py) · [Independent source audit](milestone_text_audit.py)

The additions use train grammars, native in-distribution generation and recorded train lineages. The original confirmation, adjudication and final populations keep their roles. The text audit compared prior manifests and author/book components, full historical book passages and held windows. The math audit checked the registered exclusions, canonical task and lineage keys, and tokenized records. The new sensor tasks use primes 13, 17 and 19 and at most one singular map. They share state/reset primitives with earlier work. No CONFIRM14 held repair templates, archived proof trajectories or private counting-family examples were imported. The screens establish the stated split and duplication checks; arbitrary semantic equivalence remains undecidable in general.

## What ran on CPU

A fresh index interleaves all five lanes using the unchanged target deficits. It orders rows inside each lane by a frozen root hash so the incoming cases mix with the starter from the beginning. Older indices stay intact. This new index starts from position zero. Corpus replay ran on one allowed CPU with a 1 GiB cap, a 600-second wall limit and CUDA hidden. The canonical run took 42.08 seconds elapsed and 38.58 seconds of process CPU, including the independent label checks and the repeated suffix. It emitted 4,883 packs and 384 masked pad positions. Update group 39 carries 38,528 nonpadding positions. Closing and reopening all five readers after group 19 reproduced the remaining 5,019,264 positions exactly. [Stream builder and replay](milestone_mix.py)

The extracted operator ZIP passed all 230 member checks. Its bundled verifier independently reproduced the same 39 ordered-pack hashes and both cursor commitments. Run `python -I -B verify_bundle.py --receipt reproduced.json` in a fresh extraction. ZIP SHA-256 is `8042336235ba39ae1786416d55d7fdf3f641bd2ed43c609b2ef11e59c3ea25b1`. The frozen index SHA-256 is `be25522ded491e0c0c3f81d87d786034b46733e1ccbe2b9cea2dafc47b48abf4`.

## What still blocks direct A

1. **The first 10M on target hardware.** Direct A remains the 999,352,320-parameter dense model with ten shared core blocks traversed twice and 2,048-token context. The full-size segmented BF16 runner still needs validation on the intended topology, including document isolation, global label-weighted loss and gradients, sharding, memory fit and finite updates. This stream replay executed zero optimizer updates. There is no trained 10M checkpoint.
2. **Durable training restart.** The data cursor is qualified. The production checkpoint must persist and restore model shards, AdamW moments, token-schedule state, per-rank RNG and these exact lane positions across interruption. The existing numerical CPU test exercised a small model. Accelerator parity, checkpoint storage and recovery time remain unmeasured.
3. **Beyond the first checkpoint.** Only 2,360 admitted positions remain in this snapshot after 10M, with the custom-research lane exhausted. The 100M milestone needs at least another 89,997,640 positions. The full first leg needs **9,989,997,640** more, in the frozen proportions. Append future supply while preserving this consumed order and validating cursor migration against the uninterrupted prefix. New training books do not increase the independent author/book groups in each existing text evaluation role. Broader evaluation remains necessary for a production quality decision.
4. **Cost and authorization.** Inclusive useful throughput and credit burn on the intended accelerator are unmeasured. The prior 55-hour estimate is an operator planning figure. Paid execution still requires explicit approval. The tournament decision on global damping is unchanged. [Frozen model and optimizer](RECIPE.json) · [Original launch gates](READINESS.md)
