# Aleph candidate tournament decision

The [first 10M checkpoint follow-up](MILESTONE10M.md) fills the starter gaps reported below and records the current launch blockers.

**Close the reasoning-allocation tournament. Keep direct A frozen. Hold the first paid run.** A fixed symbolic portfolio reached the retrospective first-action oracle's coverage in this bounded gym. Neither learned allocator earned promotion. The production stream now contains verified examples from the winning and losing methods, but it supplies only 4,214,754 of 10 billion positions. The first bottleneck is the custom-research lane.

## What the kernel established

Lean 4.19.0 verified all 1,364 historical goals, accepted all 136 unmodified contexts and rejected all 136 invalid proof transports. A cross-platform reconstruction restored the exact archived STATE-11 paths. For 89 particularly large proofs, the compiler exposed the original typed rewrite steps as named equalities and combined them in a balanced tree. It retained the endpoints, rewrite leaves and axiom restrictions. The other 1,275 roots used the literal declarations. All 15,989 auxiliary declarations passed audit. Of the roots, 168 used no axioms and 1,196 used only `propext`.

The eight public CPU shards submitted 567 Lean jobs. Together the jobs consumed 889.30 seconds of summed elapsed time and 941.42 child CPU seconds. The largest 562-step proof took 104.71 seconds even with the balanced elaboration. Windows independently reproduced the Linux coverage-and-cost summary from the hash-verified inputs and logs. The literal batch trial hit its 25-minute public CI limit with 1,056 declarations credited. The local literal trial stopped at a completed batch boundary with 1,048 credited. Each retained four timeouts. Those partial attempts are separately costed and never added to the verified total. [Kernel results and reproduction](../state01/STATE15_JOURNAL.md) · [Machine summary](../state01/STATE15_KERNEL_SUMMARY.json) · [Public CI](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45605910185)

## What to use in the gym

The comparison charged the solver, failed attempts, verification, time spent obtaining prior state, policy inference and training. The primary 4.98112-second deadline came from four times the median of nine local singleton Lean checks. Every policy saw the same public state and had at most four solver dispatches and three kernel checks. Twenty-four held lineages from four new families each supplied an unchanged equality, a revised true equality and a false transport. Two families required composition across heterogeneous or nested products. The two train-fitted policies and six neural fits froze before held measurement. [Preregistered design](../state01/ALLOCATION_PROTOCOL.md)

| Policy at the primary deadline | True goals proved / 48 | Complete lineages / 24 | Held deployment, s | Training + deployment, s |
| --- | ---: | ---: | ---: | ---: |
| Historical congruence-first order | 19 | 6 | 194.09 | 194.09 |
| Best fixed order selected on train | 23 | 10 | 181.92 | 747.69 |
| Symbolic branch selected on train | 23 | 10 | 183.56 | 749.39 |
| Strongest of 24 fixed orders, selected on held | **26** | **13** | **180.59** | **180.59** |
| LDT, seeds 401 / 409 / 419 | 23 / 23 / 23 | 10 / 10 / 10 | 181.90 / 181.83 / 181.82 | 748.50 / 748.48 / 748.18 |
| MLP, same seeds | 23 / 17 / 24 | 10 / 8 / 11 | 196.34 / 205.35 / 183.71 | 762.84 / 771.57 / 749.75 |
| Retrospective first-action oracle | 26 | 13 | 176.30 | — |

The train-selected fixed order is **congruence → two-frontier search → normalization → checked feedback**. It is the defensible default for this gym. The stronger held order was **two-frontier search → congruence → checked feedback → normalization**. Its 26/48 is descriptive. Choosing it after seeing the four held families consumes that evidence. Standalone normalization, congruence, two-frontier search and feedback proved 9, 13, 15 and 21 goals respectively. The strongest held order solved 8/12 mixed pairs and 8/12 nested products, against 6/12 and 7/12 for every LDT seed. All three LDTs solved exactly the same individual goals as the train-selected fixed order. The best MLP seed gained two goals and lost one against that control. Its other seeds exchanged three wins for three losses and one win for seven losses. The 24/48 result therefore fails the three-seed rule.

At the primary clock there were zero false accepts. The common denotational guard screened all false goals. A separate direct Lean challenge obtained a valid antecedent and diagnosed a type mismatch for 35 of 48 stale transports. Eight lacked a checked antecedent and five timed out. These 13 count as unresolved, never as successful refutations. The clean 136/136 historical negative result and the new 35/48 result test different proof sets.

There were 364 distinct singleton kernel submissions across train and held, including 54 timeouts. They consumed 1,539.05 summed wall seconds and 1,031.62 child CPU seconds. Original training labels cost 565.76 seconds before fitting. Over this cohort, training alone costs more than serving any fixed order's entire capped deadline budget (72 × 4.98112 = 358.64 seconds). Deployment-only comparisons still matter for repeated use. The strongest held fixed order was both cheaper and more successful at the primary deadline than every LDT or MLP fit. The clock changes the ranking: at half the deadline the LDT seed 401 solved 15 against 14 for that fixed order; at twice the deadline it solved 26 against 30. Against the train-selected fixed order, the strongest held order gained three goals across three lineages and lost none. Host startup, CPU contention, proof shape and the four held families all limit generalization.

## What carries into Aleph

STATE-01–04 and CONFIRM14 showed why a locally correct state transition is a weak training target on its own. At 14M, global damping reduced mean fresh text loss by 0.39% against a 1% benefit gate. Neither arm solved a complete held repair orbit. STATE-05–09 supplied executable tests for identifying operators and exposing missing counterexamples. STATE-10–14 supplied an explicit distinction between a reusable proof, a valid changed goal and an invalid transport. This final round converts the **training-side** versions of those distinctions into 6,955 checked examples, each with a bounded task, verifier and custody record. A symbolic solver can contribute a usable example even where the learner fails to beat it. [CONFIRM14](../CONFIRM14_RESULTS.md) · [STATE-01](../state01/JOURNAL.md) · [STATE-09](../state01/STATE09_JOURNAL.md) · [STATE-14](../state01/STATE14_JOURNAL.md) · [Curation](CURATION.md)

Keep the neural model dense. Direct A has 999,352,320 unique trainable parameters, ten physically shared core blocks traversed twice and context 2,048. The routing experiments select external gym solvers. They supply no evidence for inserting an allocator or damping into the model. The user’s 55-hour, 350-credit estimate remains a planning assumption for one leg. No accelerator credits were spent.

The independently extracted train-only release has 138 members, 7,806 presentations, 2,911,698 next-token targets and 8,222 padding positions across the five individual lane scans. It includes source attribution, licenses and detached verifier evidence. ZIP SHA-256 is `91fab1badf1af05475608db3a43aa6e6741be641eb546bf9e736235bbb2e525a`.

[Launch readiness and blockers](READINESS.md) · [Machine decision](DECISION.json) · [Frozen recipe](RECIPE.json) · [Corpus manifests](SUPPLY.json) · [Exact commands](REPLAY.md)
