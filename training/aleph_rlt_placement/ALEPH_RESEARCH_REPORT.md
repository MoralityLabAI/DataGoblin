# Data Goblin Aleph research report

## Decision for the 10B-token tournament

**Keep the credits unspent.** Dense direct recurrence is the full-scale reference. Whole-pass Rectified Looped Transformer (RLT) is the leading placement challenger. At 14.18M parameters it improved average byte loss against direct feedback, but constant damping beat it on text in every seed. The newer receipt experiments answer a separate question about learning what to communicate to another researcher.

| Decision | Current evidence | Action |
| --- | --- | --- |
| Dense loop placement | Whole-pass RLT led the registered small screen. Its larger replication split the language and checked-task metrics against damping. | Keep direct, RLT and damping together for fresh confirmation. |
| Source rereading | A second-pass prelude read is distinguishable from pass damping. Its value on verified repairs remains unmeasured. | Use it as an alternative miniature experiment before assigning a TPU slot. |
| Receipt architectures | Coordinated supervision taught both C and D to choose the optimal parity code. The smaller MLP led actual probability scores on fresh two-rival LDT cases. | Keep C as a sender prototype and the MLP as a required control. Neither recurrent proxy earned 10B. |

### What a small researcher can contribute

A worker can propose a compact rule, report its value at two deployment settings, or identify one missing counterexample. Another worker can check it independently and test how much it helps a later decision. The strongest findings concern particular interfaces and finite games. A fresh MLP read the frozen C bit better than a fresh C receiver on every seed. Even then, a calibrated side-only predictor won. The binding constraint is the probability a collaborator can actually produce from the receipt.

**Evidence base.** The project completed 51 initial wiring fits, 42 registered mixed-corpus fits and all nine prespecified larger placement fits. Two one-bit studies contributed 18 and 45 training phases. The final assay added 27 fresh-receiver fits. These counts describe distinct experiments, scales and endpoints. Full-scale language-model training and TPU costs remain projections. All experiments reported here used local CPU compute.

[Placement findings](FINDINGS.md) · [Receipt pilot](RECEIPT_RESULTS.md) · [Coordinated follow-up](RECEIPT_FOLLOWUP_RESULTS.md)

## Architecture and the strongest controls

Inkling 2 Mini motivates where to repeat computation. Its fixed decoder groups 24 physical layers into four six-layer blocks. Executing the prelude, three visits to the two middle blocks, and the coda gives 48 logical layer applications. Later blocks can read the prelude and prior messages. Aleph borrows the tied middle and uses dense computation throughout. The research compiler, RLP, SGRT/HoloArc, LDT, curation and external verification live in software.

| Configuration | Intervention being tested |
| --- | --- |
| Direct anchor | Ten physically shared middle blocks repeat in order. Each visit passes its endpoint directly into the next visit. |
| Whole-pass RLT | For pass input x and endpoint C(x), write x + a[C(x) − x]. At three visits the coefficients are (1/3, 1/2, 1). |
| Matched damping | Use the same site and mean nonterminal coefficient. At three visits the schedule is (5/12, 5/12, 1). At two visits both methods are exactly (1/2, 1). |
| Alternative prelude read | Divide the ten-block middle into U and V. Halfway through visit two, use ¾ of U’s current output and ¼ of the saved prelude as input to V. The local control substitutes the second pass’s starting state for the prelude. |
| C and D receipt proxies | C compares two option scores. D lets a proposal read an evidence slot twice. Each uses 6,880 parameters and the same dense recurrent shapes. The MLP uses 1,601 parameters. |

### What the architecture audit established

The CPU float32 audit checked physical reuse and equal trainable counts, 36 causal suffix checks, and exact identity of the terminal RLT update through a paired Adam step. A checkpoint restart reproduced parameters, optimizer moments, the next batch, loss and RNG state bit for bit. For a stationary linear pass J, the three-pass RLT-minus-damping operator is −J(J−I)²/144. The schedules agree to first order around identity, so damping is an unusually strong rival.

The strict model has 999,352,320 unique parameters, width 2,048, 16 query heads, four KV heads, tied vocabulary weights and two answer blocks. The 10 dense core FFNs use width 8,960. Four shell and answer blocks use 8,192. Both two-pass and three-pass variants preserve this parameter count.

[Inkling model card](https://huggingface.co/thinkingmachines/Inkling-2-Mini) · [Loop Schedule Algebra](https://github.com/MoralityLabAI/LoopedTransformers/blob/31256abc921ae9aa500ad4fb85a167a990be0984/docs/lsa_typed_architecture_calculus_v0.md) · [Aleph design](ARCHITECTURE.md) · [Prelude reference](inky_pair.py) · [Audit](audit.py)

## What the placement tournament found

The 36 two-pass and 15 three-pass original wiring fits exposed a weak benchmark. Every arm in the three-pass cohort emitted a single prediction per seed across 192 examples. The mixed screen established direct competence on text and at least one checked task on three seeds. Its 42 fits compare 1,414,480-parameter models over 192 updates on text, Lean, rule revision, verified repair and counterexample generation.

| Small-screen comparison | Relative byte NLL vs direct | Checked macro vs direct | Strong control |
| --- | ---: | ---: | --- |
| Two-pass global RLT | −2.42% | −0.55 pp | Global damping won on both means. |
| Three-pass core-edge RLT | −0.43% | +2.08 pp | +4.41 checked points vs edge damping, but failed the complete direct gate. |
| Three-pass whole-pass RLT | −2.98% | −0.08 pp | Damping won text; RLT gained 1.28 checked points over it. |

The screen included global first, penultimate and combined writes, alongside physical core boundaries and whole-pass updates. First-plus-penultimate beat its own damping control on text but failed the direct guard. The prespecified cross-depth tie band was 0.5 percentage points. Whole-pass RLT exceeded that band by about 0.064 points relative to global RLT and advanced to the larger development replication.

| 14.18M model, three seeds | Byte NLL ↓ | Lean @4 | Joint revision | Repair | Counterexample |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 2.9579 | 44.0% | 1.0% | 28.5% | 19.4% |
| Whole-pass RLT | 2.9196 | 42.9% | 0.5% | 30.6% | 26.4% |
| Matched damping | 2.9048 | 42.9% | 0.0% | 23.6% | 25.7% |

RLT improved mean byte NLL by 1.27% and checked macro by 1.83 points against direct. Against damping it lost 0.51% on NLL and gained 2.04 checked points. The descriptive paired interval for RLT minus damping was +0.0066 to +0.0240 nats/byte and −1.71 to +5.51 checked points (1,000 clustered draws, fixed seeds). RLT beat direct on text in two seeds. Damping beat RLT on text in all three. All nine prespecified replication cells finished. The earlier generic five-arm status display expected two additional edge arms outside that plan. The near-zero joint revision scores and small number of independent proof families limit the selection evidence.

On the larger fixed probe, RLT’s native NLL was 3.0716; switching its runtime clock to damping gave 3.0717. Moving from three to six test passes worsened its probe NLL to 3.0975. Peak activation RMS was 85.3 for direct, 54.2 for RLT and 57.4 for damping. Lower activation scale alone did not identify a better clock.

[Registered protocol](PROTOCOL.md) · [Screen and controls](FINDINGS.md) · [Review and replay](RESUME.md)

## A playground for distributed researchers

EvalMath asks which interventions distinguish two explanations. Compactification asks what smaller representation preserves the answers to those interventions. For Data Goblin the operational question is specific: after the source disappears, what can a different worker still infer from the public receipt, an independently checked observation and a limited verification budget?

| Finite gym | Verified outcome | What it lets a small worker do |
| --- | --- | --- |
| Merge partial evidence | Target-only uploads gave 32 wrong feasible target sets in 144 two-lab cases. Projecting onto the shared boundary matched centralized evaluation on all 2,704 tested pairs. | Preserve a relation involving a partner’s variable even when it gives no standalone answer. |
| Pick a repair to teach | With 16 checks, selecting uncovered target mass gave 85.94% downstream finite-student accuracy; checked random selection gave 73.44%. Six targeted new prompts completed coverage and reached 100%. | Buy a transition the downstream student lacks and measure its incremental value. |
| Certify a Lean slate | Checking candidate differences certified the same decisions in 328 calls as 492 exhaustive calls. | A deterministic exclusive-first policy matched the learned call count. |
| Predict a clock intervention | For a shared deterministic core, normalized two-pass order effects satisfy q(a,c) = q(a,b) + q(b,c) on a common projected state. | Combine two independently checked measurements into a forecast for a third. |

### A one-bit game with an exact answer

A producer sees a uniform three-bit world and a deployment prior λ. It uploads one bit. Later a fresh receiver sees a target parity, the direction and value of a separately verified side parity, and the uploaded bit. With probability λ the target is the first bit; otherwise it is uniform over seven nonzero parities. The receiver predicts a probability and receives Brier loss, the squared error against the binary outcome.

Over all 128 canonical encoders, including the constant, the best incremental reduction in Brier over side-only is the larger of 1/14 + λ/56 and 3/49 + 15λ/98. The optimum changes at λ = 4/53. A proposed code’s fixed risk is affine in this prior, so two verified endpoint reports price that code at every intermediate prior. Complemented codes and all equal-value optima earn the same credit. Fable’s supplied enumeration and the independent rational calculation agree on the three-bit switch; a separate four-bit enumeration gave 16/667.

[EvalMath interventions](https://github.com/MoralityLabAI/EvalMath/blob/497f87cd83fd1ca8d53f9e39ebd74474612f1dfe/paper/bpd_interventional_separation.md) · [Compactification Theory](https://github.com/MoralityLabAI/Compactification-Theory/blob/e70534fbfebc0d1124e4cdf23f55531694dd2870/note/intervention-preserving-compactification.md) · [Merger](FEDERATED_RECEIPTS.md) · [Repair](REPAIR_TRANSFER.md) · [Lean certificates](SLATE_CERTIFICATES.md) · [Clock gym](CLOCK_ORDER_GYM.md)

## Why the first receipt never switched

In the first study, C, D and the MLP each trained on three matched seeds for 1,200 parity updates. The producer learned from the expected Brier loss and a term crediting information an ideal receiver could extract. Training integrated over two possible hard messages. At scoring time exactly one bit crossed into a fresh receiver. All nine models stored the first bit at every tested prior. Their receivers decoded it almost optimally. That policy wins above 4/53 and misses the diffuse-prior optimum.

| First LDT transfer | Actual Brier ↓ | Ideal readout of its learned bit ↓ | Constant uploads |
| --- | ---: | ---: | ---: |
| C, option comparison | 0.3335 | 0.1499 | 21/24 |
| D, evidence revision | 0.3037 | 0.1536 | 24/24 |
| MLP | 0.3175 | 0.1244 | 3/24 |
| Exact side-only control | 0.1536 | — | — |
| Fixed majority with ideal receiver | 0.0399 | 0.0399 | — |

After 600 updates on LDT training families, the held cases introduced two rivals and asked whether a focal entry survived or remained unresolved. Native LDT rules graded all 384 world/query pairs across eight frames. The two point rivals gave a target equivalent to “majority OR the shared metric”; the point-plus-interval case gave “majority AND the isolated metric.” One majority bit achieved the unrestricted one-bit ceiling on every frame with an ideal receiver. A stock majority rule was a serious control.

### Why local edits can teach the wrong lesson

Starting from the first-bit code, flipping the upload in one world changes optimal-readout risk by (15λ − 1)/196. At λ = 0.07 a single edit costs 0.000255 Brier, yet a coordinated change to majority gains 0.000740. The first-bit code is locally stable over a region where a nonlinear code is globally better. This suggests testing training signals that compare whole proposed codes.

The first LDT split also introduced the second-rival input slot and its role simultaneously. The next study exercised both slots throughout training and reserved their containment relationship for evaluation. Direct numeric comparisons across the two LDT studies mix different tasks and populations.

[One-bit protocol](RECEIPT_PROTOCOL.md) · [Pilot results and commands](RECEIPT_RESULTS.md) · [Exact parity oracle](delayed_parity.py)

## Coordinated supervision and transfer

The external enumerator identified two train-optimal truth tables, masks 46 and 170. They differ only in worlds 2 and 7. The sender learned the signed risk difference; the receiver trained on a teacher’s hard bit. An outcome-only control matched sender exposure and the learning-rate schedule. The four held bracket points were .070, .072, .079 and .080. A pass required both an optimal code and realized Brier better than side-only at each point.

| Three-seed result | C | D | MLP |
| --- | ---: | ---: | ---: |
| Original / exposure-matched outcome passes | 0 / 0 | 0 / 0 | 0 / 0 |
| Coordinated passes | 3 | 3 | 2 |
| Passes retained after sequential LDT adaptation | 0 | 0 | 0 |
| Coordinated realized bracket Brier ↓ | 0.1671 | 0.1878 | 0.2015 |

Every supervised sender selected the correct bracket code. One MLP receiver missed the side-only criterion. The original C/D models scored about 0.1416 while always sending the first bit. Improving code choice made actual predictions worse. A 201-point grid showed the two world-specific changes could land at adjacent or different priors. At the exact tie, either complete code is optimal and a half-completed change costs exactly 1/1484 Brier.

| Two-rival held cases | Actual Brier ↓ | Ideal readout of learned bit ↓ | Constant frames |
| --- | ---: | ---: | ---: |
| C | 0.2275 | 0.1181 | 0/48 |
| D | 0.2284 | 0.1456 | 16/48 |
| MLP | 0.2200 | 0.1163 | 0/48 |
| Side only / best linear with ideal readout | 0.1749 / 0.0720 | — | — |

Both rival slots carried evidence during training. Four held groups covered both containment orientations and native statuses, for 16 frames and 768 world/query pairs. Both statuses reduced to the singleton metric on these constructions. Per-frame best linear reached the unrestricted one-bit optimum in all 16 frames, making the strict beat-linear gate impossible on this draw. Independent grounds for holding promotion remained: all learned arms lost to side-only on Brier, neither C nor D beat the MLP on all seeds, and parity competence was lost after adaptation. C’s forced-majority receiver still scored 0.2210 against an ideal majority readout of 0.1166.

[Frozen follow-up protocol](RECEIPT_FOLLOWUP_PROTOCOL.md) · [Paired results and transition grids](RECEIPT_FOLLOWUP_RESULTS.md)

## Can a fresh receiver recover the information?

The follow-through fixed all nine post-LDT senders from the coordinated study and crossed them with fresh C, D and MLP receivers. Each of the 27 fits got 500 Adam updates on the same 24 training frames. Only a hard bit crossed the boundary. The target for each observed query, side bit and upload was its exact conditional success frequency among the eight worlds. With the sender fixed, this weighted squared-error objective differs from expected Brier by the irreducible within-cell variance. The change tested here is a fresh, separately trained receiver.

| Same frozen C sender | Seed 137 | Seed 163 | Seed 191 | Mean Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| Intact C receiver | 0.23027 | 0.22505 | 0.21893 | 0.22475 |
| Fresh C receiver | 0.21142 | 0.22637 | 0.22501 | 0.22093 |
| Fresh MLP receiver | 0.20379 | 0.21138 | 0.21406 | 0.20975 |
| Bayes receiver of C’s actual bit | 0.10784 | 0.11499 | 0.11379 | 0.11221 |
| Side-only Bayes receiver | 0.16934 | 0.16934 | 0.16934 | 0.16934 |

On C senders, an untrained C receiver averaged 0.25577 and the trained fresh D receiver averaged 0.21977. All 27 fits beat their own random-initialization controls. The fresh MLP improved on intact C in all three seeds. The fresh C receiver did so in one seed. All 27 learned sender/receiver seed combinations scored worse than side-only on their held aggregates. A calibrated receiver of C’s actual bit could improve on side-only by 0.0571 Brier on average. The best of the two displayed fresh receivers still left about 0.0975 Brier between its predictions and the ideal C-bit readout.

### What the experiment settles

The uploaded C bit retained information a collaborator could use. The learned readouts remained poor at mapping a held relation, a calibration and that bit to a probability. The private draw changed bounds, query contexts and their weights within the previously defined containment families. This tests frame transfer within a known topology. More evidence is needed for transfer to new verifier semantics.

The sender entered this experiment after LDT adaptation, when the parity switch had already been lost. All 162 archived parity probes measured exactly zero sender-logit drift during readout training. The earlier loss of competence was already present at entry. Future grading should keep sender value, actual partner utility and retained past skills as separate endpoints. The paid decision remains on hold.

[Readout protocol and replay](RECEIPT_READOUT_PROTOCOL.md) · [Frozen-sender experiment](receipt_readout.py) · [Paired readout results](RECEIPT_READOUT_RESULTS.md)

## Corpus, budget and checkpoint selection

The mixed pilot came from the evidence branch at commit 48c99a7. All 263 archived members verified. The adapter loaded 512 training and 128 held text windows, 36 training and 28 held Lean goals, paired rule-revision turns, first-fault repair and counterexample tasks. Only seven structural proof families were held out. The fixed Lean order solved 8/28 within four proposals; the candidate pool could solve at most 16/28. Small-model gains have to be interpreted against these floors and ceilings.

| Planning scenario | Two initial 10B legs | Third 10B leg | Total |
| --- | ---: | ---: | ---: |
| Operator estimate, 350 credits and 55 h per leg | 700 credits | 350 credits | 1,050 credits / 165 h |
| Three passes if 350 credits measured two passes | ≈978 credits | ≈489 credits | ≈1,467 credits |

The stated balance is 1,000 credits. At the quoted rate, the entire plan exceeds it by 50. Moving from two to three visits adds an estimated 39.7% training work. The second row assumes proportional credits; the TPU rate at either depth has not been measured in this study.

### The two 10B checkpoints and the continuation

If fresh confirmation supports a clock-specific gain, use three-pass direct and three-pass whole-pass RLT with identical initial weights, ordered corpus, 10B processed nonpadding positions, optimizer and verifier budgets. If the question instead becomes source access, the two-pass direct versus prelude-read pair needs its own small matched study. Choose one question before spending. Use fresh document and generator families to exclude leakage and predeclare the primary outcome and minimum worthwhile improvement.

Use byte or token loss on held text, Lean success within four attempts, joint rule revision, verified repairs and accepted counterexamples. Hold promotion on text degradation above 1%, checked-macro degradation above two points, or measured throughput loss above 10%. Use paired uncertainty at the document or family level. A single seed per 1B arm supports a checkpoint choice. Continue the selected 10B checkpoint with its optimizer state and data cursor for a further 10B positions; preserve both original checkpoints and retain an explicit unresolved outcome.

The planning recipe uses BF16, AdamW, peak learning rate 0.00025, 200M warmup positions and an LR clock spanning 20B positions. Freeze tokenizer, packing boundaries, source rights, corpus mixture, and measured throughput before launch. The frozen local evidence archive is a development corpus; a licensed and deduplicated 10B stream remains an intake requirement.

[Strict-1B configuration](fullscale.json) · [Packing and cursor implementation](packing.py) · [Data and training replay](RESUME.md) · [Selection rules](PROTOCOL.md) · [Evidence commit](https://github.com/MoralityLabAI/DataGoblin/tree/48c99a7d9c2d37c82fe127369dfa0124b88bc8ab)
