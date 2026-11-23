# Verified action sets and bounded repair in the native Storyworld courier

DG-METTA-13 tests whether a weight-tied Looped Transformer benefits from supervising equivalent actions, intervening at premature STOP, or combining the two. The experiment uses independent synthetic courier roots and measures accepted STOP after MeTTa-certified delivery under a shared 13-query ceiling. Both positive and null findings concern this finite task and verifier tariff.

## Historical audit

The 144-world native courier pilot remains at DataGoblin commit `1c65eb2a56f120aa8e530df4c4d308dee30f0ea7`. The protocol lock points to `5697d37f0adb782d1829ecd545e2afa92263434a`. All five published archive digests matched, all 344 serialized public root identities reconciled, and the original MeTTa kernel reproduced 400 training and 1,440 frozen replay receipts. The 1,452 training frames include 132 shortest ties. All nine archived fit records have distinct checkpoint digests, matched initial digests within each seed, 44,417 parameters, and 8,640 optimizer draws. The archive contains fit identities and omits the historical checkpoint files, so tensor-level checkpoint verification was unavailable. [Historical report](https://github.com/MoralityLabAI/DataGoblin/blob/1c65eb2a56f120aa8e530df4c4d308dee30f0ea7/MeTTa/STORYWORLD_COURIER_RESULTS.md)

The paired calculation reproduces canonical 387/432, hashed 390/432, and full set 388/432. Full-set supervision changed completion by +0.23 percentage points against canonical, with a 95% interval from −2.08 to +2.78 points. It saved 0.181 primitive actions per attempt and 234 recurrent visits. The +2-point target was missed. That pilot had a shared native-checked legal mask, separate replay accounting, and uniform cross-entropy over shortest ties. The original frozen roots were accessed only for the receipt audit.

| Design | Historical native pilot | DG-METTA-12 at `88c9109` | DG-METTA-13 confirmation |
| --- | --- | --- | --- |
| Domain | Native permit and one dispatch | Python multi-delivery and evidence reference | Pinned native movement, permit and dispatch, with native independent-source evidence |
| Roots | 200 train, 144 frozen across disjoint topology families | 192 train, 48 development, 160 frozen, 120 transfer | 64 train, 80 independent held-out roots in four strata |
| Optimizer allocation | 180 AdamW updates, 48 draws each | 600 updates, 48 draws each registered | 180 updates, 48 draws each |
| Shortest-set loss | Cross-entropy to uniform target | Negative log mass of the whole set | Negative log mass of the whole set |
| Horizon and calls | 12 actions, shared native successor validation and one replay per trial | Five charged reference action calls, with native confirmation gated | 12 actions and 13 actual MeTTa queries, including STOP, with one whole-proof query per macro attempt |
| Main comparisons | Full set versus canonical | Full set versus hashed, plus separate full-set repair contrast | Three preregistered factorial contrasts |

DG-METTA-12 registered 520 feasible reference roots and 380 root-level shortest ties. The Storyworld SHA that was unresolved when its draft was written now resolves and passes the three original courier tests. Its reference results establish a distinct domain and accounting contract. [DG-METTA-12 draft](https://github.com/MoralityLabAI/DataGoblin/pull/57) · [Pinned Storyworld](https://github.com/MoralityLabAI/metta-storyworld/commit/4cfe1713d73b4e508833b668477921adfbe570a6)

## Model and verifier

A courier state contains location, permit, cargo, and witnessed source identities. Time and calls remaining accompany that physical state. Two public outlets can attest one source, and delivery requires evidence from both independent identities. The LoopedPointer observes public facts, local legal choices, and resource counters. It uses the pinned 44,417-parameter LoopedDecoderLM with one shared physical block and three visits. The obligation-aware prototype retains the parameter count and injects selected public slot embeddings before the second and third visits. [Pinned Looped Transformer](https://github.com/MoralityLabAI/LoopedTransformers/commit/950d4d594d87a1f74c6139a928a9b6cb4ab70617)

The standard control filters repeated physical states from its ordinary proposals. Bounded repair can re-enter them if a complete suffix remains legal. Search supplies candidate actions. MeTTa proves every primitive transition and returns the whole ordered macro proof in one query. A failed query charges the ledger and preserves the state digest. A successful macro still needs a separately accepted STOP. Native call counts measure actual runtime evaluations. Query counts alone do not capture native work, since macro depth and search increase the work within a query.

The shortest-set objective treats all exact shortest ties as eligible. The budget set includes every first action with a successful continuation under both caps. The soft objective weights that support by independently checked continuation cost. When every legal primitive belongs to the budget set, the loss can discourage early STOP while supplying little direction among those primitives. All objectives share the same presentation and optimizer budget.

## Finite resource theory

The formal specification defines equivalence by the language of successful certified suffixes under the primitive horizon and verifier budget. Equivalent states have the same feasible optimum under the specified tariff. The converse can fail because cheap suffixes can use different actions. A macro proof expands into valid primitive transitions by induction on its length. Consequently the physical state graph is preserved given unlimited query allowance. A single-query certificate for multiple transitions can strictly expand the reachable set under a small verifier-call ceiling.

Verified gains depend on how often the policy proposes incomplete STOP while a certified suffix still fits the remaining action horizon and repair depth. If reserve repair activates only at incomplete STOP, a baseline successful prefix is preserved by construction. Whether an eligible failure occurs, and where supervision changes the probability of eligibility, are empirical questions. The difference-in-differences measures that interaction at the registered root distribution. It cannot assign a function to an internal activation.

## Development

On 12 development roots, standard canonical, hashed, and shortest-set models completed 5, 7, and 9 respectively. Their reserve counterparts had identical outcomes. The exact reference completed 12 and the broad budget set completed zero. The 144-trial local run made 1,529 scoring calls and used 191 verified training frames, including 34 shortest ties and 171 rows with at least one successful longer continuation. Windows and Linux CPU development runs produced identical training frame and native scoring receipt digests, including identical endpoints and paired estimates. The fitted tensor digests differed across platforms. These observations are descriptive. [Development workflow](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45210927456)

## Confirmation

On 80 frozen roots and three matched seeds, shortest-set reserve repair completed 122/240 attempts compared with 119/240 for standard shortest-set inference. All three paired gains followed a certified STOP repair. The +1.25 percentage-point estimate had a paired 95% interval of 0.00 to +2.50 points. The registered positive lower-bound criterion was missed. The hash-selected checkpoint with the same repair policy completed 134/240. [Registered confirmation run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45212967243)

| Arm | Verified completions | Primitive actions | Native queries | Unused allowance | Queries per completion | Recurrent visits | Incomplete STOPs | Macro attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Canonical | 107/240 | 2,512 | 2,752 | 368 | 25.72 | 8,256 | 133 | 0 |
| B · Hash singleton | 133/240 | 2,390 | 2,630 | 490 | 19.77 | 7,890 | 107 | 0 |
| C · Shortest set | 119/240 | 2,359 | 2,599 | 521 | 21.84 | 7,797 | 121 | 0 |
| D · Canonical with repair | 108/240 | 2,516 | 2,753 | 367 | 25.49 | 8,259 | 133 | 1 |
| E · Hash singleton with repair | 134/240 | 2,394 | 2,631 | 489 | 19.63 | 7,893 | 107 | 1 |
| F · Shortest set with repair | 122/240 | 2,369 | 2,602 | 518 | 21.33 | 7,806 | 121 | 3 |
| G · Budget set with repair | 105/240 | 2,583 | 2,818 | 302 | 26.84 | 8,454 | 137 | 2 |
| H · Exact reference | 80/80 | 657 | 737 | 303 | 9.21 | 0 | 0 | 0 |
| I · Soft weights with repair | 138/240 | 2,435 | 2,675 | 445 | 19.38 | 8,025 | 102 | 0 |
| J · Early spending | 121/240 | 2,335 | 2,542 | 578 | 21.01 | 7,626 | 119 | 12 |
| K · Learned gate | 119/240 | 2,359 | 2,599 | 521 | 21.84 | 7,797 | 121 | 0 |
| L · Obligation-aware | 126/240 | 2,357 | 2,593 | 527 | 20.58 | 7,779 | 116 | 2 |

| Prespecified completion contrast | Difference in points | Paired 95% interval | Seed effects in points, 1101 / 1102 / 1103 |
| --- | ---: | ---: | ---: |
| F−C · Repair on a shortest-set checkpoint | +1.25 | [0.00, +2.50] | +2.50 / +1.25 / 0.00 |
| F−E · Shortest set versus hashed singleton with repair | −5.00 | [−10.00, 0.00] | 0.00 / −5.00 / −10.00 |
| (F−C)−(E−B) · Supervision-by-repair interaction | +0.83 | [0.00, +2.08] | +1.25 / +1.25 / 0.00 |

The 10,000 bootstrap draws resample roots within the fixed topology strata, retaining the seed pairing in each sampled root. D−A and E−B each gained one completion with zero regressions. F−C gained three with zero regressions. All five lifts were attributable to native-certified macros at incomplete STOPs.

Standard shortest-set inference saved 0.638 primitive actions per attempt against canonical, with a paired 95% interval of 0.363 to 0.900 saved. It also saved 1.913 recurrent visits per attempt, with an interval of 1.088 to 2.700. Its completion estimate against canonical was +5.00 points [−2.92, +12.92]. Under the common repair policy, the shortest set had lower observed completion than the hashed singleton and the interval reached zero. The interaction estimate remains uncertain.

Soft weights gained an exploratory +6.67 points [+0.83, +12.50] against the full shortest set with the same controller and used 0.304 additional native queries per attempt. The broad budget set changed completion by −7.08 points [−15.83, +1.67] and used 0.900 additional queries per attempt. Early spending saved 0.250 queries per attempt against reserve repair [0.071, 0.471 saved] while the completion interval covered zero. The learned gate made no macro attempt and matched the standard shortest-set cell. These estimates are exploratory.

## Failure analysis and recurrent probes

The retrospective replay found three eligible repair opportunities among the 121 standard shortest-set failures. Seventy-five failures arrived at STOP with no primitive actions remaining. Of the other 46, 43 had no complete suffix within the lesser of the remaining horizon and the four-action macro limit. Each eligible opportunity was rescued. The three rescue macros used three, four, and three actions respectively. Each corrected the route and delivered cargo. Targeted tests verified permit, pickup, and evidence recovery.

The learned policies produced no invalid masked proposal. The raw top score frequently pointed to a masked action, including 2,227 decisions in F, so feasibility masking did substantial work. All 80 roots were feasible for the exact native-certified reference. Full-set reserve completion ranged from 11/60 on directed detours to 43/60 on gated forks. These descriptive differences suggest that opportunities for late repair depend on how much of the route remains when the learned policy stops. A controller that intervenes while more of the primitive horizon remains could reach more failed trajectories. The registered early-spending result leaves its completion advantage open.

The matched obligation-aware prototype completed 126/240 against 122/240 for the shortest-set reserve model, +1.67 points [−2.50, +6.25]. Three native-backed state probes per root verified locally decisive permit, evidence, and terminal conditions. Replacing the obligation-aware target-slot activation after visit one shifted the mean raw move, delivery, and STOP logits by −0.749, +0.191, and +0.224 respectively. The corresponding baseline shifts were −0.746, +0.236, and +0.222. Placebo-slot shifts were near zero. The direction of the permit effect concerns the raw logit of one newly legal move. Native feasibility and the preferred next move are different quantities. These interventions establish local sensitivity of the measured readout at that visit. They leave recurrent specialization into route search, constraint satisfaction, and STOP assessment open.

## Inky auxiliary controller

The 12-root comparator smoke test uses native-verified prefixes that leave four actions on a shortest route. Deterministic MeTTa search completed 12/12 with 48 calls. The development LoopedPointer completed 6/12 with 36 calls and 144 recurrent visits. Each attempt had the same five-call ceiling. These seeded repair opportunities differ from complete rollouts and establish a baseline for the four-root Inky pilot.

The registered CPU pilot used four development roots, one per topology, and all six skill prompts with the pinned Qwen3.5-0.8B checkpoint. It generated 24 first replies covering 20,636 input tokens, 4,108 output tokens, and 900.07 seconds of generation time on one CPU thread. It completed 0/8 repair cases with 16 native calls and answered 0/16 advice tasks correctly. The matched LoopedPointer completed 0/8 with 24 native calls and 96 recurrent visits. Deterministic MeTTa search completed 8/8 with 32 native calls. Counts include the prefix and terminal checks.

All eight Inky route replies reached the 256-token cap and failed the JSON contract. No Inky route reached macro certification. Across the advice tasks, seven other replies failed JSON parsing and the nine parseable replies missed their labels. This measures the behavior of the pinned model with the selected instructions and structured-response contract. Free-form explanation quality and other model backends require separate evaluation. [Inky CPU run](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45215556692) · [Pinned Qwen checkpoint](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)

## Provenance and reproduction

Freeze the source registration, split, analyser, and source pins before fitting. Run the confirmation workflow at the frozen manifest commit. The `fit` phase verifies native graph edges and training replays, then records the complete tensor and file digests. The separate `score` phase checks every fit identity before materializing held-out inputs. Raw traces and weights are published as short-lived CI artifacts and excluded from the Git tree. The legacy archive audit is read-only and can be run separately with `python -m metta_research_bench.courier_hybrid.audit --archive <archive> --storyworld <pinned-checkout>`.

Confirmation used the source registration at `8e5ee63` and frozen source and split commit `fc4b65c82bcb2f9e8f5f8288799e29b1f28ff066`. The four new topology strata are disjoint from all 11 historical topology families. The 64 confirmation train roots and 80 score roots are disjoint by source serialization and labeled graph semantics from both development splits. Native checks produced 826 training frames, with 176 exact shortest ties and 707 frames with successful longer alternatives. Training used 6,843 native queries. Scoring used 29,931 queries across 2,720 trials. The recurrent probes used a separate 640-query ledger.

Use Python 3.12, CPU PyTorch 2.7.1 and Hyperon 0.2.10. Put DataGoblin and the pinned LoopedTransformers checkout on `PYTHONPATH`. From the frozen DataGoblin checkout, run these phases with a fresh run directory.

```sh
python -m metta_research_bench.courier_hybrid.experiment fit \
  --family confirmation --storyworld <pinned-storyworld> --out <run-directory>
python -m metta_research_bench.courier_hybrid.experiment score \
  --family confirmation --storyworld <pinned-storyworld> --out <run-directory>
```

| Confirmation artifact | SHA-256 |
| --- | --- |
| Canonical manifest object | `4fa1312f47ac90a605a45de239985f52b8d437b8f4d43f08d723df82b5f46cf8` |
| Combined native rules | `7fe85980f5161d7341a2490465a4e31acb2066e9f71e852405e2bee63290212b` |
| Locked fits | `a1333b0c6c58cf744f0b09f92e74ae3e43e38b5f1e121ce1e7ff56efe3de55ce` |
| Training frames | `e472dfe70f5eae9064d16a8b06e939d8cafce45aa49e7874b06510a4e7b37dff` |
| Training proof receipts | `8d905cf30ff53c41706f9f281664f91081ef587cae173ccb9d39a0f7c39d3af2` |
| Scoring receipts | `148488b3d4505f6885e17f35b94023133c766707045c232759c2cae89da9e3c0` |
| Probe receipts | `24b06a7ec7f6c272dc02ccc5fdfa2338233aedd6e65ddbe02621de275ac92d92` |
| GitHub receipt archive | `e29b2738bfb9060609df2a09fdf4c562cd5eb4b636e2953ba3a7ed5204259e3b` |

[Confirmation receipt](CONFIRMATION_RECEIPT.json) records the full checkpoint identities, source hashes, and analysis summaries. The archive check recomputed outcomes and uncertainty from all 2,720 scoring rows, matched every artifact digest and checkpoint file digest, and reconciled each episode ledger with its proofs. All 18 confirmation checkpoints were separately fitted and verified against the lock on Windows. The reported confirmation endpoints are from the Linux CPU runner. [Confirmation artifacts](https://github.com/MoralityLabAI/DataGoblin/actions/runs/45212967243#artifacts)
