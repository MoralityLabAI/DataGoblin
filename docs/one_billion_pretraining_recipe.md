# DataGoblin 1B pretraining recipe

The registered pilot compares placements and training objectives in a
shared small cell. The training-pool evidence below informs candidate
collection. The structural holdout and integrated comparison will inform
the final routing and curriculum choices. The strict-1B layout is a
planning allocation. Confirm its parameter count and profile the
instantiated model on the training backend before allocating the main run.

## Module layout

| Module | Starting allocation | State and boundary |
|---|---:|---|
| Tied embeddings and text readout | 100,663,296 | The input shell owns the causal token context. The output shell reads the final state through the tied vocabulary. |
| Input and output shell | 121,643,008 | One block on each side of the recurrent center. The input block runs once per context and the output block runs once for the selected answer state. |
| Recurrent reasoning stack | 608,215,040 | Ten blocks shared across reasoning visits. The state carries causal token context, grounded premises, or a task message across visits. |
| Recurrent answer stack | 121,643,008 | Two blocks shared across answer visits. It receives the current reasoning state and the previous soft answer. Readouts emit only typed proposals. |
| Bridges, adapters, and router | 47,185,920 | Eight dense maps, 12 rank-256 pairs, and a two-matrix router. Include learned retrieval, proposer, and codec weights in this allocation when instantiating the model. |
| Final RMS norm | 2,048 | Normalizes the final readout state. |

The layout totals **999,352,320 unique weights** at width 2,048 with
16 query heads, four key/value heads, SwiGLU width 8,192, and vocabulary
49,152. It leaves 647,680 parameters within a strict one-billion limit.
The tuned router owns the visit budget. The planning layout executes 24
blocks at two reasoning visits and one answer visit, and 74 at six visits
of each kind. Profile the real visit histogram as well as the maximum.

The router owns the evidence epoch and planned refinement horizon. A
module receives the input mask, typed context, visit index, and planned
depth. At each configured RLT boundary, replace a direct write with
`state += (endpoint - state) / (planned_depth - visit_index)` using a
zero-based visit index. The verifier may stop at any readout. Keep the
original planned clock in the receipt. A new epoch clears committed
proof state and prevents the caller from replaying an old action as a
current certificate. A tested persistence policy may carry soft answer
state. The environment owns certificate checks, and the next epoch
must verify a reused result before committing it.

Run inexpensive native Lean tactics first. After a stall, retrieve from
the current environment and offer grounded actions to the learned ranker
or the lexical fallback. Stop on the first kernel-checked proof. For a
rule revision, compare reset, soft carry, and current-epoch no-op reuse.
Charge traversal and checking work as well as neural block calls. Prior
pilots support using direct transport for graph reasoning and treating
answer interpolation as a candidate for promotion by task. The mixed
pilot tests whether that choice holds with jointly trained weights.
[Prior routing evidence](rlt_routing_decision.md)

## Lean training-pool evidence

The 36 development goals in the frozen `lean-pool-compose-v4` training
split yielded the following verified results. The proposal limit applies
per goal. Work at limit 12 is the replayed cost of following that order
and stopping after the first kernel-checked proof.

| Proposal order | Proofs at 6 | At 12 | At 24 | Anywhere in pool | Tactic requests at 12 | Kernel checks at 12 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 4 | 8 | 8 | 8 | 388 | 8 |
| Wider retrieval | 8 | 12 | 16 | 16 | 352 | 12 |
| Composed tactics | 12 | 28 | 32 | 32 | 268 | 28 |

Wider retrieval and composed tactics share the same retrieved lemmas.
They recovered 72 of 76 goal–witness-lemma pairs and every witness lemma
on 32 goals. Legacy recovered 64 pairs and every witness lemma on 24
goals. At full pool depth, changing how the same retrieved lemmas were
composed doubled the verified goal count on this split.
The four composed-pool misses all belonged to `append_filter_map`, and
the retriever had found every lemma in their registered witnesses.
Five other families had a proof beyond the first six proposals for every
goal. These are distinct opportunities for better composition and ranking.

The audit paid for the deduplicated union of all three pools. It issued
2,424 tactic requests and 124 kernel checks, yielding 124 positives and
2,300 negatives. The tactic requests took 495.51 seconds and the kernel
checks took 14.93 seconds on this host. The registered learned ranker
still compares a limit of six. Keep wider prefixes in the coverage audit
to measure whether remaining misses come from order or candidate supply.

## Training and promotion

The text lane uses 512 training windows and 128 held-out windows from
the pinned byte corpora. The rule lane adds 448 independently replayed
episodes. It has 256 training episodes and 64 each for development,
longer routes, and held-out revision structures. Keep each rule lineage's
edited and unchanged variants together. Keep Lean inputs separate from
their verifier outcomes and hold out entire goal structures.

The registered pilot trains causal byte prediction, checked Lean
candidate ranking, and before/after rule endpoints together. Use a
normalized next-token loss, a verified-positive set likelihood over
known candidate outcomes, and an availability loss on pools whose
availability is known. Mask unknown grades. Rule episodes train both
endpoints and backpropagate through the soft answer carried into the
revision. Use development data to choose any future backward horizon
and verify that each promoted objective still trains the intended shared
parameters.

Compare a constant 3/4/3 text/Lean/revision batch, the same batches with
text-first objective weights, and a 5/2/3 text-heavier batch at matched
planned cell calls. The constant and text-heavier arms use loss weights
0.50/0.25/0.25. The curriculum starts at 0.75/0.125/0.125, switches to
the constant weights, and ends at 0.25/0.375/0.375. Compared with the
constant arm it preserves both the examples and mean objective weights.
Log examples, target tokens, optimizer loss weights, and wall time
separately. A scale manifest must specify real-token exposures after
tokenization and packing. Select the next staged mixture and curriculum
from the held-out trade-off across both fitted seeds.

The real-batch preflight loaded 512 text windows, 36 Lean goal pools with
checked tactic outcomes, and 256 rule-revision episodes for training. All
three losses sent nonzero gradients into the five shared cell parameter
groups. A four-update direct-feedback run reproduced the uninterrupted
weights bit for bit after resuming from update two. It checked a recorded Lean proof
under the current epoch, rejected the same proof under a wrong epoch, and
ran live graph verification on one edited and one unchanged training
episode. The check peaked at 93.4 MiB within a 128 MiB job. This preflight
exercised the data and verification interfaces before the matched fits.

Promote a learned module or routing policy only after comparing quality
and work on unseen structures, separately by task and by training seed.
Preserve text competence, show a useful verified task effect, and count
neural visits, retrieval, verifier calls, and checker traversals. Compare
the fitted schedules and runtime schedule swaps before attributing an
effect to training. Record uncertainty and keep the direct and lexical
controls available where evidence is tied or inconclusive.

Before authorizing the main run, instantiate the candidate layout and
verify unique/trainable parameter counts, causal-mask and epoch parity,
checkpoint continuation, and gradient access across the chosen boundaries.
Profile the reference backend at fixed context and matched update budget.
Measure tokens per second, peak activation and optimizer memory, compiler
warmup, checkpoint cost, and verifier time before setting the paired-run
budget. The planning ledger counts about 2.0 GB for BF16 weights and 16.0 GB
for the illustrative full Adam inventory. Both figures are decimal and
exclude activations, caches, and sharding effects. Keep measured training
memory and total charged work in the promotion receipt.
