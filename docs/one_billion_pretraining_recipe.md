# DataGoblin 1B pretraining recipe

Start the 1B staging comparison with direct feedback at both boundaries
and the constant mixed curriculum. Use exact traversal on the rule
benchmark and a direct learned ranker after native Lean stalls. Keep RLT
and damping as registered placement alternatives. These decisions come
from the checked Lean holdout and the integrated two-seed pilot below.
The strict-1B layout is a planning allocation. Confirm its parameter
count and profile the instantiated model on the training backend before
allocating the main run.

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
pilots supported direct transport for graph reasoning. The integrated
results below preserve that default and measure answer interpolation
on jointly trained weights.
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

## Lean structural holdout

Native Lean ran eight tactics on each of 28 held-out goals. It returned
zero verified proofs and reached a recursion limit on four attempts.
The grounded fallback reused a common checked cache and follows each
proposal order to its first verified action at each limit.

| Proposal order | Proofs at 6 | At 12 | At 24 | Anywhere in pool | Tactic requests at 12 | Kernel checks at 12 |
|---|---:|---:|---:|---:|---:|---:|
| Legacy | 0 | 4 | 4 | 4 | 336 | 4 |
| Wider retrieval | 4 | 8 | 12 | 12 | 296 | 8 |
| Composed tactics | 8 | 8 | 16 | 16 | 264 | 8 |

Wider retrieval and composition each recovered 88 of 104
goal–witness-lemma pairs, including every indexed pair. Both retrieved
all witness lemmas on 24 goals. Legacy recovered 64 pairs and complete
witness lemma sets on 12 goals. Composition doubled the number of
verified goals within six proposals relative to the wider order.

The held-out collection used 1,904 deduplicated proposal checks and 72
kernel checks in addition to the 224 native tactic requests. It retained
72 positive and 1,832 negative proposal outcomes and kept the four native
resource failures in the task receipts. Candidate tactics accumulated
244.83 seconds and kernel checks 15.13 seconds on this host. Native tactics
accumulated another 163.09 seconds. At the six-proposal limit the
composed order used 144 fallback tactic requests and eight kernel checks.
Use that work and coverage as the primary lexical control for the mixed
pilot. The verified ceiling in the composed pool is 16 of 28 goals.
Standalone Lean replay compiled 244 proof declarations across the two
splits, including registered witnesses and accepted proposals. The
checked pack binds each goal and candidate to its environment and receipt.

## Training and promotion

Include the [checked Campsite repair pack](rlt_hermes_repair.md#training-pack)
in Aleph development under procedural games. Supervise the repaired grid
and compare direct, answer-state RLT, and matched damping on the same data.
Score both verified repairs and verification work on held-out puzzle roots.

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
tokenization and packing. Start with the constant 3/4/3 sample mix and
0.50/0.25/0.25 loss weights. Run the alternative schedules as controlled
branches when new data or a larger model changes the trade-off.

The real-batch preflight loaded 512 text windows, 36 Lean goal pools with
checked tactic outcomes, and 256 rule-revision episodes for training. All
three losses sent nonzero gradients into the five shared cell parameter
groups. A four-update direct-feedback run reproduced the uninterrupted
weights bit for bit after resuming from update two. It checked a recorded
Lean proof under the current epoch, rejected the same proof under a wrong
epoch, and ran live graph verification on one edited and one unchanged
training episode. The check peaked at 93.4 MiB within a 128 MiB job. This
preflight exercised the data and verification interfaces before the
matched fits.

### Frontier teacher-trace candidates

See the [frontier teacher-trace curriculum catalog](../training/data/README.md#frontier-teacher-trace-curriculum-candidates-2026-11-24) for Fable/Opus, GPT/Codex, open reasoning and software-repair source leads, with an Astra discovery slot. The proposed 500M-token comparison lane evaluates raw response, compressed subgoal and independently checked state-transition supervision at matched budgets. Source admission and token accounting must precede any change to the existing curriculum. This catalog is planning metadata, with `training_ready=false`.

## Mixed-pilot evidence

All 14 fits used 8,760 parameters, 192 optimizer updates, and 156 planned
cell calls per update. The constant mixture and text-first curriculum each
processed 18,432 text targets, 768 Lean pools, and 576 revision episodes
per fit. The text-heavier control processed 30,720 text targets and 384
Lean pools with the same number of revision episodes. Match cell calls
when reading these arms and retain actual wall times in the run receipt.

On the six-proposal Lean comparison the direct ranker recovered all 16
provable goals in each seed. It used 100 ranker visits, 200 cell calls,
and 100 checked-action replays per seed. Those actions account for 100
tactic requests and 16 kernel checks in the recorded Lean environment.
The composed lexical order recovered 8 goals with 144 tactic requests
and 8 kernel checks. The ranker doubled coverage here and reduced the
number of tactic requests. It spent eight additional kernel checks
confirming the extra proofs. Retain separate budgets for model visits,
tactics, and kernel checks in the next promotion comparison.

Use the direct ranker as the first learned fallback on this task. When
the checked pool contains no proof, route the remaining budget to new
candidate generation. Twelve of the 28 goals reached this pool ceiling.
Use four visits as the first budget in this configuration. Both seeds
recovered the same 16 proofs at four visits with 76 tactic requests and
16 kernel checks each. The four-attempt lexical control recovered 8 with
104 tactic requests and 8 kernel checks. The ranker spent 152 cell calls
per seed. Summing the original action times gives 12.14 and 13.62 seconds
of projected Lean work for the ranker and 15.83 for the lexical control.
These timings describe the recorded actions. Count ranking runtime
separately when deploying the fallback.

### Feedback placement

The matrix evaluated 74 cells at depths 4, 6, and 10, including all
training-versus-runtime schedule swaps at depth six. Every matched
depth-six placement recovered all 16 provable Lean goals in both seeds.
Their recorded Lean work ranged from 100 to 104 tactic requests and
16 kernel checks per seed. The following values use FineWeb-Edu and
OpenWebText for text and 128 held-out episodes for joint before/after
graph success with current-epoch verified reuse.

| Train and run at depth six | Text bits per byte, seed 137 / 163 | Verified joint graph success, seed 137 / 163 |
|---|---:|---:|
| Direct | 4.6549 / 4.6477 | 42 / 15 |
| Answer RLT | 4.7114 / 4.8851 | 35 / 1 |
| Answer damping | 4.7298 / 4.9057 | 44 / 0 |
| Reasoning RLT | 4.6397 / 4.6835 | 26 / 1 |
| Reasoning damping | 4.6450 / 4.6485 | 1 / 1 |

Answer interpolation was sensitive to training seed. Reasoning
interpolation traded away verified graph success and gave no consistent
text gain. Swapping direct weights to answer RLT at runtime moved graph
success from 42 to 31 on seed 137 and from 15 to 25 on seed 163. Keep
interpolation opt-in by task and boundary, and test a fitted schedule
alongside its runtime swap before promoting it. Direct feedback keeps
the default transport path.

The exact graph baseline solves both turns by traversal and uses 1,508
hops over 128 episodes. At depth six the direct model with verified
reuse used 963 checker hops in each seed plus 1,244/1,435 neural visits
and 1,276/1,450 verifier calls. Its terminal-only policy got 15 joint
episodes right in each seed. Verification and reuse yielded 42/15.
At depth ten the direct verified policy reached 50/45 joint episodes
with additional visits. Use the exact solver wherever the rule graph is
available. Within a learned route, recheck no-op certificates in the
current epoch. The no-op cache kept joint coverage and saved 32/45
neural visits relative to soft carry at depth six. Reset and carry
ranked differently on edits across seeds. Recompute after an edit by
default and keep learned carry gated by an independent test.

### Curriculum and data mix

| Direct training recipe at depth six | Text bits per byte, seed 137 / 163 | Verified joint graph success, seed 137 / 163 |
|---|---:|---:|
| Constant mixed | 4.6549 / 4.6477 | 42 / 15 |
| Text-first loss schedule | 4.6941 / 11.6622 | 0 / 0 |
| Text-heavier batches | 4.6829 / 4.6550 | 0 / 15 |

All three recipes recovered the same Lean ceiling. The constant mix
preserved more graph performance and had lower text loss in both
seeds. The text-first failure on seed 163 appeared on both text corpora.
Keep that outcome in the benchmark as a curriculum regression. These
two seeds and byte windows support the starting comparison and leave
the full-size text mixture and curriculum open to a new held-out test.

## Scaling gates

Promote a learned module or routing policy only after comparing quality
and work on unseen structures, separately by task and by training seed.
Preserve text competence, show a useful verified task effect, and count
neural visits, retrieval, verifier calls, and checker traversals. Compare
the fitted schedules and runtime schedule swaps before attributing an
effect to training. Record uncertainty and keep the direct and lexical
controls available where evidence is tied or inconclusive.

For staging, reproduce the direct Lean reference of 16 checked proofs
and 76 tactic requests within four visits before changing the policy.
Record all extra kernel work and measure routing latency separately.
For a new held-out set, compare against its own exact or lexical control
at the same budget. Require the verifier to reject stale and unsupported
commits, including no-op reuse. Compare text retention within each
corpus across both seeds and keep the curriculum failure in regression
tests. Review quality and the work vector jointly before adding a module.

The completed Lean collection and certification peaked at 180.4 MiB of
job commit. The mixed fit and scoring history peaked at 91.8 MiB. The
256 MiB cap applied to each process tree, and the receipts record
host-memory pauses separately from completed phases.

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
