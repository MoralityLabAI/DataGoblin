# RLT research track

Start with a shared-loop Transformer and ask when the remaining-time update improves a task
at the same compute cost. Use HRM-Text-1B later as a frozen host and external comparator.
This track can measure the value of an RLT organelle inside HRM. A decision about retiring HRM
requires a separate checkpoint-scale comparison.

## What to reproduce

For a budget of K visits, the shared block proposes an endpoint h at visit k. RLT advances
the carried state x by `(h - x) / (K - k)`, with k starting at zero. The first update takes
1/K of the proposed displacement and the last takes the whole displacement. A constant
endpoint yields evenly spaced states. A moving endpoint changes the path, and an expanding
block can still amplify perturbations. The paper's guarantees depend on regularity of the
block. Its model also conditions on time and step size, supervises every block output and
decodes the final output. [Paper, sections 3-4 and appendices C-F](https://arxiv.org/html/2611.16265v2)

| Comparison | Evidence to collect |
|---|---|
| Published reference | The paper's 278M model reports lower Pile, FineWeb-Edu and OpenWebText perplexity at each matched depth. Ten-task mean accuracy rises at K=4 (42.38% to 43.30%) and falls at K=8 (44.36% to 43.53%). At K=32, the K=8-trained RLT retains much more performance than direct feedback in that experiment. |
| First known benchmarks | Freeze text windows for FineWeb-Edu and OpenWebText. Score LAMBADA, HellaSwag and both ARC splits with the paper's scoring convention. Preserve task-level results. Use exact released checkpoints and eval items if available. Otherwise label the results a reduced-scale replication with its own pinned data and recipe. |
| Mechanism controls | Train direct, RLT and constant damping from matched initializations with the same examples, token budget, recurrent block calls, time inputs and readout losses. Match damping's coefficient sum and last update to RLT. Report actual cumulative displacement too, since the coefficient sum does not match realized displacement or downstream compute. |
| Weight versus schedule | At inference, cross the checkpoint's training rule with the runtime rule. Report these cells separately from native inference. Sweep K=2,4,8,16,32 for K=8 training and start an independent trajectory for each K. A visit from the middle of a K=32 trajectory and the last visit of a short run have different time inputs and coefficients. |
| Scale versus direction | Measure the carried state and endpoint separately before normalization. Track residual RMS, realized update RMS, logit change, and perturbation gain before and after feedback. Pair perturbations by item, visit and random direction. Compare fixed absolute and relative doses and include task scores to distinguish a scale change from a useful computation. |

Our earlier small Lean runs found lower raw state growth, often matched by damping. Their
completion counts need replay through the repaired verifier. The interrupted grounded Lean run
trained direct-feedback rankers and did not produce a fresh RLT result. Use the clean Lean data
as a later transfer test with native automation and the identical-pool heuristic as controls,
at matched kernel calls and wall time.

## First novel evaluation

The rule-revision pack isolates task horizon and response to new evidence. Within each lineage,
two episodes share the same 32-link graph, initial vertex and hop count. One replaces a
single link on the queried path and the other reissues its existing destination. The model must
commit its first answer before it sees the update, then answer the same query under the new rules.
Compare fresh context with declared persistent state. For persistent execution, tag the correction
with a new evidence epoch and log whether the system invalidated stale state.

| Split | Hops | Edit position |
|---|---:|---|
| Training and development | 3-5 | Interior |
| Length holdout | 7-9 | Interior |
| Revision-position holdout | 3-5 | First or last |

All variants of a graph and start vertex stay in one split. Holdouts probe length and edit position
within this generator. Train and evaluate over distinct graph lineages with common formatting and
32 links at each horizon. Use development to pick difficulty and stop rules. Cluster intervals by
lineage instead of treating the two variants or multiple K values as independent examples.

Pass ordered `links`, `start` and `steps` to the first turn and ask for the endpoint after
exactly that many hops. Record the answer before showing `patch.source` and `patch.destination`.
On the second turn, replace only that link and ask again from the original start. Keep evaluation
targets with the scorer. Export one prediction object per item with `id`, `before` and `after`.
The scorer checks file hashes and exact ID coverage. It reports per-turn and joint success,
correctness on both variants, and stale answers when the original answer was correct. Its counts
are `[hits, denominator]`. The scorer sees final predictions, so the model runner must enforce
turn order and retain its turn-by-turn trace.

    python -m datagoblin.training_gym.rlt_revision build --out runs/rlt-fixtures --seed 17
    python -m datagoblin.training_gym.rlt_revision score --pack runs/rlt-fixtures --split development --predictions run.jsonl

Seed 17 is a public smoke example. Generate fresh packs with steward-held seeds and move
evaluation targets outside the model's workspace before a real assessment. The runner should
expose only the training partition for updates. Later families can add composed rules and a
separate source-clean Lean transfer test.

## Hybrid experiments

| System | Minimal causal comparison |
|---|---|
| TRM refinement loop | Apply remaining-time interpolation to one continuous refinement state. Match direct and damping at the same host, bridge, TRM invocations and readout. Keep the discrete answer and halt decision as separate outputs. |
| Typed checker | Add a checker at a named proposal boundary. Compare no checker, checker alone, direct plus checker and RLT plus checker. Interpolate latent state only. Apply a typed update only after verification with the current evidence epoch. Track precision, coverage, rejections, and fallback time. |
| Frozen HRM-Text-1B | After mask, scoring and replay fidelity checks, add a bounded residual organelle at a single named boundary. Compare native host, identity bridge, direct, damping and RLT with the same intervention capacity, dose, execution count and gradient access. Try independently trained module replacement only after this test. |

## DataGoblin evidence

Use paired rows as experiment evidence. Record parent checkpoint and source hashes, item and
lineage, training rule, runtime rule, model and seed, K at train and test, block and checker calls,
actual memory and time, token counts, final decisions and bounded per-visit summaries. Attach the
matched control and both outcomes, including regressions and aborted runs. Keep calibration,
curator reward, development and final reporting separate. Before training a curator on
architecture outcomes, repeat promising contrasts from shared checkpoint and optimizer snapshots.
Freeze promotion criteria before scoring the final split. The fixture CLI produces deterministic
task and checker evidence. It makes no architecture-promotion decision and requires no checkpoint.
