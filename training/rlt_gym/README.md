# RLT gym data for the compact learner

The [RLT rollout gym](../../docs/rlt_rollout_gym.md) supplies checked task attempts, visible teacher revisions, and local recurrent-state summaries for the DataGoblin training program. Its first adapter fits a 17-parameter readout over frozen recurrent features. It provides an engineering control while the compact-model adapter is prepared.

Start a collection with the documented `make-suite`, `run`, and `replay` commands. The default suite has 140 synthetic tasks across seven families and five custody splits. Keep direct recurrence as the initial baseline and retain the constant family mixture. Cross the fitting and runtime schedules when investigating RLT or matched damping.

## Collection sequence

1. Freeze a suite and run the matched CPU matrix. Use training rollouts to identify failures and calibration rollouts to choose the next experiment. The report retains coverage by family and the work spent reaching each checked answer.
2. Stage a broad Codex batch with `teacher-prepare`. Add tasks from `frontier` when a family needs more examples. `teacher-repair` supplies an observed native failure and its checker feedback so a teacher can demonstrate a repair.
3. Import every response file. Preserve invalid attempts and unknown usage through the receipt pipeline. Continue unfinished episodes within their planned visit budgets. A separate Opus batch can add independent proposals on a selected development slice.
4. Export the five training streams and inspect their counts by family, lineage, producer, and outcome. Fit the next compact-model candidate from admitted training examples. Carry the previous candidate into the next comparison.
5. Expand the task registry with source-pinned Lean goals, rule changes, and research-plan checks. Keep future evaluation roots outside teacher batches and training exports. Follow the [Aleph experiment protocol](../aleph_protocol.md) when moving to a trained recurrent architecture.

## What enters training

`sft.jsonl` contains verified answers. `repairs.jsonl` retains the prior failed context and a checked revision; it can teach the learner how to recover after specific feedback. The transition stream preserves attempts that later become useful for reward or routing models. Outcome rows place the prediction point before the final checker result. Preference pairs require identical public context and independently checked candidate ordering.

The export manifest binds each stream to its source rollouts and frozen suite. Every row identifies its task lineage and producer. Teacher explanations remain attributed claims, while the environment supplies answer labels. Native smoke controls require an explicit export flag. Their labels remain attached through mixed native-to-teacher repair episodes.

## Connection to BPD

The recurrent trace records where an update occurred and how strongly it moved the carried state. Pulse hooks attach a chosen direction or a random control to that boundary. These records establish the observation points needed for future BPD component swaps and intervention prediction. The BPD mechanism flag remains false until its causal study meets its own gates.

A compact learner can first learn to select a useful action from the visible task and feedback. Later experiments can ask whether its internal state carries the predicted mechanism and whether a targeted intervention changes the next readout. Evaluate that question on held-out lineages with dose-matched controls, using the same task and verifier contracts throughout.

## Storage and reproducibility

Keep generated datasets, model snapshots, teacher replies, and run receipts in ignored `runs/` directories or the experiment archive. Commit the source and frozen protocol that reproduce them. Capture the actual teacher model/session identifier and available token usage with each collection. The offline queue commands prepare files for a separately budgeted teacher runner.

The complete command reference, schemas, artifact descriptions, and extension paths are in [`docs/rlt_rollout_gym.md`](../../docs/rlt_rollout_gym.md).

The [campaign exporter](../../docs/rlt_rollout_gym.md#campaign-collection-and-aleph-export) produces six response-only streams and checks lineage custody across every registered suite. Treat verified first-attempt and repair targets as candidate compact-model supervision. Apply the planned four-presentation limit across all views of a root. Count experiment-plan roots under research and mechanistic interpretation and the other roots under procedural games. For recurrent policy distillation, compare the cost and coverage of first-valid stopping with the final answer policy after an independent checker result. Preserve the frozen reservoir's control label and evaluate any transferred policy on a separate model and held-out roots.

The [21 November campaign](2026-11-21_campaign.md) records the matched CPU results and the tested early-stop policy.

## K-depth hypergame research taste

The [hypergame-taste collector](hypergame_taste/README.md) adds an exact CPU reference taskset for recursive subjective-belief worlds and budgeted research decisions. It emits separate sealed states, public teacher packets, matched policy rollouts, train-only verifier-scored decision forks, and artifact digests with full replay. Start with K=0/1/2/4/8 and frozen per-root budgets; the companion [research hill-climbing skill](hypergame_taste/skills/research_hillclimb/SKILL.md) specifies the eventual RLT correction loop. These traces are synthetic controls and do not establish learned-model results.
