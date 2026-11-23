# Native Storyworld courier transfer

On 144 frozen synthetic worlds, full shortest-action-set supervision
improved verified completion over canonical labels by **0.23 percentage
points** (95% paired root interval −2.08 to +2.78 points), below the
registered +2-point target. It saved **0.18 moves per attempt** (95%
interval 0.04 to 0.33), with the same number of premature STOPs. The
hash-selected arm had the most completions by a two-episode margin
over the action-set arm.

## Matched training

The committed protocol `5697d37f0adb782d1829ecd545e2afa92263434a`
allocated 200 training worlds in five topology families and 144 frozen worlds
in six different, non-isomorphic families. All 344 source hashes were
unique. Two shortest trajectories per training root passed native replay and
provided 1,452 distinct supervised states, including 132 with tied shortest
actions. All three arms saw the same states and legal masks. They differed in
the labels used on ties.

Each of the nine fits used 44,417 parameters, three weight-tied visits per
decision, 180 AdamW updates, and a batch size of 48 for 8,640 supervised
draws. Initialization and batch seeds 1101, 1102, and 1103 were matched
across arms. Mean final-ten-update training loss across seeds was 0.31379
for canonical, 0.27588 for hash-selected, and 0.28191 for action-set.
No development-selection step was used, and fit digests were recorded before
frozen scoring. The run used CPU PyTorch 2.7.1 and Hyperon 0.2.10.
[Pinned Looped Transformer](https://github.com/MoralityLabAI/LoopedTransformers/commit/950d4d594d87a1f74c6139a928a9b6cb4ab70617)
[Pinned Storyworld adapter](https://github.com/MoralityLabAI/metta-storyworld/commit/4cfe1713d73b4e508833b668477921adfbe570a6)

## Frozen outcomes

Success required native-confirmed cargo delivery and a policy STOP within the
12-move limit. An independent Python reference checked native legal
successors and replay results. Every model received a preverified legal
action mask. There were zero invalid routes.

| Supervision | Verified completion | Shortest first action | Premature STOPs | Move cap reached without STOP | Moves | Recurrent visits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A · Canonical | 387/432 (89.58%) | 384 | 44 | 1 | 2,433 | 8,595 |
| B · Hash-selected | 390/432 (90.28%) | 381 | 42 | 0 | 2,385 | 8,451 |
| C · Action set | 388/432 (89.81%) | 390 | 44 | 0 | 2,355 | 8,361 |
| Exact shortest-route control | 144/144 | 144 | 0 | 0 | 754 | 0 |

| Paired contrast | Completion change in points (95% interval) | Moves saved per attempt (95% interval) |
| --- | ---: | ---: |
| C vs. A · Registered primary | +0.23 [−2.08, +2.78] | +0.181 [+0.037, +0.326] |
| C vs. B | −0.46 [−2.78, +1.85] | +0.069 [−0.039, +0.183] |
| B vs. A | +0.69 [−1.85, +3.24] | +0.111 [−0.035, +0.264] |

Intervals average seed differences within each source root and bootstrap
paired roots within each fixed family (4,000 draws, seed 2101). Action-set
completed 59/72 permit-corridor trials, compared with 63/72 canonical and
64/72 hash-selected. These family counts are descriptive. The aggregate
completion interval still covers the registered target, so this pilot leaves
open whether a gain that large would replicate. The move savings apply to
this controller with a shared, precomputed native mask.

Training made 4,766 native successor queries and 400 replays. Frozen scoring
made 3,579 shared successor queries and 1,440 replays, one per trial
including all exact controls. Each learned arm used 432 final replays.
Action-set made 234 fewer recurrent visits than canonical and used 21.55
visits per completion compared with 22.21. The run took 576.08 seconds
at 396.82 MiB peak RSS under the enforced 8 GiB CPU process ceiling.

## Receipt digests

The artifact audit reconciled all 400 verified training presentations,
1,440 frozen rows, and score counts against the locked allocation.
The run directory is `runs/metta/storyworld-courier-5697d37/`. Raw traces,
root lists, and fit details are local generated artifacts.

| Artifact | SHA-256 |
| --- | --- |
| Protocol and source allocation | `3690627b0139d4dc3de4d823427b5cbcb0137e94d898a7a067cb0d31a6102030` |
| Fit records with per-model digests | `8cce365056baf9d1484064dfa7f4398d93f11a180987bf52bc75f70e22161b3b` |
| Training frames | `afdf248db150394f90c364c6a29ec437a00fb15df9c5c93b418fa21fd5d8fd9f` |
| Verified training trajectories | `392f3d398de0be558d76640ca982606a92b7ab13f9d7219d5b56490c25d3758a` |
| Frozen replay receipts | `88419586c30ce3d750f8be2bc81b07b635747242c5620ad19f39f7d3d2be1b79` |
