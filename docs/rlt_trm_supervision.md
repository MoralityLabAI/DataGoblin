# Supervised TRM feedback and spectral interventions

This study tests whether remaining-time feedback improves learning from
exact intermediate targets. The matched controls use direct feedback,
RLT, and damping with the same coefficient sum and final full update.
Each interpolated schedule acts on either the reasoning state or the
answer state. The other state uses direct feedback.

**Result.** With final-only labels, answer RLT improved the eight-step
holdout from 313 to 579 correct out of 864, with most of the gain coming
from one seed. With exact step labels, direct feedback, answer RLT, and
answer damping all reached 864 of 864 at every evaluated depth. The SVD
interventions show that the final answer could recover from a large
mid-loop disruption of the answer state while depending strongly on
where we perturbed the reasoning state.

## Cycle 1 — Register the comparison

The graph-conditioned TRM shares one update cell between a reasoning
state and an answer state. Each cycle passes a message from the node's
predecessor under the current relation, updates reasoning, revises the
answer, and reads out a score for each node. The request specifies the
relation at each cycle. The model sees the graph and starting node. A
separate walker computes the target node after every transition.

- **Successor** repeatedly follows the forward relation.
- **Predecessor** repeatedly follows the inverse relation.
- **Alternating** uses two forward relations in alternation.

Train at six cycles and score at four, six, and eight on unseen graphs.
Compare final-only labels with an additional stepwise objective of weight
0.5. The stepwise label is the node at the current cycle. For direct and
answer RLT, also assign the final label at every cycle with the same
auxiliary weight. This checks whether the effect comes from the content
of the intermediate label or the extra loss. Use three paired model
seeds and snapshot development behavior at optimizer updates 0, 16, 32,
64, 96, 128, and 160. Only development scores enter the learning curves.

Each task has 256 training graphs, 48 development graphs, 64 graphs for
fitting spectral directions, and 96 held-out graphs. Neither a sampled
relation nor its inverse repeats across the splits. The matched fits use
the same initialization, update count, and example order within each seed.

## Cycle 2 — Measure states and test the directions

Fit an SVD separately at each cycle to reasoning states, answer states,
and realized answer updates. Subtract each graph's node-wise mean before
decomposition so a feature shared by all nodes contributes no variance.
Fit directions on the spectral split and measure their energy on the
held-out graphs. Record singular values, effective rank, RMS, and the
overlap of consecutive leading two-dimensional subspaces. On direct and
answer-RLT weights, also swap only the runtime feedback schedule and
repeat the spectral measurement.

After cycles two and four, attenuate the leading two directions of either
carried state by a half or fully and continue the rollout. Compare with three random
controls matched on the size of each example's perturbation and its
centered state norm. Measure final errors introduced and repaired on the
same held-out items. Report the coverage of this matching and both dose
residuals with each intervention.

The launcher runs one worker under a 256 MiB Windows process-tree cap.
Use the same local seed and run directory to resume. The run records
the dataset and protocol digests, fits and optimizer-cycle snapshots,
per-task held-out cycle counts, singular spectra, intervention outcomes,
and resource receipts.

```powershell
python scripts/run_rlt_supervision_windows.py fit --out RUN --data-seed SEED --ramcap-package LCGRT_ROOT
python scripts/run_rlt_supervision_windows.py evaluate --out RUN --data-seed SEED --ramcap-package LCGRT_ROOT
```

The runner uses the repository's `rlt-pilot` optional dependencies.

## Cycle 3 — Score the matched runs

All 36 fits have 552 trainable parameters, 160 optimizer updates, and
two shared cell calls per visit. Each row pools three paired fits on the
same 288 held-out graphs. The resulting denominator is 864 model–graph
trials per depth. The four- and eight-cycle tests ask for a different
number of transitions from training. They measure execution-length
generalization for the same relation rules.

| Supervision | Feedback | Four visits | Six visits | Eight visits |
|---|---|---:|---:|---:|
| Final only | Direct | 575 | 864 | 313 |
| Final only | Answer RLT | 576 | 864 | 579 |
| Final only | Answer damping | 321 | 864 | 375 |
| Final only | Reasoning RLT | 89 | 624 | 49 |
| Final only | Reasoning damping | 79 | 646 | 71 |
| Stepwise | Direct | 864 | 864 | 864 |
| Stepwise | Answer RLT | 864 | 864 | 864 |
| Stepwise | Answer damping | 864 | 864 | 864 |
| Stepwise | Reasoning RLT | 855 | 741 | 61 |
| Stepwise | Reasoning damping | 576 | 863 | 562 |
| Final at every visit | Direct | 288 | 864 | 504 |
| Final at every visit | Answer RLT | 288 | 864 | 671 |

For final-only supervision at eight visits, direct scored 25, 288, and
0 out of 288 by seed. Answer RLT scored 288, 288, and 3. Stepwise direct
and stepwise answer RLT both solved every six-cycle development example
by update 32. Repeating the final label at every visit helped some
eight-cycle fits, while exact intermediate labels produced the ceiling
at all three depths for each direct-reasoning placement.

### Cycle log for intermediate predictions

These counts score the node reached at the *current* cycle on the
six-cycle holdout. Each column has 864 model–graph trials.

| Supervision and feedback | Cycle 1 | Cycle 2 | Cycle 3 | Cycle 4 | Cycle 5 | Cycle 6 |
|---|---:|---:|---:|---:|---:|---:|
| Final only, direct | 0 | 288 | 0 | 575 | 576 | 864 |
| Final only, answer RLT | 0 | 288 | 0 | 576 | 576 | 864 |
| Stepwise, direct | 864 | 864 | 864 | 864 | 864 | 864 |
| Stepwise, answer RLT | 864 | 864 | 584 | 864 | 864 | 864 |
| Stepwise, answer damping | 864 | 864 | 584 | 864 | 603 | 864 |
| Stepwise, reasoning RLT | 864 | 698 | 704 | 728 | 742 | 741 |
| Final at every visit, answer RLT | 0 | 288 | 0 | 296 | 96 | 864 |

These readings expose errors that terminal accuracy misses. In
particular, stepwise answer RLT recovered from its third-cycle errors by
the fourth cycle. Stepwise reasoning RLT also made errors at the last
cycle and extrapolated poorly to eight steps.

## Cycle 4 — Interpret the SVD and controlled interventions

On fixed direct-trained weights, switching to answer RLT reduced the
first-cycle centered answer RMS to exactly one sixth. This isolates the
effect of the feedback coefficient at the first visit. On independently
trained stepwise arms, mean second-cycle centered answer RMS was 0.480
for direct, 0.108 for answer RLT, and 0.148 for answer damping. The top
two answer directions covered at least 99.5% of held-out node-contrast
energy in all three arms at that cycle. Switching stepwise answer-RLT
weights back to direct at inference reduced six-cycle success from
864 to 509. The fitted weights and feedback schedule can co-adapt even
when their matched terminal accuracy ties a direct fit.

For stepwise answer RLT, fully removing the leading answer directions
after cycle two or four preserved all 864 final answers. Fully removing
the leading **reasoning** directions after cycle two left 115 correct.
The three equally sized random controls left 302, 543, and 574 correct.
For stepwise direct feedback, removing the leading reasoning directions
after cycle four left 107 correct, versus 482 to 655 for its random
controls. Smaller half-strength removals at those boundaries recovered
all 864 final answers. The full-removal effect varied by seed and cycle.
These interventions identify which carried state the model relies on
at the tested boundaries. They also show why variance in a readable
answer state cannot by itself identify the computation the next loop uses.

Use stepwise direct as the baseline when exact transition labels are
available. Answer RLT earns a comparison when the fit gets only final
labels. Keep the reasoning update direct for this one-edge message path.
The evidence covers small permutation graphs with a supplied relation
schedule.

The cycle log has 252 optimizer snapshots and 1,944 held-out rows. The
trace files hold 972 spectra, 288 intervention panels, and the runtime
swaps. All 864 random controls matched all 288 held-out examples. The
largest absolute error in the displacement or centered norm checks was
`5.33e-15`.
Fitting peaked at 62.7 MiB and evaluation at 91.2 MiB. One evaluation
pause recorded the host-memory limit, and scoring resumed from the
completed traces.
