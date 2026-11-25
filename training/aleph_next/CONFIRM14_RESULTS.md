# Does global constant damping earn Aleph's second 10B-token run?

**Hold the second slot. Global constant damping did not earn a 10B-token run.**

Six physically shared 14,179,584-parameter networks completed the matched comparison locally on CPU. Each used one prelude, ten shared core blocks traversed twice, and three final blocks. A applies the full residual update at all 24 visits. B takes 0.12069383 of each of the first 23 block updates and the full last update. The architecture and parameter count otherwise match. No credits were spent.

Keep A as the dense two-pass reference and leave B's paid slot open. B reduced text loss by 0.39%, below the 1% minimum. Across all six checkpoints, none got all four answers correct on a single unseen repair orbit or correctly answered both turns on a changed graph. Development qualification transferred to byte prediction; the key unresolved capability is carrying a corrected state through subsequent steps.

## Fresh comparison

| Matched seed | A text NLL | B text NLL | B text gain | A checked | B checked | B checked gain (pp) |
|---:|---:|---:|---:|---:|---:|---:|
| 137 | 2.8657 | 2.8624 | 0.11% | 13.41% | 12.11% | -1.30 |
| 163 | 2.8538 | 2.8854 | -1.11% | 12.50% | 13.28% | +0.78 |
| 191 | 2.9280 | 2.8660 | 2.12% | 12.63% | 12.50% | -0.13 |

Across seeds, mean text loss was 2.8825 for A and 2.8713 for B in nats per byte. B's relative loss reduction was 0.39% (95% interval +0.28 to +0.49%). Its gain in checked success was -0.22 percentage points (95% interval -1.56 to +1.09 pp). Positive values favor B on both measures.

The held text covered 12,288 next-byte predictions in 12 books by distinct authors, with four authors in each of three historical English prose strata. The frozen training-byte unigram scored 3.0729 and the training-only interpolated trigram scored 2.2304. The independent byte-loss reconstruction agreed for every seed and both controls. Exact stratified author resampling gave a text-gain interval of +0.28 to +0.50%.

### Reasoning transfer

| Task | A | B | Strongest registered cue | Cue score |
|---|---:|---:|---|---:|
| Unseen register programs | 24.91% | 24.22% | anchor then two tail adds | 65.62% |
| Fresh graph revisions | 0.78% | 1.04% | patch with one tail hop | 64.84% |
| Earlier graph benchmark, diagnostic | 0.00% | 0.52% | patch with one tail hop | 67.19% |

| Fresh family or regime | A | B | Best cue within family |
|---|---:|---:|---:|
| repair / initial_reset | 23.26% | 24.65% | 29.17% |
| repair / interior_reset | 25.35% | 24.65% | 100.00% |
| repair / multiple_resets | 25.69% | 24.31% | 100.00% |
| repair / no_reset | 25.35% | 23.26% | 33.33% |
| revision / length_ood | 1.56% | 1.04% | 59.38% |
| revision / revision_ood | 0.00% | 1.04% | 70.31% |

A repair orbit varies one register across all four possible values while holding the proposal and suffix fixed. Each orbit has three wrong proposals. The held set has 96 orbits across four reset patterns and three unseen program lengths. The 64 fresh graph lineages each contribute a changed patch and a no-op. A patch earns credit only if both its before and after answers are correct. The checked score weights repair and revision equally. Replaying 64 archived graph lineages supplies a separate diagnostic.

## Was the baseline competent?

The first qualifying direct horizon was 128 updates after the published 32-update prefix. Both arms restarted AdamW once because only weights survived from the prefix. A alone selected the stopping point. On the original development set each qualifying seed had to beat 3.2357 nats/byte over 4,096 positions and exceed the 59/96 last-operand cue on full native repair traces. These models trained on a small mixture of native text, Lean, counterexamples, edits, rule revisions and verified short programs.

| Seed | A dev NLL | B dev NLL | A dev repairs | B dev repairs | A / B dev gate |
|---:|---:|---:|---:|---:|---|
| 137 | 3.1015 | 3.1183 | 74/96 | 71/96 | pass / pass |
| 163 | 3.1010 | 3.1278 | 77/96 | 85/96 | pass / pass |
| 191 | 3.1753 | 3.1133 | 74/96 | 64/96 | pass / pass |

| Held competence | Lane | Gain over strongest cue (pp) | 95% interval (pp) | Seeds above cue | Gate |
|---|---|---:|---:|---:|---|
| A | repair | -40.71 | -74.65 to -7.81 | 0/3 | fail |
| A | revision | -64.06 | -74.22 to -54.69 | 0/3 | fail |
| B | repair | -41.41 | -75.61 to -8.07 | 0/3 | fail |
| B | revision | -63.80 | -73.44 to -54.95 | 0/3 | fail |

The direct development gate checks the last marked transition of a displayed trace. Fresh register repair requires executing an entire unseen program. The independent graph task requires tracing a modified graph. Development competence and transfer therefore answer different questions.

Two development-only comparisons put the qualification threshold in context. A 32-entry lookup fitted to the training triples (previous state, operation, operand) recovered all 96 native answers. A byte trigram fitted only on training text achieved 2.4106 nats/byte on development. The neural systems face the stronger independent reasoning controls above on fresh families.

A useful next CPU gate is to score a matched pair of programs that share the same public suffix but have different correct outputs after an upstream change. Train with verified intermediate states, withdraw those hints at test, and hold out whole program templates and graph lineages again. This would test whether training can teach state propagation before spending credits on another placement.

## Spending rule

| Registered gate | Result |
|---|---|
| Development competence, A / B | pass |
| Fresh competence, A | fail |
| Fresh competence, B | fail |
| Text regression bounded by 1% loss | pass |
| Checked regression bounded by 2 pp | pass |
| At least 1% text improvement, positive lower bound and two favorable seeds | fail |
| At least 2 pp checked gain, positive lower bound and two favorable seeds | fail |
| B/A controlled CPU throughput at least 90% | pass |

Median B/A useful-position throughput was 93.61% by single-process CPU time and 85.03% by elapsed wall time on the same core. The registered overhead gate uses process time. The timings come from short, interleaved forward/backward/AdamW updates. They do not estimate TPU cost. The two benefit routes are alternatives. All competence, throughput and non-regression gates are required.

The primary paired bootstrap used 20,000 draws. It kept whole repair orbits and graph lineages together, resampled families within tasks and authors within text strata, and conditioned on the three matched seeds. Also resampling matched seeds gave a relative text-gain interval of -1.09 to +2.08% and checked-gain interval of -2.08 to +1.82 pp. Four repair families and two native graph regimes limit extrapolation to other research gyms. This is a 14M miniature trained on short byte windows.

## Training ledger and reproduction

| Seed | Prefix positions | Continuation positions | Continuation padded | Supervised labels | A training seconds | B training seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 137 | 11,523 | 40,702 | 53,014 | 6,496 | 961.3 | 818.7 |
| 163 | 11,492 | 40,643 | 52,812 | 6,496 | 803.4 | 788.8 |
| 191 | 11,471 | 40,325 | 53,196 | 6,496 | 789.9 | 856.1 |

Checkpoint hashes, corpus hashes, matched training streams, negative horizons, exact cue scores and paired interval sensitivities are in the companion receipts. The original six numerical prefixes, native data pack and private panel draws are required for bit-for-bit replay. The panel builders support fresh independent draws. Raw held cases, draw keys and weights stay local.

The full repository run passed 649 tests with 14 skipped. Ruff and the placement smoke test passed. The evaluator was fixed at ad34510 before fresh scoring. Use source commit 84394be for replay. [Reproduction commands](CONFIRM14_REPLAY.md) and [machine-readable decision](CONFIRM14_DECISION.json).
