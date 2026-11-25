# Native mixed-corpus replay

The source-reread study registered a width-80, two-pass, 192-update continuation on the frozen mixed corpus. This replay uses its seeds 137, 163 and 191, 48 evaluation roots per group and six probe roots. It compares one prelude reread with an equally strong same-site read of the current pass input. Direct and global constant damping reuse the archived matched-width, seed and stream fits. Check the full training-plan digest, corpus hash, native metrics and token counts before joining the archived controls. If those archived cells are absent, the original study entry point can train all four arms.

The code change relative to the archived controls reads the saved state only inside the two explicit reread arms. The smoke test checks causality and the one-pass identity. A separate replay compares outputs and parameter gradients for both archived controls against the original source on the same synthetic tensor. The original benchmark was visible when the continuation was registered. Results are development evidence and the text metric is natural-log loss per byte.

```sh
python training/aleph_rlt_placement/cpu_worker.py --wall 10800 study.py --pack runs/placement/mixed --out ../aleph-next/mixed192 --width 80 --steps 192 --r 2 --arms prelude_reread site_damp --seeds 137 163 191 --roots 48 --probe-roots 6
python training/aleph_next/mixed_release.py --new ../aleph-next/mixed192 --controls runs/placement/study
```

The frozen corpus comes from the DataGoblin evidence branch described in the earlier placement report. Run its intake and verification procedure before starting a new independent campaign. The scored tasks are byte-level text, verified Lean candidate choice, rule revision, reasoning repair and counterexample selection. The corpus-specific lexical and cue baselines are available in every cell receipt.

## Finalizing independent seed workers

The reducer checks the pinned dataset manifest and member bytes. It reconstructs each ordered 192-update sampling plan from the admitted corpus, including nonpadding positions, supervised positions, padded positions and lane counts. It recomputes the native and crossover metrics from scored rows and matches the row order to independently reconstructed 48-root and six-root panels. Old and new forwards must have bitwise-equal logits and gradients for every named parameter on two fixtures per archived control. The second fixture uses a real trained width-80 checkpoint, one native text row and one verified repair row.

Run the preflight while cells are still training. Add each independent worker output using a separate shard argument. A final checkpoint and complete receipt must match before a shard can contribute. An existing partial cell keeps its own directory. A fully verified external cell is copied atomically into the dedicated completed-shards subdirectory. The reducer selects the earliest completed receipt for each seed and arm. The atlas uses relative checkpoint paths inside the output tree.

```sh
python training/aleph_next/mixed_release.py --mode audit --new ../aleph-next/mixed192 --controls runs/placement/study --pack runs/placement/mixed
python training/aleph_next/mixed_release.py --mode release --new ../aleph-next/mixed192 --controls runs/placement/study --pack runs/placement/mixed --shard ../aleph-next/mixed163-shard --shard ../aleph-next/mixed191-shard
```

The final summary reports prelude minus local-site damping as its same-site source comparison, with direct and global constant damping as additional references. Its percentile intervals reuse the native public clusters and fix the three paired training seeds. Text clusters are windows, Lean clusters are seven goal families, and the remaining lanes share generator semantics across seeds.

## Results on the frozen public panel

All four arms had 1,414,480 physical parameters, two passes and 192 updates at each of seeds 137, 163 and 191. The six new fits used 3,116.5 summed wall seconds on shared CPUs. Native scores are seed means. Lower text loss and higher checked scores are better.

| Arm | Text nats/byte | Checked macro | Lean@4 | Joint revision | Repair | Counterexample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct | 3.06862 | 20.25% | 28.57% | 1.04% | 25.00% | 26.39% |
| Global constant damping | 2.96774 | 21.56% | 35.71% | 0.52% | 25.69% | 24.31% |
| Prelude reread | 3.05490 | 22.23% | 39.29% | 1.04% | 26.39% | 22.22% |
| Same-site pass-input damping | 3.06141 | 23.65% | 42.86% | 1.04% | 25.00% | 25.69% |

Prelude minus same-site damping was −0.00651 nats/byte, with a descriptive 95% public-cluster interval of [−0.01119, −0.00221]. The three seed differences were −0.03174, −0.00191 and +0.01411. The checked macro difference was −1.414 percentage points [−4.415, +1.017], with seed differences of −6.771, +2.083 and +0.446 points. The intervals condition on the fitted seeds.

Global damping achieved lower text loss than prelude reread at all three seeds. The mean prelude-minus-global difference was +0.08716 nats/byte [+0.07721, +0.09769] and +0.676 checked points [−5.906, +6.011]. Against direct, prelude reread gained 0.01372 nats/byte [0.00869, 0.01876] and 1.984 checked points [−1.736, +7.491]. These results support retaining global constant damping as a tournament control. They give no evidence for advancing prelude reread to paid training.

Both same-site arms used a coefficient of 0.25. Across 15 lane-by-seed traces, the imposed displacement RMS averaged 1.882 for prelude reads and 0.552 for pass-input reads. Source RMS averaged 0.547 and 5.830, respectively. On the six-root-per-group probe, switching frozen prelude-trained weights from the native read to a pass-input read moved text loss from 3.19363 to 3.19368. Switching pass-input-trained weights to prelude reads moved it from 3.19744 to 3.20146. At six inference passes, the two native-schedule losses rose to 3.27654 and 3.27547.

Release verification matched historical initialization for all three seeds, logits and all 128 named parameter gradients for both archived controls on both fixtures, final optimizer clocks, token ledgers and checkpoint hashes. All sixteen corpus-member byte hashes matched the pinned manifest.
