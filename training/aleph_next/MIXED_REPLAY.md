# Native mixed-corpus replay

The source-reread study registered a width-80, two-pass, 192-update continuation on the frozen mixed corpus. This replay uses its seeds 137, 163 and 191, 48 evaluation roots per group and six probe roots. It compares one prelude reread with an equally strong same-site read of the current pass input. Direct and global constant damping reuse the archived matched-width, seed and stream fits. Check the full training-plan digest, corpus hash, native metrics and token counts before joining the archived controls. If those archived cells are absent, the original study entry point can train all four arms.

The code change relative to the archived controls reads the saved state only inside the two explicit reread arms. The smoke test checks causality and the one-pass identity. A separate replay compares outputs and parameter gradients for both archived controls against the original source on the same synthetic tensor. The original benchmark was visible when the continuation was registered. Results are development evidence and the text metric is natural-log loss per byte.

```sh
python training/aleph_rlt_placement/cpu_worker.py --wall 10800 study.py --pack runs/placement/mixed --out ../aleph-next/mixed192 --width 80 --steps 192 --r 2 --arms prelude_reread site_damp --seeds 137 163 191 --roots 48 --probe-roots 6
python training/aleph_next/mixed_release.py --new ../aleph-next/mixed192 --controls runs/placement/study
```

The frozen corpus comes from the DataGoblin evidence branch described in the earlier placement report. Run its intake and verification procedure before starting a new independent campaign. The scored tasks are byte-level text, verified Lean candidate choice, rule revision, reasoning repair and counterexample selection. The corpus-specific lexical and cue baselines are available in every cell receipt.
