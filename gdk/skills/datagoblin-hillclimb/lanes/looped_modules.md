# Lane: looped_modules (looped and recurrent modules)

**Climbs** rlt_success_rate: verified successes of the RLT (recurrent feedback) student cells divided by
their episodes, on the calibration split of a frozen RLT suite, scored by the environment verifier.
Gate: up, min_delta 0.02, patience 4, max 20 moves. The eval contract is the suite manifest digest:
make the suite once and reuse its run id for every move.

## Baseline move

1. `rlt_make_suite` (`seed: 7`, `per_family: 4`). Keep its run id.
2. `rlt_run` with `suite: "run:<suite run>"` and defaults (depths 1 2 4 8, fit modes direct and rlt,
   modes direct and rlt, training_depth 4, readout carried_answer). The fit seeds are pinned to 7, 19
   and 91 for every move, so no move can win by choosing a lucky seed.
3. `record_result` on the rlt_run run, then `decide`.

`rlt_run` writes `summary.json` (cells per family, mode, depth) and `REPORT.md`. Read the cells to see
which family or depth fails before choosing the next change.

## Move menu (one change per move)

- `depths`: add 16 to test deeper recurrence, or drop 1.
- `training_depth`: fit at the depth you infer at, or shallower to test extrapolation.
- `training_modes`: `matched_damping` as the fit schedule instead of, or beside, `rlt`.
- `readout_site`: `endpoint` against `carried_answer`.
- `max_wall_seconds`: raise when the run reports a wall-clock cut.

The fit is deterministic on the pinned seeds, so rerunning identical parameters reproduces the value:
on `replicate`, record the rerun and decide (two equal results fall below the gate and reject).

Keep `rlt` in `modes`: the metric counts only rlt cells, and a run without them is invalid. Keep `direct`
in `modes` as the paired control.

## After a promotion

Selection uses calibration only. Held-out confirmation on ordinary_test and ood_test is a separate,
operator-run step (`docs/inkling_variable_loop_gym.md`, `inkling_controls`); report it as pending.
`rlt_replay` recomputes a run's trajectories and checker receipts for an audit. `gym_demo` shows the
architecture gym's mock LoRA, HRM-Text and looped controls; it never counts as evidence.

Looped models trained outside DataGoblin (LoopedTransformers, HybridTRMLDT TRM and HRM routers,
Inkling variable-loop students) have no recipe here: report their next run instead.
