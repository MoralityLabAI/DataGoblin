# STOP_DIAGNOSIS: why the MeTTa RAM hill climb never produces solved_and_stopped

Campaign: INK-FP-AUDIT-03. Branch `research/metta-stop-01`. CPU only, NumPy, peak working set 43 MiB (psutil `peak_wset`, limit 1024 MiB).

## Verdict

The failure is solution generation, not stopping. Rolled out with the environment deciding when to stop (oracle stop), the default students reach the exact optimum in 0 to 2 of 180 fresh cases, so no stopping change can lift `solved_and_stopped` above that ceiling. Teacher-forced on the canonical edit path, the default TRM picks the canonical next edit 28.5% of the time, falling to 0% from the sixth canonical edit on. Free-running, it oscillates (add x, remove x, add x, ...) and never reaches the optimum.

Stopping is a second, independent defect: with the action-class STOP the 16-step TRM stops at every state it visits (stop rate 681/681 at unsolved states), which matches the 15 of 15 premature stops in INK-FP-AUDIT-02 section 9. Separate stop heads cut premature stops to 18 of 180, but there is nothing solved to stop on.

The first nonzero `solved_and_stopped` appears only when generation improves: the hybrid LDT-TRM at 200 steps, hidden 48, lr 0.3 solves and stops 6 of 180 fresh cases (17 of 180 reach the optimum). That is a 200-step budget, not matched to the 16-step default.

## Setup

- Harness: `MeTTa/stop_diagnostics.py`. Three arms per case: `never` (STOP masked, 16-step horizon), `oracle` (the episode ends the first step the selection equals the exact optimum), `own` (the student's stop mechanism through the unchanged `_next_action`, 8-step horizon as in `_rollout`). Every visited state records solved, canonical next edit, the student's action and the remaining canonical edits.
- Variants: `MeTTa/student_variants.py` (new `VariantNet`; the plain spec reproduces `RecurrentStudent` weight for weight, checked in `tests/unit/test_metta_stop_diagnostics.py`). `ram_hillclimb.py` and `airis02_*.py` are unchanged.
- Training data for the matched comparison: the bundle's 24 train roots per family (459 presentations, 120 roots), seeds 20000..20023. Same steps, lr 0.2, hidden 12 and 4 unrolls for every variant. Training seeds 20261122, 20261123, 20261124.
- Fresh evaluation roots: seeds 50000..50011 per family (12 per family, 60 roots, 180 cases over 3 training seeds). The bundle uses train 20000+, dev 30000..30007 and public test 40000..40007, so the fresh band is disjoint (asserted in `fresh_dev_rows`).
- `initial_solved` was 0 on every evaluated set, so no case is trivially solved; "ever reached optimum" is a real arrival.
- Unnecessary-visit and calibration counts are small samples. Stop rates are over canonical-trajectory states (180 solved, 681 unsolved).

## 1. Generation versus stopping, existing students

Aggregated over 3 training seeds (120 bundle-dev cases, 180 fresh cases). Source: `python -m MeTTa.stop_diagnostics`.

| config | arm | set | cases | ever reached optimum | solved, oracle stop | solved_and_stopped (own stop) | premature stops | never stopped | teacher-forced correct | stop at solved | stop at unsolved |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | trm | bundle dev | 120 | 0 | 0 | 0 | 120 | 0 | 0.268 | 120/120 | 447/447 |
| default | trm | fresh | 180 | 2 | 2 | 0 | 180 | 0 | 0.285 | 180/180 | 681/681 |
| default | ldt_proxy | bundle dev | 120 | 2 | 2 | 1 | 119 | 0 | 0.309 | 78/120 | 310/447 |
| default | ldt_proxy | fresh | 180 | 0 | 0 | 0 | 180 | 0 | 0.294 | 94/180 | 387/681 |
| train_calibrated stop | trm | fresh | 180 | 2 | 2 | 0 | 31 | 149 | 0.285 | 23/180 | 40/681 |
| train_calibrated stop | ldt_proxy | fresh | 180 | 0 | 0 | 0 | 31 | 149 | 0.294 | 23/180 | 40/681 |
| steps 32 | trm | fresh | 180 | 0 | 0 | 0 | 180 | 0 | 0.279 | 158/180 | 614/681 |
| steps 32 | ldt_proxy | fresh | 180 | 2 | 2 | 0 | 179 | 1 | 0.304 | 94/180 | 338/681 |

Findings:

- Oracle-stop solved equals ever-reached in every row. A student that gets to the optimum would stop under an oracle; the others never get there. Generation is the bottleneck.
- The default LDT proxy produced one `solved_and_stopped` of 120 bundle-dev cases (one seed; not reproduced on fresh roots).
- Per-step correctness, default TRM, seed 20261122, fresh roots, teacher-forced canonical step 0: 18/60, step 1: 22/56, step 2: 13/47, step 3: 4/33, step 4: 0/22, step 5: 0/8. Free-running the same model scores 18/60 at step 0, then alternates 30/60 and 20/60 for 15 steps: the add/remove oscillation.
- Train-set edit accuracy is 0.358 (TRM, 16 steps) and 0.373 (200 steps): the student underfits its own 459 training rows, so this is not only a generalization gap. The canonical next edit (remove the lowest-index item outside the optimum) depends on the exact optimum, which the public features do not expose and a width-12 tanh loop does not compute.
- Training views cover canonical steps 0 to 2 and the optimum only (`training_views`), so states at canonical step 3 and later never appear in training.

## 2 and 3. Variants on matched compute, fresh roots

All rows use the same 459 train presentations, the same optimizer steps, 3 training seeds and 60 fresh roots. `original_trm` is the existing path (`fit_candidate`); `plain_class_stop` is `VariantNet` with no extras and matches it exactly. The oracle-stop ceiling of any variant is its `solved, oracle stop` column; the best ceiling in the matched tables is 2 of 180 at 16 steps and 9 of 180 at 200 steps.

Variant definitions:

- `sep_stop_head`: sigmoid halting head on the loop state, train-only threshold (5% train false-stop rate), STOP removed from the action classes.
- `existing_stop_head_trm`: the repo's `StopHead` on raw features.
- `aux_correctness`: adds per-action "reduces distance to optimum" heads and a remaining-edit head on the shared loop.
- `delayed_stop_curriculum`: STOP rows enter the action loss after half the steps and ramp in.
- `verifier_conditioned_halt`: halting head also reads valid, violation count, goal coverage, cost and quality from the public checker.
- `explicit_memory`: previous action, step counter and edited-item history as inputs.
- `ldt_trm_hybrid`: linear proposal logits plus recurrent residual, routed through the environment veto.
- `budget_aware_halt`: stops when a learned remaining-edit regressor rounds to 0.

#### 16 optimizer steps (180 cases = 60 fresh roots x 3 seeds)

| variant | ever reached optimum | solved, oracle stop | solved_and_stopped | premature stops | never stopped | unnecessary visits (never / own arm) | teacher-forced correct | on-policy correct | stop at solved states | stop at unsolved states |
|---|---|---|---|---|---|---|---|---|---|---|
| original_trm | 2 | 2 | 0 | 180 | 0 | 2 / 0 | 0.285 | 0.448 | 180/180 | 681/681 |
| original_ldt_proxy | 0 | 0 | 0 | 180 | 0 | 0 / 0 | 0.294 | 0.460 | 94/180 | 387/681 |
| plain_class_stop | 2 | 2 | 0 | 180 | 0 | 2 / 0 | 0.285 | 0.448 | 180/180 | 681/681 |
| existing_stop_head_trm | 2 | 2 | 0 | 31 | 149 | 2 / 1 | 0.285 | 0.448 | 23/180 | 40/681 |
| sep_stop_head | 0 | 0 | 0 | 18 | 162 | 0 / 0 | 0.275 | 0.447 | 11/180 | 39/681 |
| aux_correctness | 0 | 0 | 0 | 18 | 162 | 0 / 0 | 0.285 | 0.456 | 12/180 | 40/681 |
| delayed_stop_curriculum | 2 | 2 | 0 | 120 | 60 | 2 / 2 | 0.266 | 0.436 | 0/180 | 39/681 |
| verifier_conditioned_halt | 0 | 0 | 0 | 155 | 25 | 0 / 0 | 0.285 | 0.467 | 0/180 | 35/681 |
| explicit_memory | 2 | 2 | 0 | 180 | 0 | 2 / 0 | 0.307 | 0.464 | 180/180 | 681/681 |
| ldt_trm_hybrid | 1 | 1 | 1 | 179 | 0 | 1 / 0 | 0.282 | 0.457 | 100/180 | 404/681 |
| budget_aware_halt | 2 | 2 | 0 | 0 | 180 | 2 / 2 | 0.280 | 0.441 | 0/180 | 0/681 |

Teacher-forced correctness by canonical step (correct/states), original_trm and ldt_trm_hybrid:

- original_trm: step 0: 63/180, step 1: 69/168, step 2: 44/141, step 3: 16/99, step 4: 2/66, step 5: 0/24, step 6: 0/3
- ldt_trm_hybrid: step 0: 64/180, step 1: 65/168, step 2: 45/141, step 3: 18/99, step 4: 0/66, step 5: 0/24, step 6: 0/3

Train-set edit accuracy: {'ldt_trm_hybrid': 0.4071, 'original_trm': 0.3579, 'plain_class_stop': 0.3579}

#### 200 optimizer steps (180 cases = 60 fresh roots x 3 seeds)

| variant | ever reached optimum | solved, oracle stop | solved_and_stopped | premature stops | never stopped | unnecessary visits (never / own arm) | teacher-forced correct | on-policy correct | stop at solved states | stop at unsolved states |
|---|---|---|---|---|---|---|---|---|---|---|
| original_trm | 0 | 0 | 0 | 180 | 0 | 0 / 0 | 0.294 | 0.494 | 100/180 | 394/681 |
| original_ldt_proxy | 7 | 7 | 0 | 129 | 51 | 7 / 6 | 0.358 | 0.465 | 101/180 | 189/681 |
| plain_class_stop | 0 | 0 | 0 | 180 | 0 | 0 / 0 | 0.294 | 0.494 | 100/180 | 394/681 |
| existing_stop_head_trm | 0 | 0 | 0 | 42 | 138 | 0 / 0 | 0.294 | 0.494 | 42/180 | 48/681 |
| sep_stop_head | 0 | 0 | 0 | 25 | 155 | 0 / 0 | 0.298 | 0.504 | 22/180 | 44/681 |
| aux_correctness | 0 | 0 | 0 | 19 | 161 | 0 / 0 | 0.300 | 0.503 | 17/180 | 41/681 |
| delayed_stop_curriculum | 0 | 0 | 0 | 180 | 0 | 0 / 0 | 0.280 | 0.493 | 75/180 | 351/681 |
| verifier_conditioned_halt | 0 | 0 | 0 | 8 | 172 | 0 / 0 | 0.300 | 0.504 | 144/180 | 34/681 |
| explicit_memory | 0 | 0 | 0 | 140 | 40 | 0 / 0 | 0.292 | 0.503 | 152/180 | 135/681 |
| ldt_trm_hybrid | 9 | 9 | 0 | 139 | 41 | 9 / 4 | 0.348 | 0.466 | 99/180 | 195/681 |
| budget_aware_halt | 0 | 0 | 0 | 0 | 180 | 0 / 0 | 0.300 | 0.503 | 0/180 | 0/681 |

Teacher-forced correctness by canonical step (correct/states), original_trm and ldt_trm_hybrid:

- original_trm: step 0: 67/180, step 1: 66/168, step 2: 46/141, step 3: 18/99, step 4: 3/66, step 5: 0/24, step 6: 0/3
- ldt_trm_hybrid: step 0: 92/180, step 1: 76/168, step 2: 46/141, step 3: 17/99, step 4: 6/66, step 5: 0/24, step 6: 0/3

Train-set edit accuracy: {'ldt_trm_hybrid': 0.6785, 'original_trm': 0.3727, 'plain_class_stop': 0.3727}

Reading the tables:

- At matched compute nothing reaches a stable nonzero `solved_and_stopped`. The only nonzero cell is `ldt_trm_hybrid` at 16 steps (1 of 180, one seed).
- Separate stop mechanisms remove premature stops (original 180 of 180; sep_stop_head 25, aux 19, verifier-conditioned 8 at 200 steps) by turning them into never-stopped episodes, because the state is never solved.
- `ldt_trm_hybrid` is the only variant that improves generation at 200 steps: 9 of 180 reach the optimum against 0 for the TRM and 7 for the LDT proxy, with train edit accuracy 0.679 against 0.373.
- `budget_aware_halt` never stops at any state (0/180 solved, 0/681 unsolved): the remaining-edit regressor does not fall below 0.5 on this data.

## 4. Climb from the hybrid

`python -m MeTTa.stop_study --climb` runs a coordinate hill climb of `ldt_trm_hybrid` over steps (16..200), lr (0.04..0.30), hidden (8..48) and unrolls (2..6) on the bundle dev roots with training seeds 20261122 and 20261123 (2 passes, 28 candidates). Start: steps 16, lr 0.2, hidden 12, loops 4. Final: steps 200, lr 0.3, hidden 48, loops 4. Selection key: solved_and_stopped, oracle-stop solved, fewer premature stops, teacher-forced correctness. The winner was re-scored on fresh roots with training seeds 20261124..26 that never entered selection (180 cases):

| config | ever reached optimum | solved, oracle stop | solved_and_stopped | premature stops | teacher-forced correct |
|---|---|---|---|---|---|
| start (16 steps, hidden 12, lr 0.2) | 1 | 1 | 0 | 180 | 0.273 |
| climbed (200 steps, hidden 48, lr 0.3) | 17 | 17 | 6 | 103 | 0.380 |

The gain came from training length, width and step size, not from any stopping change. The registered grid (`OPTIONS`, steps up to 32) cannot express this configuration. The climb used 200 optimizer steps against the matched 16, so this is not a matched-compute result.

The same configuration (hidden 48, lr 0.3, 200 steps, seeds 20261124..26, fresh roots) for other variants:

| variant | ever reached optimum | solved, oracle stop | solved_and_stopped | premature stops | never stopped |
|---|---|---|---|---|---|
| plain_class_stop (TRM) | 15 | 15 | 1 | 176 | 3 |
| sep_stop_head | 9 | 9 | 3 | 57 | 120 |
| verifier_conditioned_halt | 8 | 8 | 4 | 18 | 158 |
| ldt_trm_hybrid | 17 | 17 | 6 | 103 | 71 |

Once generation is nonzero, the separate stop heads convert reached optima into correct stops at a higher rate than the class STOP (3 of 9 and 4 of 8 against 1 of 15). The hybrid reaches the most optima and stops correctly on 6 of 17 (35%); its stop calibration is the remaining defect (103 premature stops, 71 never stopped). Combining the hybrid's generator with a verifier-conditioned halting head was not run.

## Data lever

`python -m MeTTa.stop_study --levers` was stopped after four rows when the shared CPU stalled. Recorded `solved_and_stopped` counts (3 seeds x 60 fresh cases, 1000 steps, hidden 12): 24 roots per family: plain_class_stop 0, sep_stop_head 2; 100 roots per family with the 4-view quota: plain_class_stop 6, sep_stop_head 5. Training roots matter as much as architecture for generation.

## Reproduction

```
export PYTHONPATH="$PWD/src:$PWD" CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m MeTTa.stop_diagnostics --out runs/metta/stop_q1.json
python -m MeTTa.stop_study --out runs/metta/stop_study.json --seeds 20261122 20261123 20261124
python -m MeTTa.stop_study --climb --out runs/metta/stop_climb.json --seeds 20261122 20261123 20261124 20261125 20261126
python -m MeTTa.stop_study --levers --out runs/metta/stop_levers.json
python -m pytest tests/unit/test_metta_ram_hillclimb.py tests/unit/test_metta_airis02_student.py tests/unit/test_metta_airis02_stop.py tests/unit/test_metta_airis02_fresh.py tests/unit/test_metta_airis02_verifier.py tests/unit/test_metta_stop_diagnostics.py
```

Tests: 27 existing tests passed before the change; 39 passed after (27 existing plus 12 new).
