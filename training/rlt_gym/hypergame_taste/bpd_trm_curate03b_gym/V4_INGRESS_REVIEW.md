# GYM-01 V3 ingestion review: stage-local V4 contract

Pinned BPD e625db94ee7e78f03c83840c8bd3f834c2b86fa9, DataGoblin 16c3d3ede5da38a804736a31f55256be6172ceba. Preserve V3 hashes and traces as development behavior logs. No adapter-training promotion.

## Findings

1. gym_policy_eval._prompt includes all observed acquisition-stage summaries in one prompt, while SFT completion includes earlier acquisition actions. The action at stage k cannot observe future stages; this is temporal feature leakage for a sequential controller even though all data are TRAIN. Terminal SELECT_BATCH vs ABSTAIN preference may use the final prefix but must declare terminal scope.
2. SFT imitation includes F_shuffled_k4 and E_fast_analytic_k4 (no verified nonlinear gain), along with the expensive reference and rejected actions. Use arm and label type explicitly; exclude shuffled control and unverified updates from positive supervision. Keep them as behavior/control data. A verified rollback receipt confirms rejection, not that the original policy was good.
3. A rollback followed by the next window is a transition, not verified repair success unless the successor's outcome is independently positive and the lineage supports that claim. Existing repair traces do not check this. Preserve all transitions, mark successful recovery only with verified outcome and explicit parent lineage.
4. BPD gym_campaign eval_and_step records predicted_joint_native after opt.step, using the old gradient on new moments and parameters. Quarantine that label from V3. Predicted_joint_initial/final from pre-step search and actual acceptance are distinct.
5. E27/E31 generator finite rejection sampling does not fail closed. See GYM-02 audit in BPD. Exact original sampled partition violations require same-environment replay. Do not silently replace frozen data. V4 must record actual semantic split membership or direct enumerated support audit.
6. E29 zero positive accepted gain makes retention undefined (currently coded as 1.0). Per-task separately fitted adaptive replay retains E27 99.1821%, E28 99.6552%, E31 99.6467%, G05 99.3924%; per-task logistic 79.1584%, 99.4700%, 98.3831%, 94.6292%. Pooled result alone conceals failure. Unit costs pooled across roots estimate replay work and are not prospective policy CPU. Refit cost parameters on training roots or freeze independently calibrated costs for prospective evaluation.

## V4 export contract

- One stage-local state/action transition per observed decision; initial stage 0 has no candidate scores. At stage k expose only prefix[:k], costs incurred to date, task, controller recipe, parent checkpoint/optimizer digest, feedback cohort ID and stage-specific features available at decision time. Separate next-observation/outcome fields from model inputs.
- Terminal verified preference SELECT_BATCH vs ABSTAIN conditioned on acquired prefix and the exact selected batch. Preserve actual signed gain, tolerance and outcome provenance. Never infer unseen alternative batch gain. Abstentions audited with known probability remain separate counterfactual records; no negative label from unaudited rows.
- Root-grouped splits across all tasks/arms, task-conditioned and leave-task-out diagnostics; no EVAL endpoint attached to candidate or stage label. Record per-task positive gain support and undefined retention, pooled and worst-task retention. Fit thresholds on training roots, calibrate on separate roots where feasible. Do not interpret pooled retention as reliable task transfer.
- Add high-level feedback regime routing actions: UNIFORM_REINVEST, STATIC_K4, ROTATING_K4, BALANCED_K4, ACQUIRE, ABSTAIN, VERIFY, ROLLBACK. Train on prospective measured trajectory utility under equal CPU; use TRAIN diagnostic gain for local shaping, keep final sealed evaluation independent.
- Distinguish action imitation from verified preferences and successful repair. Build dataset admission tests with synthetic future-score canaries, null outcome canaries, sham rows, wrong split, duplicate IDs, root overlap, and post-update score quarantine. Stage actions should be reproducible without reading future features.

Next: use GYM-02 feedback validity campaign to create new stage-local evidence. V3 stays archived and can support offline engineering audit; do not blend the 13,329 imitation rows wholesale into the pretraining or Inky adapter curriculum.
