# BPD-TRM-CURATE-03B: paid-acquisition controller and Inky Mini trajectory format

Evidence class: development only. Source: MoralityLabAI/BPD PR #176, branch `research/bpd-trm-adamw-screen-20261127`; the export commit and file hashes are pinned in `source_manifest.json`. The export files are referenced at their BPD source and are not copied here. The CURATE-02 ingress (`../bpd_trm_curate02/`) and its retrospective acquisition audit stay spent-development evidence: full-screen abstention avoided 275 rejected attempts and missed 1 accepted update, early-12 retained 99.82% with 9,040 instead of 10,240 acquisitions, and root-held-out thresholds retained 99.13% (below 99.5%).

## What the V2 export contains

`trajectories_v2.jsonl` holds one row per (root, arm, window), TRAIN-only. Each row has:

* **Provenance:** parent checkpoint SHA-256, optimizer-state digest, shared-warmup SHA-256, BPD source hashes.
* **State and actions:** feedback loss and stage features before each acquisition; the candidate pool in prospective acquisition order; actions from `ACQUIRE_4/8/12/16`, `ABSTAIN`, `SELECT_BATCH`, `SWAP_EDGE`, `VERIFY_BATCH`, `REINVEST_CPU`.
* **Individual labels:** predicted gain per candidate. Realized individual gain appears only where a transactional probe measured it (arm B). Everywhere else it is null with `missing_unaudited` or `not_acquired`.
* **Joint labels:** predicted aggregated and native joint gains, plus the committed, rolled-back or audited actual feedback gain, with the audit probability and draw. Unaudited abstentions are `missing_unaudited`.
* **Costs:** measured window CPU and process-tree RSS samples.

`acquisition_stages_v2.jsonl` holds the stage-level decision points with features available before the next acquisition. `horizon_outcomes_v2.jsonl` holds end-of-budget EVAL accuracies per root/arm. These are kept apart from decision rows on purpose and are never used as decision labels. `unit_costs.json` holds the measured per-candidate, per-attempt and per-window base CPU of the analytical mechanistic arm.

## Controller

`acquisition_policy.py` models acquisition as a resource-bounded sequential decision,

    Q(s, a) = E[dU + gamma V(s') | s, a] - lambda C(s, a),

with stage decisions after 4, 8 and 12 candidates and an attempt/abstain decision after 16. It evaluates, root-held-out over the eight fresh development roots and replayed per window on the recorded `D_analytic_mech` parents:

* constant `ACQUIRE_16` and constant `ABSTAIN`;
* fixed zero gates (full-16, early-4/8/12);
* the root-held-out adaptive threshold grid (343 triples, as in CURATE-02);
* a learned logistic stage model;
* the same model fit on shuffled labels.

All fitted policies use the CURATE-02 constraint: pooled retention of 99.5% or more and every training root at 98% or more, then minimum estimated CPU. Controller inference CPU is measured and charged. The replay cannot see what skipping a window would have done to later checkpoints.

`--traces DIR` writes `inky_sft.jsonl` (state prompt → executed action sequence), `inky_preferences.jsonl` (attempt vs abstain, verified by the matched-RNG TRAIN feedback acceptance test, including rejected decisions) and `inky_repair_traces.jsonl` (rolled-back minibatch → next decision). These are formats for later Inky Mini SFT and preference training. No Inky Mini training ran.

## Results (root-held-out over 28301–28308, 1,270 D_analytic_mech windows)

Unit costs measured on the D arm: 0.0287 CPU-s per candidate, 0.0831 per attempt, 0.0899 per-window base.

| Policy | Candidates acquired | Committed windows missed | Retained positive gain | Est. CPU vs always-16 |
|---|---:|---:|---:|---:|
| constant ACQUIRE_16 | 20,320 | 0 | 100% | 1.000 |
| full-16 zero gate | 20,320 | 0 | 100% | 0.933 |
| early-12 zero gate | 17,536 | 6 | 99.14% | 0.828 |
| early-8 zero gate | 14,312 | 14 | 98.89% | 0.708 |
| early-4 zero gate | 10,228 | 28 | 95.96% | 0.552 |
| adaptive threshold grid | 18,192 | 2 | 99.26% | 0.856 |
| **learned logistic stage model** | **15,328** | **2** | **99.58%** | **0.751** |
| learned, shuffled labels | 20,320 | 0 | 100% | 0.934 |
| constant ABSTAIN | 5,080 | 219 | 0% | 0.324 |

The learned stage model is the only fitted policy above the 99.5% pooled retention target, and it uses less estimated CPU than the early-12 gate. The shuffled-label control fits no useful thresholds and degenerates to the full gate. Held-out root 28301 retains only 97.43% under the learned model (early-12 and the adaptive grid give 94.34% there; see the folds in `acquisition_policy_receipt.json`), so the per-root 98% bar fails on that root. Controller inference costs 0.031 CPU-s across all held-out decisions; Windows `process_time` resolves 15.6 ms, so this is close to the timer floor.

Scope limits:

* This is a per-window replay on recorded parents. Abstaining would have changed later checkpoints, and the replay cannot see that.
* In the live eight-root pilot, every gated arm still trailed uniform CPU reinvestment by 3.3–4.7 pp on delay-31 accuracy (BPD `results/curate03b/development/summary.json`).
* A cheaper gate does not by itself make curation competitive at this budget.

Inky traces (`inky_v2/`, deterministic gzip -n; `inky_manifest.json` holds the uncompressed SHA-256s): 11,470 SFT rows, 1,270 verifier-grounded preferences, 5,471 rollback repair traces.

## Replay

From a BPD checkout at the pinned commit:

    python acquisition_policy.py <BPD>/mechinterp/bpd/experiments/trm_curation/results/curate03b/datagoblin_export_v2 --traces inky_v2

Python 3.12 + NumPy. DataGoblin's earlier audits are Node scripts; this one is Python because the run host had no Node runtime.
