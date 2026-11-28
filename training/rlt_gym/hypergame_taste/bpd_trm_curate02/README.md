# BPD-TRM-CURATE-02: native curation export ingress

Source: MoralityLabAI/BPD pinned commit af43e48b1ed472eef88bedd0cfa8f4f8aefb49b5, PR #170. Source manifest includes exact Git blob and SHA-256 identifiers. The two original JSONL files are referenced at their source, rather than copied into DataGoblin. No DataGoblin neural training or native gym campaign has run in this step.

## Verified custody

Both full blobs retrieved through the GitHub connector, SHA-256 checked against source manifest. Reconstructed all directed semantic edges and exact Python-repr SHA-256 stratified split (624 fit, 144 feedback, 144 tune; 312 EVAL edges excluded). Every candidate belongs to fit; each root/window has 16 candidates, at most four selected, one checkpoint and consistent batch acceptance fields. All 248 preference pairs resolve within same root/window/checkpoint and match original candidate fields. Record IDs and selected unordered-edge cap <=4 per root checked. This is an independent byte/structure/label consistency audit of the export, not checkpoint re-evaluation or source raw-row replay. Source manifest root-level hashes checked against the exported provenance, not recomputed from the eight original source logs.

The manifest's 809 successful selected decisions mean positive *individual counterfactual probe gain*, of which 679 occur in accepted four-edge windows and 130 in rejected windows. 203 of 640 windows accepted. Individual gains are not additive under AdamW; no row may be relabelled as an actual committed parameter update. The 248 preferences are within-window hypothetical candidate-gain comparisons, not preferences over held-out performance. Keep ranking and actual window outcome objectives separate, group validation by model root, keep all candidate labels out of final EVAL cohorts.

## Post-hoc development ranking

All 8 folds train a fixed 20-feature ridge model on 7 roots and rank candidates in the remaining root's 80 windows. Train-only feature normalization and L2=1; no realized or window-acceptance labels as inputs. This is a retrospective diagnostic, not preregistered evidence.

| Method | Within-window pair accuracy | Mean top-4 individual realized gain |
|---|---:|---:|
| Loss | 0.361510 | -0.0845705 |
| Predicted gain | 0.881263 | -0.0449961 |
| Predicted gain / observed probe CPU | 0.775677 | -0.0476449 |
| Ridge | 0.829492 | -0.0450774 |
| Oracle realized gain | 1 | -0.0412296 |
| Recorded selected | n/a | -0.0561010 |

Ridge reduces held-out-root individual gain MSE 0.0031210 -> 0.0018385, yet mean top-four gain difference vs raw score -0.00008124 (descriptive interval [-0.00026647,0.00010400], 4/8 roots positive). Ranking objective has no demonstrated improvement. Ranking uses all 16 candidates irrespective of exploration/cap and ignores batch interactions; oracle top-four negative average is a diagnosis of pool-level one-step headroom, not a prediction of final task performance. Recorded selection includes exploration and fallback; observed probe CPU is retrospective and may be noisy. Do not use measured CPU or post-step gain as uncharged pre-probe features. Native BPD report's warmup-fork calibration diagnostics differ from these whole-campaign ranking diagnostics.

## Replay

From a checkout of BPD at the pinned commit, set EXPORT to mechinterp/bpd/experiments/trm_curation/results/datagoblin_export and run these scripts from this directory:

    node audit_export.mjs "$EXPORT"
    node rank_audit.mjs "$EXPORT/candidate_decisions.jsonl"

Both function bodies ran directly in V8 on the full source blobs. CLI wrappers and DataGoblin repository CI have not been run in this turn. Native training is separate. A proposed optimizer-aware screening helper and actual synthetic PyTorch parity smoke are in BPD PR #176 (4dddca6844d9c1d250cff245e1d60aa46ec57de3); no fixed-CPU training benefit has been measured for the new helper.

Next: candidate screening based on actual predicted AdamW displacement, a pure no-op/abstention candidate, and batch-aware selection under a matched CPU budget. Collect fresh candidate counterfactuals only in TRAIN feedback, and actual batch commits separately. Use E29 bounded Dyck-1/E31 pointer-chase exploratory headroom, E27/E28 failure and shift controls, and E32–35 saturation sanity controls. Preserve failed causal admission and EvalMath decision. Neither the ingress audit nor retrospective ridge fit changes the MTR-02I disposition.
