# HRM-Text learning and interpretability milestone

This milestone tests whether streaming summaries at the released HRM-Text recurrent boundaries provide useful,
calibrated evidence about errors in DataGoblin constitutional verdicts. It does not train or align a 1B model. The
executable `ConstitutionalValidator`, not the proposal model, supplies authoritative decisions and violation masks.

## Frozen experiment

The preregistration is `configs/preregistrations/cycle_uncertainty_v1.yaml`. Its primary hypothesis is that a larger
relative H1-to-H2 update predicts a higher probability of an incorrect verdict after controlling for final-output
entropy, prompt length, output length, and schema complexity. Ordinary test is the confirmatory split. OOD and
sealed-poison are report-only. The sealed file contains no labels, receipts, expected decisions, or expected violations;
labels are recomputed independently for analysis and are never included in the model prompt.

The recurrent feature set is relative H1-to-H2 update, mean final-L update, maximum late token update, exponential
update decay, H1/H2 Jensen-Shannon logit divergence, and residual growth. Baselines are verdict entropy, maximum verdict
probability, verdict-token margin, sequence log probability, prompt length, output length, and schema complexity.

Reports distinguish:

- ranking quality: error-prediction AUROC and Spearman association;
- probability calibration: Brier score and expected calibration error;
- selective prediction: risk-coverage data and area under that curve.

Calibration parameters are fit only on the calibration split. Logistic and isotonic calibration are supported. The
report includes bootstrap intervals and preserves null or negative results.

## Trace boundary

`TraceableRecurrentModel` is independent of Transformers and HRM-Text. Tests and the offline workflow use a deterministic
mock. The optional adapter observes the released source's two H cycles and three L cycles per H cycle in this boundary
order: `L1, L2, L3, H1, L4, L5, L6, H2`. This naming describes call order only; interpretations of H and L dynamics are
hypotheses. A regression gate requires ordinary and instrumented output text to be identical and final logits to agree
within the configured tolerance.

Only streaming aggregates are serialized: step and pooled update norms, selected-token update norms, H1/H2 logit
divergence, residual norms, exposed gate summaries, verdict probabilities/output, labels, and hashes. Sampled activations
require explicit opt-in and a byte budget. Full activations are never persisted by the default path.

## Real-checkpoint operation

No command downloads weights implicitly. `hrm-status` uses package metadata only and does not import frameworks or load a
checkpoint. To run a local checkpoint, install `.[hrm-text]`, launch DataGoblin inside an operator-controlled hard-cap
wrapper, and supply `--local-model-path ... --confirm-hard-cap-wrapper`. A Hugging Face ID is accepted only when supplied
explicitly; remote retrieval additionally requires `--allow-download`, and checkpoint-provided Python requires
`--trust-remote-code`. Review that source before opting in. The adapter uses `shell=False` nowhere because it does not
launch a shell or subprocess. DataGoblin does not claim that Python hooks constitute a security sandbox.

## Role conditioning

The role ablation compares direct, `synth,direct`, `synth,cot`, textual SCOUT, textual ALCHEMIST, and textual RAT prompts.
It adds no tokenizer tokens. Prompt effects do not establish distinct internal roles. A later special-token study is
specified in `docs/hrm_text_special_token_sft_design.md`.

## Claim boundary

- Convergence does not imply correctness.
- The experiment does not establish alignment.
- Recurrent signals may merely track prompt complexity.
- HRM-specific value requires comparison with recurrent baselines.
- A null result is valid.
