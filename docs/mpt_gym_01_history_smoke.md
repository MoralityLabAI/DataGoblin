# MPT-GYM-01 development smoke: checkpoint loss and history routing

2026-11-25. **PILOT_ONLY / REDESIGN.** Connects [EvalMath PR #69](https://github.com/MoralityLabAI/EvalMath/pull/69) and its MPT-GYM-01 protocol to a minimal CPU-trained predictor. No protected split, pretrained checkpoint, GPU, or live Lean invocation used. The actual Lean replay gate is outstanding.

## Design

A generated root has a hidden Bernoulli type shared across its obligations; the correct candidate index agrees with that type with probability 0.8. Each obligation has one hypothesis of type P and one of type Q; a lossy binary cue agrees with the correct candidate with probability 0.65. The trained logistic predictor sees the cue and a signed count of previous successful candidate indices. The reset intervention zeroes the signed count, retaining the same weights, cue, within-obligation retry and integer attempt budget floor(1.5 d). The toy checker independently tests the hypothesis type against P. It does not read the predictor's score. Both learned arms have the same candidate menu. A generator-informed greedy Bayesian posterior baseline uses the same lossy observations; a privileged full-type symbolic oracle sees the entire local typing environment. The latter demonstrates the narrowness of this artificial interface: a real Lean tactic policy could inspect the hypothesis types directly.

Fit 512 generator roots (128 per depth), 64 gradient steps with equal total weight per root. Measure teacher-forced designated-witness binary NLL on 64 separately generated development roots (16 at each depth 2,4,8,16). These are independently generated synthetic draws from one shared generator, not independent theorem families or a sealed cohort. Checkpoint schedule 0,1,4,16,64, with the last step used for the descriptive primary comparison. All attempts logged. Native replay source contains one theorem per development obligation with the incorrect candidate required to fail and the correct candidate required to close, followed by an explicit axiom report. It is not yet a native trace replay. No claim of comparable wall-clock/FLOP costs: the controlled budget counts abstract checks.

## Local CPU result

| Training steps | Witness NLL (nats) | Reset solves | History solves | History checks |
|---:|---:|---:|---:|---:|
| 0 | .693147 | 36 | 36 | 657 |
| 1 | .658833 | 61 | 60 | 606 |
| 4 | .634152 | 61 | 60 | 606 |
| 16 | .613741 | 61 | 62 | 605 |
| 64 | .591836 | 61 | 62 | 612 |

The final reset uses 642 checks; the greedy Bayesian baseline gives 60/64 and 606 checks, and the type oracle gives 64/64 and 480 checks. History/reset paired wins/losses/ties = 1/0/63; exact two-sided McNemar p=1, one discordant independent root. Absolute gain 1/64 is below the provisional five-point scientific floor even before calibrating a matched sham envelope. The checker-call reduction is descriptive and does not include feature, training or controller overhead. No matched-sham distribution or preflight power estimate yet; by the EvalMath preflight contract do not open confirmation. At steps 16 and 64 the same closure count masks a seven-check increase even as NLL falls. This is a useful engineering witness, not evidence of a general monotonicity or phase transition.

Three local tests passed. Source SHA-256 of the final tested runner: 63c822af975d4460df103db0a95e32623425f2291008a86bd31d7115e47b10e9. Generated development certificate SHA-256: 8fbb78fb5cae99642a78aeb613ab896d144672a21fe6ea84c6c8575bf19b41d4. Generated runs and weights remain outside the repository.

## Reproduce and gate

    PYTHONPATH=src python -m pytest -q tests/test_mpt_history_pilot.py
    PYTHONPATH=src python -m datagoblin.training_gym.mpt_history_pilot --out NEW_DEV_DIR
    PYTHONPATH=src python -m datagoblin.training_gym.mpt_history_pilot --out NEW_NATIVE_DIR --lean PATH_TO_PINNED_LEAN_BINARY

Supply the actual compiler binary, not an elan shim. The native command fails if the return code or count of explicit empty-axiom reports differs from the expected 480; no native claim until checked. Existing native infrastructure is documented in [RLT-LIVE-01](RLT_LIVE_LEAN_MCP_FIRST_DEV_20261122.md). The next substantive step uses pre-declaration Mathlib goal states with blinded theorem families and naturally occurring candidate failures, not artificial removal of hypothesis types. Log context, candidate-menu identity, chronological failures, dependency leakage sentinels, full inference costs and checkpoint witness NLL. Compare same-state/history-reset against a competent full-history symbolic policy and the full-information oracle. Before confirmation, run development-root power and matched-sham resolution preflight with a frozen scientific threshold; allocate new families only if ready. Retain the current development run as a negative resolution result.
