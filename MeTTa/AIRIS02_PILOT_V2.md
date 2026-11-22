# AIRIS-02 v2 corrective pilot: legal actions, fixed-point STOP supervision

**Measured source:** [CPU run 44992526441](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44992526441) on PR #33. Exact synthetic oracle and full source-root replay passed. Dataset version metta-airis02-bundle-v2, 160 train roots and 60 public development roots with four arm presentations per root. Reserved public test and steward root labels opened: zero.

The initial [v1 result](AIRIS02_PILOT_V1.md) had inapplicable edits in every evaluation episode. V2 added a structural membership-only action mask in training and inference. A deterministic root-hash schedule replaced about one fifth of source-root presentations with the oracle's fixed-point selection, giving positive STOP training targets. All four intervention arms share the same source-root groups and policy parameters, and every trained model evaluates on the same uniform-intervened development worlds.

| Intervention training arm | Recurrent exact first edit /60 | Solved with explicit STOP /60 | Premature STOP /60 | Inapplicable edit |
| --- | ---: | ---: | ---: | ---: |
| Uniform | 26 | 1 | 59 | 0 |
| Motif guided | 29 | 3 | 57 | 0 |
| Budget first | 33 | 1 | 59 | 0 |
| MOSES gate | 29 | 3 | 57 | 0 |

The recurrent checkpoint is trained at depth 4; its inference-depth 2/4/8 ablation produced the same success counts in this pilot. The direct linear models completed 1–2 tasks in each arm. The model weights contain roughly 3.4k linear parameters and 3.7k recurrent parameters, with 48 gradient steps per arm.

**Critical negative finding:** STOP is now over-predicted once the model has made one or more repairs. From 57 to 59 of 60 episodes end with an incorrect STOP. This dataset change repaired basic action applicability without producing a successful learned stopping policy, recurrence benefit, or a demonstrated motif-training advantage.

## Registered next ablation

Train an additional explicit binary STOP head on the same verified synthetic training examples, with its threshold fixed using training labels alone. The action head must remain unable to use the exact oracle or development labels at inference. Compare the existing direct STOP class with a separate conservative learned stop gate and report premature stops, *ever reaching the exact optimum*, and completed tasks. Keep the old v2 checkpoint metrics immutable. An ACT-style learned iteration budget and independent steward evaluation remain separate research milestones.

No model or data-source promotion is authorized by this result.
