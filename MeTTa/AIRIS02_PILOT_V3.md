# AIRIS-02 v3: separate learned STOP gate, measured negative result

**Source:** [GitHub Actions run 44992979510](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44992979510) on [DataGoblin PR #33](https://github.com/MoralityLabAI/DataGoblin/pull/33). Dataset manifest SHA-256: ff7949ee49d26e47e0ac12cefc6a14b96eca84d07f0271d12b669ddbe56b235b. Shared public development worlds SHA-256: 2d1aa6ef3bf781526a45d2479a42c2c82a7e2f2e265fd839b22f8abc888e3b29.

The exact Python source-root replay passed before training. The sample contains 160 original synthetic training roots and 60 public development roots per arm, four intervention presentations per root, and 31 STOP-labelled training presentations per arm. None of the registered public-test or steward roots were opened. No third-party source text or hidden labels were consumed.

A separate 164-parameter logistic STOP head was trained from scratch on the same root-grouped public features, with one fixed train-only threshold per arm. Calibration limited **observed training false-stop rate** to 6/129 (4.65%) for each arm. This is an empirical bound on the training partition, without guaranteed performance on development or real sources. The original STOP-as-action policy is preserved as an ablation.

## Shared 60-task public development result

All policies below trained the same base 3,675-parameter weight-tied recurrent repair model for 48 gradient updates. Its weights were trained at four internal iterations; no trained adaptive recurrent-depth controller was used. The same source-root tasks and uniform-intervened public development worlds were scored for every arm.

| Intervention data arm | STOP-as-action solved | STOP-as-action premature | Learned-head solved | Learned-head premature | Learned-head unsafe additions | Ever reached oracle optimum (learned head) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 1 | 59 | 1 | 11 | 43 | 14 |
| Motif-guided | 3 | 57 | 0 | 6 | 52 | 16 |
| Budget-first | 1 | 59 | 0 | 13 | 40 | 14 |
| MOSES-gated | 3 | 57 | 1 | 11 | 43 | 14 |

The separate learned gate reduces premature stopping by allowing more edit attempts. It also exposes a serious safety problem: the unconstrained ADD head repeatedly introduces invalid synthetic selections. The upstream DataGoblin ResearchMembrane was never invoked to authorize those additions. These scored invalid edits are **research traces**, not real data releases.

The model sometimes reaches the exact optimum along a trajectory but fails to stop there. It also lacks reliable source-constraint reasoning and has not shown transfer, improvement from recurrent depth, or an advantage from motif-selected training data. Fixed-depth 2/4/8 results remain nearly identical on this development distribution, apart from compute cost.

## Registered next safety ablation

Add an **environment-owned, read-only verifier veto** that checks an ADD proposal against the public synthetic constraint verifier before modifying the working selection. A vetoed proposal must be counted, its source identity retained, and execution must either choose another legal proposal or abstain. The verifier may confirm synthetic admissibility and constrained monotone repairs, but it must never supply the hidden oracle-optimal subset, the model's target edit, or hidden public-test labels.

Compare the original unconstrained STOP-as-action and learned-STOP policies against verifier-gated versions on the same frozen public development worlds. Report attempted invalid additions, vetoes, accepted invalid additions, reached-optimum events, completed tasks with STOP, abstentions, added verifier calls and inference budget. Cost accounting is essential because verifier-assisted behavior performs extra computation.

Keep all policies advisory until an independent steward evaluation on fresh source lineages and true native MeTTa/ResearchMembrane integration. The current model is an intentionally small CPU research control and should not be promoted to a general curation agent.
