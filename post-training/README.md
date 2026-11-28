# DataGoblin research post-training

**Status:** PT-00 design scaffold, 2026-11-28. No live jobs, weights, new data admission or paid spend authorized by this directory.

Goal: make the ~1B direct-A Aleph checkpoint a research policy that can choose informative experiments, revise beliefs, budget recurrent computation and produce reproducible findings. The outer auto-research agent (e.g. Fable on Silico) designs and audits experiments; the inner learner (DataGoblin) learns from the recorded outcomes and eventually generates its own research trajectories.

## Entry points

- [35-RU campaign protocol](campaigns/PT-01.md)
- [Machine-readable campaign policy](campaigns/35ru.json)
- [Research transition contract](schemas/research_transition.v1.json)
- [Silico agent handoff](SILICO_AGENT.md)

Existing integration surfaces: [training archive](../training/README.md), [10B continuation handoff](../training/reports/2026-11-28_aleph_10b_colab_handoff.md), [data admission](../training/data/README.md), [RLT gyms](../training/rlt_gym/README.md), [training gym](../docs/training_gym.md), [hill climb](../gdk/README.md). Connect EvalMath and Compactification Distill through versioned adapters with frozen source revisions, independent evaluator receipts and explicit input/output contracts. Neither adapter is implemented in PT-00.

## Two loops

Outer loop: freeze baseline + research question -> authorize one capped campaign -> Fable proposes competing hypotheses -> run bounded experiments -> archive results, failures and decisions -> audit -> propose next campaign.

Inner loop: pinned Aleph checkpoint -> research action or tool call -> environment observation -> independent reward -> candidate checkpoint -> held-out development evaluation -> conditional promotion. Branch from the same parent for matched controls. Keep the pre-training checkpoint lineage unchanged. A weak early checkpoint may only support small constrained action vocabularies; gate free-form on-policy work on verified capability.

## Data views of the same trajectory

| View | Objective | Admission |
|---|---|---|
| SFT | Masked cross-entropy on teacher action or public explanation | Rights, source and outcome provenance; observation spans are context |
| Weighted imitation | Weighted CE using checked return or estimated advantage | Frozen weights, clipped extremes, independent outcomes |
| Preferences | DPO or ranking on alternatives from a shared observable state | Both candidates evaluated under same protocol; ties kept separately |
| Offline value learning | Return/value prediction; conservative supported-action evaluation | Complete transitions, terminal/truncation flags, coverage audit |
| On-policy RLVR | Student rollouts, independent rewards, policy-gradient updates | Fresh student behavior, log-probability and version records, matched controls |

Fable traces are off-policy relative to Aleph. Never substitute teacher log-probabilities for Aleph log-probabilities or pass Fable completions to an on-policy GRPO/PPO trainer as fresh student samples. A recorded alternative is a proposal until executed or independently verified. A negative finding can be valuable when the experiment falsifies a registered prediction. Unknown outcomes stay unknown. Do not require private reasoning traces; request public derivations, plans, testable predictions and evidence-backed decision summaries.

## Research state and reward

Record the *observable* pre-action state, hypothesis, alternatives, chosen action, forecast, evidence refs, environment outcome, next state, termination type, costs and provenance. Outcomes are produced by the environment/verifier; student and teacher statements are attributed proposals. Preserve timing of every event and never expose post-action evidence to the action policy.

Freeze reward version before collection. Keep a vector of task progress, independently checked claim/negative finding, information gained, calibration, compute cost, verifier cost, integrity failures. A scalar utility is a versioned projection, not immutable ground truth. Information gain is null unless a belief model or registered proxy is defined. Reward incremental evidence once, avoid paying twice for restated findings. Hard integrity violations quarantine data and block promotion; do not turn them into an optimizable bonus.

Optional compactification: preserve raw state IDs and attach a separate quotient ID only after testing action availability, reward and transition congruence for the relevant policy/budget. Never merge states just because their text is similar. Group all descendants, alternatives, near duplicates and quotient-related roots within the same split. Keep sealed test results inaccessible to the outer agent.

## Checkpoint lineage

Start from an explicitly verified direct-A checkpoint (10M engineering parent or a later admitted pre-training snapshot). Save immutable parent, architecture and tokenizer fingerprints, objective, optimizer/scheduler/RNG state or a declared reset, dataset manifest, code hash, seed, hardware, measured cost, and eval receipts. Branch SFT / weighted / preference / on-policy arms independently; do not overwrite the pre-training branch. Track cumulative processed positions separately from loss-bearing tokens, recurrent visits, teacher tokens and provider RUs.

PT-00 is deliberately provider-neutral. Raw teacher text, secrets, account data, restricted sources and unreviewed provider outputs remain outside this public repo. Verify licenses and provider terms (including training/distillation rights) before teacher calls or release; only authorized sanitized data enters training. No automatic grant renewal or calendar-triggered jobs.
