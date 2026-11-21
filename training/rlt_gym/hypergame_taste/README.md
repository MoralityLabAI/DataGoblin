# K-depth Hypergames and Research Taste: RLT Gym

**2026-11-21 status:** A CPU-only exact-reference collector is implemented on this branch. Its traces are synthetic controls, with no trained RLT or frontier teacher outputs claimed. The local hard-pool commit 4b88fa6 was absent from remote GitHub when this branch began.

## What to train

DataGoblin should learn a **research-reorientation skill** for a mesh of looped reasoners. A TRM can route the problem and find relevant context, an LDT can filter alternative proposals, and an RLT can revisit decisions after experiments or verifier feedback. The environment-owned checker remains the sole authority on pass/fail and realized utility.

Separate the axes that often get collapsed into generic "harder reasoning":

- **K: epistemic nesting depth.** Each successive actor holds a subjective interpretation of what the previous actor believes. This prototype makes each layer an explicit noisy and possibly inverted endorsement channel, with its own perceived game. It is a K-level nested-belief proxy, rather than a complete multiplayer game with strategic best responses.
- **T: experimental decision horizon.** The number of remaining decisions, independent from K.
- **d: state dimension.** Later adapters should borrow the 140/512/1024 dimensionality controls from HG-02 to HG-04.
- **C: research cost.** Pilot tests, replications, and audits consume finite resources.

A research agent estimates P(H1 | observations), then chooses publication of H0/H1, abstention, a low-cost pilot, or a higher-fidelity replication. Wrong publication costs 15, correct publication earns 10; experiments cost 1 or 2. The exact Bellman oracle computes optimal expected value with rational arithmetic, while the sealed simulator produces actual results. The policy input sees only observations, known experimental channels, and the declared utility.

## Build on the existing hypergame work

| Existing source | Reuse |
| --- | --- |
| [GPTStoryworld HG-01](https://github.com/MoralityLabAI/GPTStoryworld/tree/main/verifiers_envs/storyworld-hypergames-env/Hypergames) | Agent-subjective G_i and G_i^j, observer-owned p/p2 maps, exact goal-consistency and robust probe-cover certificates |
| [HG-02](https://github.com/MoralityLabAI/GPTStoryworld/tree/main/verifiers_envs/storyworld-hypergames-env/Hypergames/single_player) | Hidden-state audits, safe-release gates, exact Bellman policies, proxy-reward traps |
| [HG-03](https://github.com/MoralityLabAI/GPTStoryworld/blob/main/verifiers_envs/storyworld-hypergames-env/paper/hypergames_qwen_frozen_trial.md) | Pre-registered Qwen 140/512-dimensional evaluation, with private seeds kept separately |
| [HG-04](https://github.com/MoralityLabAI/GPTStoryworld/blob/main/verifiers_envs/storyworld-hypergames-env/HG04_RESULTS.md) | Observer-quotient sufficiency, source-hashed CPU checks, misleading objectives and information-loss controls |
| [DataGoblin Aleph](../2026-11-21_aleph_proxy.md) | Verified repair, fitted candidate ordering and the check-every-visit comparator |

HG-01 measures first- and second-order perception through p/p2. K>2 requires an explicitly defined generator of higher-order views and independent objective/subjective transition tables before drawing conclusions about real agents. This version supplies the former only for synthetic reports.

## External research-taste sources and interfaces

| Institution | Research | Proposed adapter |
| --- | --- | --- |
| [Anthropic: TASTE, Aug 2026](https://alignment.anthropic.com/2026/taste/) | 92 strong-confidence pairwise evaluations of safety proposals, estimated 77% human agreement | Human-judgment external assessment; official benchmark access is by request, so keep any released items out of training |
| [OpenAI: PaperBench](https://openai.com/index/paperbench/) | Replicate 20 AI papers against 8,316 rubric items | Multi-stage plan/implementation/checking rubric, with rights and data splits reviewed before any import |
| [OpenAI: FrontierScience](https://openai.com/index/frontierscience/) | Expert scientific reasoning problems | Separate knowledge/reasoning transfer score |
| [Google DeepMind: Co-Scientist](https://deepmind.google/blog/co-scientist-a-multi-agent-ai-partner-to-accelerate-research/) | Generate, debate, rank and evolve research hypotheses | Independent hypotheses and idea tournaments, scored by resulting experiments rather than ranking self-consistency |
| [Google DeepMind: AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | Code evolution with automated evaluators | Exact algorithm and regression harnesses to test candidate research patches |
| [Prime Intellect: autonomous AI research, Aug 2026](https://www.primeintellect.ai/blog/measuring-autonomous-research) | 153 nanoGPT speedrun trajectories across 18 models and multi-day experiment search | Cost-adjusted decision-fork mining and independent multi-seed verification |
| [Prime Intellect: verifiers v1](https://www.primeintellect.ai/blog/verifiers-v1) | Taskset, harness and runtime split | Keep task/validator stable while swapping RLT, Codex, local or other agent harnesses |
| [Nous: Atropos](https://github.com/NousResearch/atropos) | Async multi-turn RL, game and executable environments; the repository is archived | Optional trajectory export matching its envelope, without depending on archived infrastructure |
| [Nous: Hermes 4 technical report](https://nousresearch.com/wp-content/uploads/2025/08/Hermes_4_Technical_Report.pdf) | Transparent sample-level logs, explicit parser failures, minimal evaluation clients | Keep unsuccessful parses and harness errors distinct from task failures |
| [TasteVal, Oct 2026](https://arxiv.org/abs/2610.06824) | Compute-normalized experimental research taste, with a fixed coder and withheld tasks | Separate research selection from coding and avoid attempting to reproduce withheld test tasks |
| [Taste-Bench, Sep 2026](https://arxiv.org/abs/2609.25804) | Mine trajectory decision forks and distill outcome-aware teacher judgments | Learn corrected actions from verifier-scored identical predecision contexts |
| [rTaste, Oct 2026](https://arxiv.org/abs/2610.40488) | Pre-register minimum viable experiments, forecast success, independently execute them | Cost/novelty/forecast calibration eval, subject to corpus provenance |
| [IdeaTaste, Nov 2026](https://arxiv.org/abs/2611.03424) | Freeze past literature and evaluate future research direction selection | Strict temporal literature split and proposal-quality audit |

These test distinct abilities. Pairwise research *judgment*, serial *experiment choice*, and expensive *research execution* each need separate tracks. Current public results motivate the metrics; they do not provide performance labels for our RLT.

## Reproduce the local CPU controls

Install the package in development mode:

    python -m pip install -e ".[dev]"
    python -m pytest tests/unit/test_rlt_gym_hypergame_taste.py -q
    ruff check src/datagoblin/training_gym/rlt_gym/hypergame_taste.py tests/unit/test_rlt_gym_hypergame_taste.py

Small preflight, with fresh output directory:

    python -m datagoblin.training_gym.rlt_gym.hypergame_taste --seed 23 --per-cell 4 --depths 0 1 2 4 8 --out runs/hypergame-taste-smoke
    python -m datagoblin.training_gym.rlt_gym.hypergame_taste --replay runs/hypergame-taste-smoke

Larger, streaming collection:

    python -m datagoblin.training_gym.rlt_gym.hypergame_taste --seed 20261121 --per-cell 256 --depths 0 1 2 4 8 --out runs/hypergame-taste-38400

Here per-cell is roots per (split, K), giving **6,400 roots and 38,400 matched policy episodes** with the last command. A 1,024-per-cell run gives 153,600 paired episodes, with storage needs growing with trace size. These are generator-backed controls, not LLM rollouts. The stream writer keeps memory bounded to a single root, a small scorecard, and the set of lineage IDs.

For proper private held-out generation, create a 256-bit random seed in a file outside Git, then add --private-seed-file /private/path/seed.bin. The manifest contains a seed hash commitment, and the original key stays offline. Publicly seeded runs remain engineering smoke tests.

## Files and promotion custody

- **public_tasks.jsonl:** predecision visible state and root/split IDs; no sealed truth or experiment stream.
- **private_sealed.jsonl:** hidden hypothesis and matched future experiment results, evaluator-only custody.
- **rollouts_scored.jsonl:** six matched reference policies, realized utility, visits and post-choice expected regret. Offline scoring archive, not model input.
- **train_teacher_requests.jsonl:** public training requests for optional external Codex/Opus/local teacher collection.
- **train_forks.jsonl:** exact expected-value comparisons between actions at the same public decision state. Each root has at most three distinct forks plus one teacher request, giving the four-presentation limit.
- **manifest.json:** archive digests, source hash, split/K/policy counts, per-cell costs, utility and outcome counts.

The --replay command hashes each file and reconstructs every scored policy trace. Each rollout writes independent post-choice oracle values for training or audit; those values are never fed into the online policy observation.

## RLT correction skill and promotion gates

The companion [research-hillclimb skill](skills/research_hillclimb/SKILL.md) accepts public state, a proposed research action and optionally a checker result. The skill must propose a falsifiable next experiment or a safe commit; after a failed check, it produces a revision with a new evidence citation and a costed prediction. All rejected proposals remain visible in the offline trace.

Build a compact ranker on the synthetic training forks and then collect independent teacher repairs. Compare against greedy, nested-belief-blind, always-probe, fixed-depth, learned-order check-each-visit and deferral policies on frozen roots. Track regret and realized utility with confidence intervals over root-level paired comparisons; record verifier calls, recurrent visits, research cost and family/K coverage.

The next **serious Storyworld** suite needs source-locked actor-visible p/p2, higher-order belief maps, subjectively perceived action sets and legal successor tables, an independently controlled objective transition oracle, changing goals, irreversible effects, and research hypotheses whose value can be falsified by experiment. If a counterfactual transition is unavailable, classify that case as insufficient_counterfactual_data rather than fabricate a result. Test novel agent/goal/payoff templates and private held-out seeds, not merely reworded train cases.

Evidence class in this first branch is synthetic_exact_oracle; architecture_promotion_eligible remains false until a trained RLT meets the preregistered held-out improvement gates.
