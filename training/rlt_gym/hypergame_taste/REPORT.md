# Hypergame research taste campaign

The CPU run replayed **195,240 synthetic episodes** and **14 actual Qwen episodes**. The fitted recurrent controller repaired 968 of 971 verified native-pool misses across the two tests. It tied matched damping and used 111 more checks than heuristic repair. The paired expected-utility intervals include zero on both splits.

## Collection and custody

| Archive | Roots | Replayed episodes | Training presentations |
| --- | ---: | ---: | ---: |
| Original binary-signal proxy, public seed | 6,400 | 38,400 | 5,120 |
| Strategic worlds, private seed | 25,600 | 153,600 | 12,452 |
| Strategic K=12 and K=16 stress | 540 | 3,240 | 261 |
| Qwen3-4B-Instruct-2507 Q8_0, separate train-only roots | 14 | 14 | 32 |

The strategic verifier regenerated every world, tape, decision, training row and scorecard. The separate rational solver checked 2,400 sampled roots, including 50 stress roots. The main strategic run took 68.7 seconds to collect and 76.0 seconds to replay. The private key and underlying traces stay in the local run archive. The [campaign receipt](campaign_receipt.json) includes source fingerprints, dataset hashes, archive counts and run failures. The [cohort metrics](cohort_metrics.json) include 444 rows stratified by K, horizon, dimension, research family and OOD condition.

## Research decisions on strategic worlds

These results are from simulator policies. The exact reference maximizes expected true utility at each fork after accounting for remaining time and research cost. Each test split uses 5,120 matched worlds.

| Policy | Ordinary mean utility | OOD mean utility | Ordinary cost | OOD cost |
| --- | ---: | ---: | ---: | ---: |
| Exact | 2.80 | 3.02 | 1.65 | 1.65 |
| Two-ply | 2.75 | 3.00 | 1.52 | 1.54 |
| Ignore nested views | 2.04 | 2.29 | 1.79 | 1.77 |
| Greedy terminal | 0.84 | 0.95 | 0.00 | 0.00 |
| Native pool only | 0.53 | 0.72 | 0.61 | 0.60 |
| Random | -5.70 | -5.66 | 0.75 | 0.75 |

The exact policy identified the causal mode by its final decision on about 77% of both tests. Initial causal Brier scores were 0.57 ordinary and 0.56 OOD. The policy that dropped actor views scored 0.62 on both. The marginal model of the outer actor's endorsement had ordinary Brier 0.206. The OOD condition scores ranged from 0.186 to 0.208. Modes 1 and 2 had identical endorsement likelihoods on all 1,671 alias worlds. Every one admitted a probe whose observation law distinguished the two modes. In those worlds, the best initial probe had mean expected value 1.90 above the best immediate terminal decision.

HG-01's observer-owned games, HG-02's costly audit and safe abstention, HG-03's dimensional controls, and HG-04's observation fibres informed the simulator. The relevant mathematical sources are the finite subjective transition model, the Bayesian sufficient state and the aliasing argument. The implementation uses original frozen worlds. Native GPTStoryworld replay remains pending. [Pinned GPTStoryworld source](https://github.com/MoralityLabAI/GPTStoryworld/tree/5cd0b480e9a01d07418cdf8cb3bedbc9d38ef906/verifiers_envs/storyworld-hypergames-env)

## Fitted correction controller

The NumPy head saw 5,376 capped decision presentations on 1,344 fitting roots. The evaluator used distinct calibration, ordinary and OOD roots, with 756 per split. Routing by family and a fixed evidence abstraction stand in for the future TRM and LDT components. The trained layer revises the candidate ordering after verifier rejections. Controls share a ceiling of four visits and three checks per fork. An action passes when expected utility is nonnegative and at most 0.5 below the exact best. A native miss requires a positive best action and a failed pass criterion for every native action. These are first-fork checks and their value includes optimal remaining play.

| Arm | Ordinary solves | Ordinary miss repairs | Ordinary checks | OOD solves | OOD miss repairs | OOD checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed depth | 708 | 470 | 756 | 709 | 478 | 756 |
| Greedy | 121 | 0 | 756 | 126 | 0 | 756 |
| Heuristic repair | 750 | 475 | 1,267 | 754 | 488 | 1,280 |
| Learned order, single check | 710 | 472 | 756 | 698 | 482 | 756 |
| Check every visit | 753 | 478 | 1,314 | 755 | 489 | 1,355 |
| Learned deferral | 754 | 479 | 1,229 | 754 | 489 | 1,251 |
| First valid in native pool | 274 | 0 | 1,744 | 266 | 0 | 1,767 |
| Matched damping | 754 | 479 | 1,314 | 755 | 489 | 1,344 |
| Random | 392 | 208 | 1,864 | 398 | 219 | 1,899 |
| RLT | 754 | 479 | 1,314 | 755 | 489 | 1,344 |

RLT's paired gain over heuristic repair was +0.0118 expected utility on ordinary roots (95% interval −0.0009 to +0.0244) and +0.0045 on OOD (−0.0024 to +0.0114). The fitted recurrence and matched damping produced the same aggregate results. On ordinary roots, learned deferral matched the RLT solve count with 85 fewer checks and 55 more visits. The fit took 9.4 seconds and scoring 5.7 seconds. Policy metrics for each stratum and the weight and trace hashes are in the receipt.

The earlier [224-root Aleph comparison](../hard_pool_proxy.md) is preserved. Fixed recurrence, heuristic repair, learned ordering and deferral solved 15, 154, 168 and 138 roots with 224, 503, 459 and 227 checks. The heuristic repaired 137 of 168 confirmed pool misses. The two experiments use different verifiers and miss criteria.

## Actual local model

Qwen produced 32 legal actions on 14 fresh training roots. The model tested or published an action outside the native pool 18 times. It published correctly on nine roots and incorrectly on five. Total realized utility was −22. The exact policy obtained +77 on the same tapes with 31 visits. Qwen made 32 visits and checks, used 9,622 prompt tokens and 277 completion tokens, and took 640 seconds on six CPU threads. The pilot gave Qwen the public posterior and action menu. Raw-vector belief recovery remains a separate evaluation.

The model export preserves 20 verifier-accepted demonstrations and 12 explicitly synthetic repairs of rejected model choices. The first engine attempt appended a console footer to valid JSON. That partial archive was retained and the corrected execution preserves the engine's raw response and parses its JSON separately. Codex and Opus produced no collected rollouts in this run. The 1B DataGoblin model remains untrained.

## Benchmark exchange and training mixture

The rTaste adapter processed all 415 public experiment registrations as 83 evaluation-only cohorts, preserving missing audit and rerun values. The Taste-Bench research adapter passed paired-order and label isolation tests against a schema fixture. Its gated data would require the upstream release. PaperBench and FrontierScience remain separate evaluation tracks. The crosswalk describes which contracts from TASTE, TasteVal, IdeaTaste, Co-Scientist, AlphaEvolve, Prime Intellect and Atropos/Hermes inform synthetic cases. [Benchmark crosswalk](README.md)

The main strategic export includes 2,035 SFT, 2,064 transition, 2,103 repair, 2,080 preference, 1,487 outcome prediction and 2,683 research selection rows. To start the curriculum, interleave the six views with a slight emphasis on repair and research selection. Mix shallow and deep cases by root, then sample stress worlds separately. Treat the small Qwen set as a calibration slice and preserve its evidence-class labels. Count reformatting as the same presentation. Keep the aggregate cap at four per root.

## Replay and continuation

Run from the repository root after installing the development extras. Set `RESEARCH_KEY` to the retained private seed path when replaying the recorded private archives. A different key starts a fresh campaign. Output folders must be fresh.

```sh
python -m pip install -e ".[dev]"
ruff check .
python -m pytest
python -m datagoblin.training_gym.rlt_gym.hypergame_taste --seed 20261121 --per-cell 256 --depths 0 1 2 4 8 --out runs/hypergame-taste-38400
python -m datagoblin.training_gym.rlt_gym.hypergame_taste --replay runs/hypergame-taste-38400
python -m datagoblin.training_gym.rlt_gym.research_campaign --per-cell 1024 --private-seed-file "$RESEARCH_KEY" --out runs/strategic-153600
python -m datagoblin.training_gym.rlt_gym.research_campaign --replay runs/strategic-153600 --private-seed-file "$RESEARCH_KEY"
python -m datagoblin.training_gym.rlt_gym.research_campaign --per-cell 54 --depths 12 16 --private-seed-file "$RESEARCH_KEY" --out runs/strategic-stress
python -m datagoblin.training_gym.rlt_gym.research_campaign --replay runs/strategic-stress --private-seed-file "$RESEARCH_KEY"
python -m datagoblin.training_gym.rlt_gym.research_diagnostics --archive runs/strategic-153600 --out runs/strategic-153600/diagnostics.json
OPENBLAS_NUM_THREADS=1 python -m datagoblin.training_gym.rlt_gym.research_controller --private-seed-file "$RESEARCH_KEY" --train-per-depth 192 --eval-per-depth 108 --out runs/strategic-controller
python -m datagoblin.training_gym.rlt_gym.research_benchmarks format-training --archive runs/strategic-153600 --out runs/strategic-training
python -m datagoblin.training_gym.rlt_gym.research_teacher --out runs/qwen-cpu-compact --private-seed-file "$RESEARCH_KEY" --per-depth 2 --timeout 100 --binary "$LLAMA_COMPLETION" --weights "$QWEN_GGUF" --model Qwen3-4B-Instruct-2507-Q8_0
python -m datagoblin.training_gym.rlt_gym.research_teacher --replay runs/qwen-cpu-compact --private-seed-file "$RESEARCH_KEY"
python -m datagoblin.training_gym.rlt_gym.research_model_exports --archive runs/qwen-cpu-compact --private-seed-file "$RESEARCH_KEY" --out runs/qwen-training
```

Local Ruff and pytest passed, with 286 tests passing and 21 skipped. The original RLT CI path replayed 196 full trajectories and independently rechecked 365 candidates. The two historic hosted workflow runs reported three failed jobs with no step data and returned 404 on log retrieval. Their execution cause remains undetermined. [General CI](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44878370657) · [RLT CI](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44878370579)
