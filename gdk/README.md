# Data Goblin: a Goose GDK app for the DataGoblin hill climb

`data-goblin` is a Rust app built on the Goose Development Kit (GDK): `goose-agent`, `goose-providers`
and `goose-provider-types`, pinned at `0.1.0-alpha.14` like the goblin apps in HybridTRMLDT
(`MCP/goose_gdk_app`) and BitAgent Desktop. A local model drives a goose-agent state machine through
DataGoblin's hill climb, one gated move at a time:

| lane | what is climbed | metric (selection split) | tier of the main recipes |
|---|---|---|---|
| `self_adapter` | an adapter on the driver model itself, trained on verified hill-climb decisions | bench_accuracy (dev) | offline, loopback; GPU training |
| `looped_modules` | recurrent (RLT) students fitted on CPU | rlt_success_rate (calibration) | offline |
| `agonist_evals` | eval worlds a bigger local model builds and the AGN-01 verifier accepts | verified_pair_yield (dev) | offline, loopback |
| `agonist_rollouts` | RLT rollouts from a bigger local teacher, rechecked by the environment verifier | rollout_accuracy (train) | offline, loopback |
| `curation` | train roots admitted under split custody for looped models and adapters | admitted_roots (train) | offline |
| `research_loop` | other DataGoblin runs (research demo, recipe validation, gyms); nothing is promoted | none | offline |

The model chooses lanes, hypotheses and recipe parameters. The hill-climb MCP server owns everything
that decides: it runs allowlisted recipes from [`configs/hillclimb/lanes.json`](../configs/hillclimb/lanes.json),
reads metric values from run artifacts, applies each lane's frozen gate (baseline, promote, replicate,
reject, invalid; plateau and move budgets; a fixed eval contract per lane) and appends every proposal,
run, result and decision to a hash-chained ledger in `runs/hillclimb/ledger.jsonl`.

## Parts

- **State machine** ([`src/agent.rs`](src/agent.rs)): GDK `SlashCommandOperation` (skills as `/name`
  commands), `TextToolCalls`, `ToolApprovalOperation`, `ToolOperation`, `UnhandledToolCallsOperation`,
  `MaxTurnsOperation`, the Data Goblin prompt, and `InferenceRunner` over `OpenAiCompatibleProvider`.
- **Skill** ([`skills/datagoblin-hillclimb/SKILL.md`](skills/datagoblin-hillclimb/SKILL.md)): the move
  loop, gate rules, data flow between lanes, custody rules; one card per lane in `lanes/`. Skills are
  found with GDK `discover_skills`; `/datagoblin-hillclimb <lane>` loads the skill, and the
  `load_skill` tool loads it or a lane card (`datagoblin-hillclimb/lanes/curation.md`).
- **Tools** ([`src/datagoblin/hillclimb/server.py`](../src/datagoblin/hillclimb/server.py)): `suite_status`,
  `ledger_status`, `list_recipes`, `propose_move`, `record_result`, `decide`, `abandon_move`, and one
  tool per recipe the session's tier allows, with that recipe's exact parameter schema (`rlt_run(move_id,
  suite, depths, ...)`), over MCP stdio (protocol 2025-06-18). `run_recipe` (recipe name plus a params
  object) stays callable for scripts but is not listed: in a live run a small model sent an empty params
  object on every call, so recipes get declared schemas instead.
- **Approval**: `--approve read-only` hides the write tools; `ledger` (default) writes the ledger and
  runs offline recipes; `local` adds loopback recipes that call already running local model servers
  (Agonist worker, rollout capture, driver bench); `all` adds GPU recipes (adaptergym launches through
  vramorama leases). The app denies a call above its level before dispatch, and the server is started
  with the same tier cap.

The bigger local models are reached through recipes: `agonist_send`
([`agonist_worker.py`](../src/datagoblin/hillclimb/agonist_worker.py)) sends staged AGN-01 requests to a
loopback server with a `/tokenize` preflight against the context cap and keeps provider receipts;
`rollout_capture` uses the RLT gym's existing local capture. The self-adapter lane's decision bench
([`bench.py`](../src/datagoblin/hillclimb/bench.py)) asks exact questions about the gate, recipe-call
legality and lane eligibility, answered by the same functions the server uses; a task lands in train,
dev or test by its digest, so no task appears in two splits. `export_driver_sft` turns exact answers,
verified teacher answers and this app's clean transcripts into SFT rows for the driver's adapter.

## GPU queue and RAM

Gym runs go through the [vramorama Python client](https://github.com/patrickdugan/vramorama/tree/main/python)
(`pip install "vramorama @ git+https://github.com/patrickdugan/vramorama#subdirectory=python"`, or a
sibling `vramorama` checkout, or `DATAGOBLIN_VRAMORAMA_PY`). Each recipe in `lanes.json` declares
`resources`:

- `{"ram": "3G"}` (every CPU gym): the run waits for room in the RAM reservations shared by all
  vramorama clients on the machine, then runs under a hard cap (on Windows a Job Object memory limit
  over the whole process tree, kill-on-close) at below-normal priority. The card is not involved, so a
  power-capped GPU never holds a CPU gym back. A run that never got room ends `no_room`.
- `{"vram": "6G", "ram": "8G"}`: the run queues as a `vramorama run` lease whose commit gate leaves
  room for its RAM, and runs under the same cap inside the lease. Without the client it is refused.

The run record carries the mode, the cap, the peak, seconds waited and whether the limit was hit;
`suite_status.resources` shows the card, the lease queue, power cap, commit charge and reservations.
`adaptergym_launch` is not wrapped: it starts detached vramorama jobs of its own, which a kill-on-close
cap around the launcher would end. Without the client, CPU recipes run uncapped and say so.

## Build and run

```bash
cargo build --release            # in gdk/; on Windows: cargo +stable-x86_64-pc-windows-msvc build --release
./target/release/data-goblin skills --root ..
./target/release/data-goblin tools --root .. --python python --approve local
./target/release/data-goblin climb --root .. --lane looped_modules --moves 3 \
    --base-url http://127.0.0.1:18090/v1 --model inkling-2-mini --approve ledger
./target/release/data-goblin run --root .. --skill datagoblin-hillclimb --args agonist_evals \
    --base-url http://127.0.0.1:18104/v1 --model bonsai-2-4b \
    --teacher http://127.0.0.1:18200/v1 --approve local
```

`--python` must import DataGoblin's dependencies (`pip install -e ".[dev,rlt-pilot]"`); the app starts
`python -m datagoblin.hillclimb.server` with `src/` on `PYTHONPATH`. `--teacher` lists bigger local
servers for `suite_status` to probe. Only loopback model URLs are accepted. Each run prints a JSON
receipt and appends it to `runs/hillclimb/receipts.jsonl`: tool calls and outcomes, moves, verdicts,
token usage and an OpenAI-style transcript. `climb` repeats `/datagoblin-hillclimb <lane>` with a fresh
conversation per move and stops when the lane plateaus or is exhausted, or after two runs without a
decision. `python -m datagoblin.hillclimb --lane looped_modules` prints the ledger state.

Keep Cargo's target directory outside OneDrive (`CARGO_TARGET_DIR`); `gdk/target/` is ignored.

## Checks

```bash
python -m pytest tests/unit/test_hillclimb.py
cargo test                                          # in gdk/
python gdk/ci_smoke.py gdk/target/release/data-goblin --python python
```

[`ci_smoke.py`](ci_smoke.py) drives the binary with a scripted OpenAI-compatible endpoint: the slash
command and a lane card, `suite_status`, `propose_move`, `rlt_make_suite`, the same bad `rlt_run` call
three times (two refusals that name the missing parameter and suggest `run:r0001`, then the app's guard
against repeating a failed call), a real CPU fit of RLT controls, a loopback recipe the app must deny at
`--approve ledger`, `record_result` and `decide`, then checks the offered tool schemas, the receipt, the
transcript and the ledger chain. A second run checks the text tool protocol with write tools hidden. CI
runs all three in [`.github/workflows/datagoblin-gdk.yml`](../.github/workflows/datagoblin-gdk.yml).

## Live check (2026-11-28)

Driver: Ternary Bonsai 2 4B PTQ1_0 on CPU (PrismML llama.cpp b12178, 6 threads, thinking off, 3-10
prompt tokens/s on a shared, saturated CPU; the GPU was power-capped with other jobs queued), lane
`looped_modules`, `--approve ledger`. Each conversation took 15-28 minutes.

| run | app at the time | what the model did |
|---|---|---|
| 1 | `run_recipe` with a free-form `params` object | status, lane card, recipes, a proposal and `rlt_make_suite`, then 17 `rlt_run` calls with empty params against refusals that did not name the parameter, until the 24-turn cap |
| 2 | refusals name the parameter; repeated-failure guard | the same empty params; the guard stopped each loop after two failures, and the model abandoned the move and switched lanes |
| 3 | one tool per recipe | 24 calls, none failed: baseline 0.301, three moves below the gate, then a promote to 0.424 by fitting seed 19 alone |
| 4 | seeds pinned; abandoning a move with a result refused | `climb --moves 3`: baseline 0.317, then readout `endpoint` with depth 16 (0.308), `training_depth` 2 (0.305) and a return to the baseline settings (0.317 again; the fit is deterministic), each closed by `decide` as reject; plateau 3 of 4 |

Run 3's promotion came from choosing a lucky fit seed, which is why `rlt_run` now pins seeds 7, 19 and
91; its below-gate moves were closed with `abandon_move`, which is why that tool now refuses a move that
has a result. Across runs 3 and 4 the model kept to one lane, filled recipe parameters, chained outputs
with `run:<run_id>`, and recorded the metric for every evaluation run. It sometimes changed two settings
in one move and reran `rlt_make_suite` instead of reusing the suite run; the suite digest was identical,
so the eval contract held. Its remaining refused calls were a run on an already closed move and
`propose_move` with `parent: ""` or `"None"`, which the server now reads as no parent.

## Closing the self_adapter loop

adapter-bits (MoralityLabAI/Compactification-Theory) `experiments/adaptergym/plans/driver_adapter.json`
trains and measures an adapter on the driver itself:

- `train-driver`: bonsai-gym config `b25_driver_fp_r8`, a rank-8 LoRA on Ternary Bonsai 2.5 27B
  PTQ1_0, one epoch (250 steps x 8 rows) over the curated `admitted.jsonl` of
  `bench make --split train --seed 1 --count 2000` -> `bench export-sft` -> `curate` (2000 exact-oracle
  rows). Val: 89 rows from train seed 2, minus the 7 whose view also occurs in seed 1. Dev tasks are
  never trained on (the splits are disjoint by view digest).
- `eval-driver` (adaptergym step kind `hillclimb`, one 9 GB lease): for each arm, base and then base
  plus adapter, it serves the model with llama-server (plain decoding, thinking off) and runs
  `data-goblin run --skill datagoblin-hillclimb --args "self_adapter. ..." --approve local`, so the
  served model drives its own evaluation move: `bench_make` (dev, seed 0, count 120), `driver_bench`
  (limit 120), `record_result`, `decide`. If no app run reaches a decision,
  `python -m datagoblin.hillclimb.drive` ([`drive.py`](../src/datagoblin/hillclimb/drive.py)) makes the
  same calls: it continues an open move, reuses the dev-bench run whose task list matches the lane's eval
  contract (and refuses before writing anything when none does), decides, and on replicate records one
  more `driver_bench` run. Each arm's verdict, and whether the app or the script recorded it, goes to
  `<work>/results/driver-eval-<arm>.json`.
