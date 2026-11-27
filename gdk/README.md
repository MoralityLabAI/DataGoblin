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
  `ledger_status`, `list_recipes`, `propose_move`, `run_recipe`, `record_result`, `decide`,
  `abandon_move`, over MCP stdio (protocol 2025-06-18).
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
command and a lane card, `suite_status`, `propose_move`, a real CPU fit of RLT controls
(`rlt_make_suite`, `rlt_run`), a loopback recipe the app must deny at `--approve ledger`,
`record_result` and `decide`, then checks the receipt, the transcript and the ledger chain. A second run
checks the text tool protocol with write tools hidden. CI runs all three in
[`.github/workflows/datagoblin-gdk.yml`](../.github/workflows/datagoblin-gdk.yml).
