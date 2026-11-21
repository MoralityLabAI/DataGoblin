# RLT-LIVE-01: native Lean 4 kernel and MCP-03 source-effect verification

Date: 2026-11-22. **Development engineering positive controls.**
These native runs connect the verifier interfaces merged in DataGoblin PR #10
to actual Lean 4 and the HybridTRMLDT MCP-03 repository tool substrate.
The candidate menus are fixed and intentionally elementary. They do not
demonstrate a learned tactic selector or cross-domain reasoning transfer.

## Provenance

[Successful GitHub Actions #44934900809](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44934900809)
ran both jobs. Artifacts are named dg-native-lean4-postfailure and
dg-native-mcp03-postfailure. TRAIN, CALIBRATION, ordinary_test and
ood_test source roots come from seed 20261122; old Lean-CGRT goals,
SWE-bench targets and protected/sealed data are excluded.

Lean: pinned version 4.12.0 (commit dc2533473114), actual compiler ELF
SHA-256 b52e0a7b6a36c30281085d9b9db64c289e4368a2694b4185668e7d63822f43ac.

MCP: HybridTRMLDT source commit
7e90a2d446dbb2f994a33632350e7219bdee1ba4.
Actual MCP-03 tool and snapshot source checksums are in the artifact.

## Lean: real kernel-checking and full replay

| Source split | Fresh goals | Kernel-certified within 3 candidates | First candidate passed | Later repaired |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 8 | 8 | 2 | 6 |
| CALIBRATION | 8 | 8 | 2 | 6 |
| Ordinary development | 8 | 8 | 2 | 6 |
| OOD development | 8 | 8 | 2 | 6 |
| Total | **32** | **32** | **8** | **24** |

The driver compiles a standalone Lean theorem for every attempt. A
successful proof is recompiled and its explicit Lean axiom report
is parsed, then the existing LeanKernelChecker checks the permitted
axiom set. The source and replay required 304 actual compiler
invocations, including 112 kernel-focused replays. The artifact
contains 56 distinct certified Lean proof source files with SHA-256
digests, together with 96 candidate receipts and independently
replayed source-root labels.

Four generated theorem families cover elementary addition equations,
self-equality, Nat addition commutativity and conjunction. None of
the previously archived 588 Lean-CGRT tasks are reused.

The first native attempt failed the exact receipt replay because
state IDs changed between compiler passes. A subsequent successful
run certified only 24/32 tasks because the commutativity proof
had an incorrect second-line indentation. Both engineering failures
are preserved. Stable proof IDs and a per-family proof-success gate
now protect the corrected 32/32 campaign.

## MCP: actual tool calls with hidden gold-symbol judge

| Source split | Fresh repositories | Gold recalled within 3 candidates | First candidate recalled gold | Later repaired |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 8 | 8 | 0 | 8 |
| CALIBRATION | 8 | 8 | 0 | 8 |
| Ordinary development | 8 | 8 | 0 | 8 |
| OOD development | 8 | 8 | 0 | 8 |
| Total | **32** | **32** | **0** | **32** |

The MCP driver generates a fresh Python source tree per root, loads
it through MCP-03 snapshot_from_dir, and executes the real in-process
index_children, candidate_refine, code_search, read_outline and
read_symbol operations. Each candidate receives a new CodeEnv and a
fresh snapshot-hash check. A separate private gold-file/symbol judge
evaluates the resulting repository context packet.

Across source recording and independent replay, the campaign made
704 actual MCP-03 tool calls, 192 fresh environment resets, 192 LDT
path checks and 192 private target judgments.

The public task includes a module and symbol hint, with a deliberately
successful late action in the fixed menu. This validates source
inspection and feedback plumbing. It is an in-process MCP-03 tool
surface, not a stdio or HTTP transport test. There is no learned
agent, actual repository edit, SWE-bench score or native Lean-to-MCP
controller transfer demonstrated by these results.

## Training/evaluation custody and next step

The replay_corpus interface checks a trusted source revision, a
pinned verifier hash and an explicit fresh TRAIN-root allowlist.
Eight new examples per substrate are authorized for TRAIN-format
exports. All remaining source roots are evaluation-only. No controller
weights are fitted on these labels in the current experiment.

The next study needs new source-root-disjoint proof tasks with an
actual trained tactic ranker and MCP symbol-localisation tasks
with blinded issue hints and unseen repository snapshots. A learned
stop policy can then be compared with fixed-depth, exact verification
and matched random continuation, using measured native execution
costs and preregistered failure penalties.

## Reproduction

Lean: python -m datagoblin.training_gym.rlt_gym.live_lean_kernel
      --lean PATH_TO_PINNED_LEAN --per-split 8 --out RUN_DIRECTORY

MCP: PYTHONPATH=PATH_TO_PINNED_HYBRIDTRMLDT
     python -m datagoblin.training_gym.rlt_gym.live_mcp03
     --per-split 8 --out RUN_DIRECTORY

Python source tests and Ruff are in the same GitHub Actions workflow.
