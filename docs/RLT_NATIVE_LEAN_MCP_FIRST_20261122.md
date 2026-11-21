# RLT-NATIVE-01: real Lean kernel and MCP-03 stdio verifiers

**Date:** 2026-11-22 (Asia/Manila), first execution and exact-source follow-up.
**Evidence class:** integration positive controls on freshly generated **synthetic tasks**.
No pretrained model, learned stopping gate, new Lean tactic learning, SWE-bench
repair success or Inkling cross-loop feature is established.

## Live Lean4 compiler and kernel recheck

[GitHub Actions #44934589904](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44934589904)
and later
[success #44934740098](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44934740098)
run the actual pinned Lean **4.22.0** native executable on a hosted Linux
runner. The release identifies Lean commit
`ba2cbbf09d4978f416e0ebd1fceeebc2c4138c05` and binary SHA-256
`cbf5fd536e142ef1beaccf33f788fd8a7f3f29fb214e75c11319a8d8677b4b2b`.

The campaign creates 12 new salted elementary Nat propositions, using three
roots in each TRAIN/CAL/ordinary/OOD partition and a deterministic two-candidate
tactic menu. Successful tactics are recompiled in a **separate Lean process**
with `#print axioms` and rejected unless the axiom output is explicit and
within the existing allowlist.

| Native Lean measurement | Count |
| --- | ---: |
| Independent generated source roots | 12 |
| Original tactic checks | 24 |
| Additional full trajectory replays | 24 |
| Actual Lean compilations including axiom audits | 72 |
| Successfully kernel-certified proof roots | 12/12 |
| First-visit valid proofs | 3/12 |
| Later certified repairs | 9/12 |
| Approved training-source roots | 3 |
| Separate evaluation-only roots | 9 |

The compiler audit uses the **same pinned Lean executable** in a separate
process, not a second proof assistant. New source goals and tactic selection
are engineering controls. Previous Lean-CGRT results and Mathlib goals were
never copied into the training partition.

The first archive is in the Actions artifact
`dg-rlt-lean4-native-kernel-roots`, including every compiled source,
compiler-status hash, independently replayed proof, task-root identity,
verification label and model/source digest. Exact executable identity is
recorded, including its full SHA-256 rather than only an elan launcher path.

## Real MCP-03 JSON-RPC tool execution

The actual HybridTRMLDT repository source was pinned to commit
`7e90a2d446dbb2f994a33632350e7219bdee1ba4` on branch
`gym/mcp-local-retrieval-20261119`. Its MCP-03
`goblin_server.py` was launched through the native stdio JSON-RPC transport.
Each candidate received a new initialized subprocess and used
`tools/list` and `tools/call(read_symbol)`. The source itself was a
freshly generated small Python working tree, SHA-bound prior to the tool call.

The independent evaluator parsed the original frozen Python source,
checked that the returned symbol's bytes matched the separately read
file and compared the tool's observable effect to a private target
symbol. The model's claimed result was never trusted.

| Native MCP measurement | Count |
| --- | ---: |
| Fresh source-repository roots | 12 |
| Original real stdio `read_symbol` tool calls | 24 |
| Independent full candidate-replay tool calls | 24 |
| Actual initialized stdio sessions | 48 |
| Target symbols covered by depth two | 12/12 |
| First-visit target coverage | 4/12 |
| Later tool-read rescues | 8/12 |
| Approved training-source roots | 3 |
| Evaluation-only roots | 9 |

[First actual MCP tool run #44934619090](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44934619090)
completed the real transport/verification step and failed in a later
Ruff formatting-only step. A subsequent
[green rerun #44934751616](https://github.com/MoralityLabAI/DataGoblin/actions/runs/44934751616)
passed the unit tests, source-pin check, native tool interactions and lint.

**The tool effect here is verified symbol coverage.** These calls do
not edit repository files and do not establish SWE-bench patch
correctness, remote MCP-provider behavior or real model-driven tool
choice. The MCP-03 server supports a documented small subset of MCP
2025-06-18 over newline-delimited JSON-RPC.

## Custody and scientific boundary

The common external checker contract hashes the operator-owned
source revision and verifier, independently replays each attempt,
and separates explicit approved TRAIN roots from ordinary
development/evaluation-only tasks. The output's
`future_rescue` field is labeled hindsight data; it is
never supplied to the candidate-selection policy.

The deterministic choice menu creates an engineering positive control.
High apparent rescue counts are constructed by the available menu
order and cannot establish predictive value-of-compute learning.
The actual wall time and tool behavior can later serve as a
hardware-cost baseline when an RLT controller chooses whether to
continue using observable failure codes.

A follow-up with independent fresh user-like Lean problems and
repository defects must select task/model interventions on TRAIN
roots, freeze a stopping policy on CAL, and evaluate verified
success/cost on never-observed source roots. The production-oriented
Lean/MCP adapters need a native pinned Lean REPL and source-snapshot
judge, with no historical-corpus training leakage.

## Reproduction

Use the repository workflows
`.github/workflows/rlt-live-lean-kernel.yml` and
`.github/workflows/rlt-live-mcp03-stdio.yml`.
The Lean workflow installs `leanprover/lean4:v4.22.0` with elan and
checks the installed binary before execution. The MCP workflow checks
out only exact HybridTRMLDT commit
`7e90a2d446dbb2f994a33632350e7219bdee1ba4`, then uses
the local stdio client and server. All sources and result digests are
retained in the Actions artifacts.

Each job enforces bounded task count, source size, subprocess or tool
timeout, and CPU-only execution. Neither experiment opened any
sealed-poison, NC09, or archived protected population.
