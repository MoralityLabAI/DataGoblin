# RLT-BLIND-02: verifier-cost baseline and hidden-symbol development campaign

Date: 2026-11-22. Repository: MoralityLabAI/DataGoblin.

## Evidence baseline from merged PR #12

The two native archives from GitHub Actions run 44934900809 have 32
independent source roots each. The retained Lean proof files and JSONL
manifests can be checked with:

```sh
python -m datagoblin.training_gym.rlt_gym.learned_stop_audit \
  dg-native-lean4-postfailure.zip dg-native-mcp03-postfailure.zip \
  --output local-audit.json
```

At fixed depth 1, Lean certifies 8/32 and MCP recalls 0/32. At depth
2, both reach 32/32. The fixed depth-3 menu makes 96 candidates per
substrate. Exact verifier stopping at the first success makes 56 Lean
and 64 MCP attempts, respectively. These are candidate-attempt counts,
**not** measured end-to-end execution-cost savings.

The native receipts are infrastructure controls. In the TRAIN portion,
Lean has 6 first failures that are rescued and 0 that remain unsuccessful.
MCP has 8 first failures rescued and 0 that remain unsuccessful. A
continuation-versus-stop classifier cannot be identified from these
positive-only first-failure labels. Every historical MCP task gives
the target symbol name or module in the public task.

## New blind MCP development task

`blind_mcp_sources.py` creates 36 opaque Python functions in six modules
per fresh source root. Its public issue describes two input-output pairs
and never supplies a symbol or module name. The first example has a
confuser, so both examples matter. Every group of eight roots contains
six answerable and two negative requests with no matching implementation.

Data custody: TRAIN fits model parameters; CAL chooses the stop
threshold; ordinary_test and ood_test receive a frozen policy. OOD
rewrites the arithmetic expression ordering. Source trees are
disjoint, hashed, and treated as immutable by the independent judge.
The upstream MCP environment receives an empty oracle, so the tool
server has no evaluator-owned target to consult.

The `blind_mcp_pilot.py` runner uses the real MCP-03 CodeEnv interface
from exact pinned HybridTRMLDT commit
`7e90a2d446dbb2f994a33632350e7219bdee1ba4`. It calls
index_children, code_search, read_outline, and read_symbol. It parses
only the generated, constrained arithmetic grammar and trains a
six-feature logistic classifier from TRAIN candidate outcomes.

The pilot collects complete native tool traces and then scores learned
stopping over each prefix. Reported stopped read counts are thus
**counterfactual**. They are not timed online interventions. The exact
behavior matching rule is an essential comparison: if it matches the
trained policy, this shows instrumentation coverage and provides
no evidence of a novel learned search strategy.

## Native execution and promotion gates

The workflow `.github/workflows/rlt-blind-mcp-learned-stop.yml`
installs the package on Python 3.12, runs the tests and Ruff, fetches
the exact upstream commit, executes the real tools, and retains a
development report. The report omits private target symbols.

The first acceptance questions are whether the tool layer returns
the registered opaque symbols, whether all 8 negative development
cases are treated as absent, and how much exact-source reading
is needed for the correct decision. A source mismatch or private
gold exposure invalidates the run.

A subsequent study must add a genuinely trained Lean tactic ranker
on new multi-step theorem roots, with kernel rechecking, negative
candidate pools, and CAL-frozen continuation decisions. Hard Lean
problems, blinded real repository defects, live online latency,
pretrained model comparisons, and sealed confirmation are future gates.
This pilot does not establish them.
