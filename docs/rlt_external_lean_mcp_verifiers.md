# RLT external verifier adapters: Lean and MCP post-failure decisions

Date: 2026-11-21. Status: **source-only integration interfaces and fixture tests**.
No new Lean kernel campaign or MCP platformer execution is claimed.

Two separate source modules extend the seven-family post-failure gym:

- external_checked_exit.py requires an independent trusted checker for **every**
  attempted candidate and recomputes the complete fixed-depth trajectory.
- external_native_adapters.py contains LeanKernelChecker and MCPStateChecker,
  which connect that contract to an operator-controlled Lean4 session or
  an MCP environment and separate private effect judge.

## Observation contract

An episode is a frozen full-depth source-root trajectory of 2–8 attempts.
Each attempt stores a zero-based visit, a candidate and an
environment-owned verification receipt. Split, source root, model SHA-256,
source revision SHA-256 and verifier SHA-256 are bound to the episode.

The post-failure stopping decision sees only the first candidate, the
public task, the returned checker code and the remaining check budget.
It may learn the **hindsight** label that a later attempt succeeded, but
never receives that future label at inference. Measured future visits
are replay evidence; they are not runtime model-compute savings.

The caller supplies:
- a live checker function with a pinned verifier SHA-256;
- a trusted source revision SHA-256 from its frozen environment;
- an explicit allowlist of fresh TRAIN roots eligible for training.

Fresh source labels alone do not unlock training. Previously exposed
Lean-CGRT, historical MCP datasets and all CAL/ordinary/OOD cohorts
remain training-ineligible. Sealed-poison inputs are rejected.

## Lean adapter

LeanKernelChecker is designed around the existing
[DataGoblin rlt_lean_ldt](../src/datagoblin/training_gym/rlt_lean_ldt.py)
Lean session API. It receives a trusted map of each public task digest
to a **frozen root state SID** and an ID-to-tactic menu.

The model proposes only a candidate ID. The checker retrieves its
pre-registered tactic, replays it from the original root SID with
lean.run, confirms a closed goal list, and calls lean.kernel on the
new state. A tactic passes only if the kernel returns ok=true and
its axioms are drawn exclusively from propext, Quot.sound and
Classical.choice. Invalid IDs and forbidden tactic forms cannot
be executed. Kernel failure remains a negative outcome.

The actual Lean 4 binary version, Mathlib/REPL sources and
trust-compilation policy must be verified by the launcher, with their
hash bound into trusted source and verifier revisions.
Testing with a FakeLean object only establishes interface behavior.
This branch has not run a new live Lean kernel experiment.

Historical Lean-CGRT's 588 certifiable goals and their archived ranker
outcomes are an evaluation-only population. New proof roots must be
sampled and checked with a fresh version-pinned Lean environment
before providing training labels.

## MCP adapter

MCPStateChecker accepts a frozen source case keyed by public task
digest, an action-ID menu and registered immutable snapshot SHA.
Every proposal must be one authorized action ID. The provided
execute_fresh function resets the environment for this specific
candidate to its registered snapshot and returns a result containing
its action ID and snapshot identity. A separate judge checks whether
the **actual state effect** satisfies the task contract.

For the existing
[HybridTRMLDT MCP-03 gym](https://github.com/MoralityLabAI/HybridTRMLDT/tree/gym/mcp-local-retrieval-20261119/MCP/mcp03),
candidate effects can include a repository context packet. The
independent judge should score required-file and symbol coverage
against the private gold patch, plus token budget and source SHA.
For MCP platformer defects, an independent test replay should check
the edited environment or repository state at the task's base commit.

A valid JSON result or the model's assertion of success is insufficient.
The test harness contains a fixture executor and separate symbol judge,
which validates the boundary. Genuine MCP-03 or SWE-bench execution
must be wired by an operator with access to its immutable repository
snapshot and private judge. No actual tool calls were executed by
these fixture tests.

## Source-only validation

An isolated local Python 3 implementation of the checker contracts
passed **11 fixture tests** on 2026-11-21, covering source binding,
historical-data exclusion, training allowlists, invalid IDs, kernel
axiom rejection, MCP snapshot mismatch and altered verifier receipts.
The matching fixture tests were committed to PR #10. The isolated
check was not a checkout-wide run of DataGoblin's repository, since
this container cannot clone the private GitHub repository and the
repository's GitHub Actions jobs fail before runner assignment.

Once a capable environment is available, run:

    python -m pytest -q tests/unit/test_rlt_gym_postfailure_exit.py
    python -m pytest -q tests/unit/test_rlt_gym_external_checked_exit.py \
      tests/unit/test_rlt_gym_external_native_adapters.py
    ruff check src/datagoblin/training_gym/rlt_gym/postfailure_exit.py \
      src/datagoblin/training_gym/rlt_gym/external_checked_exit.py \
      src/datagoblin/training_gym/rlt_gym/external_native_adapters.py

The new code does not alter the sealed populations, results, or verdicts
in the existing BPD/Lean/EvalMath campaigns.
