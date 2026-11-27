# Research Verifiers v2.1: Controller-Assisted Research and Externalized Persona

`datagoblin-research-v2.1` is a separately versioned successor to `datagoblin-research-v2`
([PR #81](https://github.com/MoralityLabAI/DataGoblin/pull/81) and historical benchmark baseline
[PR #93](https://github.com/MoralityLabAI/DataGoblin/pull/93), commit `d4aa46c3ff68f03395cc0edadd5faae05782cd84`).
All `datagoblin.research_v2` modules and their frozen `implementation_digest()`
(`73ac4d4d57825c708f45cf73aa56a516290f825425996146131bab181891a339`) remain completely unchanged.

## Motivation & Architectural Delta from v2.0

Empirical evaluation of compact models on Research Verifiers v2.0 (PR #93) isolated three
structural failure modes distinct from underlying model capability:

1. **Cross-episode budget starvation:** v2.0 capped total actions at 64 across an entire 32-case run,
   so deterministic oscillation in episodes 1–8 left episodes 9–32 unattempted.
2. **Stateless observation amnesia:** v2.0 replaced the prior turn's observation on each turn,
   causing deterministic `temperature=0` models to enter infinite `run_experiment <-> check`
   2-cycles even *after* receiving `check_passed = True`.
3. **Binary terminal collapse:** v2.0 collapsed valid JSON syntax, real optimizer fits, and
   verified intermediate `check_passed = True` solutions to `0.0` whenever `submit` was omitted.

`datagoblin-research-v2.1` addresses these directly while keeping the strict terminal `primary_score`
and independent Linux `RLIMIT` worker sandbox intact:

| Dimension | Research Verifiers v2.0 (Frozen) | Research Verifiers v2.1 (Successor) |
|---|---|---|
| Schema / Package | `datagoblin-research-v2` (`datagoblin.research_v2`) | `datagoblin-research-v2.1` (`datagoblin.research_v2_1`) |
| Action Budget | Global 64 actions across all episodes | **Per-episode action budget** (up to 8 turns/episode, no cross-episode starvation) |
| Episode State | Stateless current observation only | **Persistent `compact_state`** (`turn_log`, `disallowed_actions`, `next_valid_actions`, `verified_ready`, `matched_pairs`) |
| Scoring | Terminal `primary_score` only | Strict terminal `primary_score` **reported separately alongside** transparent intermediate `capability_score` in `[0, 1]` |
| Controller Modes | Autonomous only | 5 auditable modes: `autonomous`, `deterministic_stopping`, `learned_ldt_stopping`, `trm_action_routing`, `rlt_state_correction` |
| Persona Semantics | `research_v2.suite.persona_memory` | **Independent declarative event-sourced reference** (`persona_store_reference.project_persona_store`) with explicit rejection/tombstone ledger |

## Intermediate Verified Capability Score (`capability_score`)

`capability_score` is strictly auxiliary and never overwrites `primary_score`, `success`, or
`integrity_pass`:

- **Symbolic families** (`experiment_design`, `boolean_repair`, `belief_update`, `evidence_custody`, `cursor_recovery`):
  - `valid_json_turn`: `+0.15`
  - `schema_valid_action`: `+0.15`
  - `verifier_check_passed`: `+0.35`
  - `terminal_submitted`: `+0.15`
  - `terminal_verified` (`primary_score == 1.0`): `+0.20`
- **Learner families** (`trm_training`, `ldt_filter`, `persona_externalization`):
  - `valid_json_turn`: `+0.10`
  - `schema_valid_config` (at least one persisted artifact): `+0.15`
  - `real_optimizer_fit` (`steps > 0`, `weights_sha256 != init_weights_sha256`, `loss_after < loss_before`): `+0.20`
  - `matched_zero_baseline` (matched `steps = 0` initialization pair): `+0.15`
  - `terminal_submitted`: `+0.15`
  - `integrity_and_generalization`: `+0.25 * primary_score`

## Auditable Controller Modes

Every controller operates strictly on **public episode state** (`CompactResearchState`, public task
metadata, and the model's own proposed JSON text). No controller accesses private evaluation seeds
or privileged `reference_answer()`. Every intervention and skipped model turn is recorded inside
`report.json["controller_accounting"]`.

1. **`autonomous`:** Zero controller interventions; tests the effect of persistent `compact_state`
   and per-episode budgets alone.
2. **`deterministic_stopping`:** Pre-turn deterministic stopping after `verified_answer` is
   established (`check_passed = True`) or a verified matched `(baseline, trained)` learner pair
   exists (plus independent `reference_persona_memory` projection on `persona_externalization`).
3. **`learned_ldt_stopping`:** 45-parameter additive LDT stopping classifier (`14 x 3 + 3`) trained
   on fresh synthetic state transitions with a hard structural safety veto (cannot stop unless
   verified state exists).
4. **`trm_action_routing`:** 2-step tied Tiny Recursive Model (TRM) 5-class action router
   (389 parameters) that stops on verified state, sanitizes schema-aliased learner configs
   (`trm_width -> width`, `ldt_critical_veto -> critical_veto`, strips extra keys while preserving
   model-chosen hyperparameters), inserts missing matched `steps=0` baselines, and routes illegal
   symbolic `run_experiment` / exhausted `check` calls.
5. **`rlt_state_correction`:** Combines the TRM action router with a `K=3` damped RLT residual
   corrector trained using **exact unrolled BPTT** through the residual sum, `tanh`, top1-top2
   margin, and softmax Jacobian (verified against central finite differences to `< 1e-6`).

## Independent Declarative Persona Reference

[`src/datagoblin/research_v2_1/persona_store_reference.py`](../src/datagoblin/research_v2_1/persona_store_reference.py)
implements `project_persona_store(...)` as a pure, first-principles event-sourced projection:

1. Filters raw events by owner binding, scope binding, temporal horizon (`revision <= at_revision`),
   and key allowlist, counting every rejected event by category.
2. Partitions admitted events by key and orders each key's history by `(revision, arrival_index)`.
3. Folds `set` and `revoke` operations with explicit tombstone tracking (supporting clean
   resurrection if a later `set` supersedes a `revoke`).
4. Projects only live (non-tombstoned) key-value pairs alongside the full rejection and tombstone
   audit ledger.

## CLI & Verifiers v1 Adapter Usage

```bash
# Install v2.1 Verifiers adapter alongside DataGoblin
python -m pip install -e '.[dev]' -e environments/datagoblin_research_v2_1

# Validate gold & noop trajectories under Linux RLIMIT subprocess isolation
vf-validate datagoblin-research-v2-1 --runtime.type subprocess -n 8

# Build a fresh suite disjoint from all historical and training registries
python -m datagoblin.research_v2_1 build \
  --out /path/outside/git/prospective-v2.1 \
  --split heldout \
  --per-family 4 \
  --exclude-registry /path/outside/git/heldout-panel-20261126/held_registry.json

# Run a bound endpoint under any controller mode
python -m datagoblin.research_v2_1 run \
  --suite /path/outside/git/prospective-v2.1 \
  --out /path/outside/git/run-rlt \
  --model-id qwen3-4b-instruct-2507-q8 \
  --binding /path/outside/git/binding.json \
  --controller-mode rlt_state_correction

# Compute paired within-family bootstrap comparisons (primary_score & capability_score)
python -m datagoblin.research_v2_1 compare \
  /path/outside/git/run-autonomous \
  /path/outside/git/run-deterministic_stopping \
  /path/outside/git/run-learned_ldt_stopping \
  /path/outside/git/run-trm_action_routing \
  /path/outside/git/run-rlt_state_correction \
  --out /path/outside/git/comparison.json
```
