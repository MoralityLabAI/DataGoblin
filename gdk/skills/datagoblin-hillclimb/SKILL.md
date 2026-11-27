---
name: datagoblin-hillclimb
description: Run one gated DataGoblin hill-climb move. Lanes are adapters on this model itself, looped and recurrent modules, new evals and rollouts from bigger local models (Agonist), curation of training data, and other suite runs. Propose a falsifiable move, run allowlisted recipes, record the metric and let the frozen gate decide.
metadata:
  version: 1
  lanes: [self_adapter, looped_modules, agonist_evals, agonist_rollouts, curation, research_loop]
---

# DataGoblin hill climb

You climb one lane by one gated move at a time. The hill-climb tools own the truth: they run
allowlisted recipes, read metric values from run artifacts and apply each lane's frozen gate. You
choose which lane to climb, what to change and why. A move whose result the gate rejects is still
useful: it narrows the search and stays in the ledger.

## Lanes

| lane | climbs | metric (selection split) |
|---|---|---|
| self_adapter | your own decision accuracy, by training an adapter on verified hill-climb decisions | bench_accuracy (dev) |
| looped_modules | verified success of recurrent (RLT) students | rlt_success_rate (calibration) |
| agonist_evals | eval worlds a bigger local model builds that the world verifier accepts | verified_pair_yield (dev) |
| agonist_rollouts | RLT tasks a bigger local teacher solves under the environment verifier | rollout_accuracy (train) |
| curation | unique train roots admitted under split custody | admitted_roots (train) |
| research_loop | offline DataGoblin demos and checks; nothing is promoted | none |

Each lane has a card with its move menu: `load_skill(name: "datagoblin-hillclimb/lanes/<lane>.md")`.

## One move

1. **Observe.** Call `suite_status`. Note `max_tier`, and for each lane its status, incumbent,
   plateau against patience, moves_used against max_moves and open_move. Note `endpoints`: the
   local model servers and their model ids (the driver, and bigger teachers for the Agonist lanes),
   and `gpu`.
2. **Choose the lane.** If this skill was invoked with a lane, use it. Otherwise finish any open
   move first; then pick a lane with `eligible_for_new_move: true` whose next step is runnable at
   `max_tier` and whose inputs exist (see Data flow). Prefer the lane where one change is most likely
   to clear the gate.
3. **Read the card and the recipes.** Load the lane card, then `list_recipes(lane)`. A recipe with
   `allowed: false` is above this session's tier; with `available: false` it needs an external
   checkout. Omitted parameters take their defaults.
4. **Propose.** `propose_move(lane, hypothesis, plan, expected)`. The hypothesis names one change,
   why it should move the metric, and the result that would refute it. `plan` lists the recipes you
   will run; `expected` is the metric value you predict.
5. **Run.** Each recipe is its own tool, named like the recipe (`rlt_make_suite`, `rlt_run`,
   `curate`, ...). Call it with `move_id` and the recipe's parameters as top-level arguments, for
   example `rlt_run(move_id: "m0003", suite: "run:r0007", depths: [1, 2, 4, 8, 16])`. Pass an
   earlier output as `run:<run_id>` or `run:<run_id>/sub/path` (the `out` field of a run's result);
   outputs of earlier moves are reusable. Read `status` and `stderr_tail`. If a call is refused, the
   error names the parameter and what it expects: fix that argument. If a run fails twice,
   `abandon_move` with the reason.
6. **Record.** `record_result(move_id, run_id)` on the run whose `produces_lane_metric` is true. The
   service extracts the value and previews the verdict.
7. **Decide.** `decide(move_id)`, whatever the value: a worse result is a reject, and the ledger
   must say so. `abandon_move` is only for a move whose runs cannot produce a result at all. On
   `replicate`, rerun the evaluation recipe (a new seed where the recipe takes one, on the same eval
   set), record that run, and decide again.
8. **Report and stop** in one short block: lane, move id, runs (ids and recipes), recorded value
   against the incumbent, verdict, and the next move you recommend. One move per invocation:
   `data-goblin climb` starts the next move in a fresh conversation.

## Gate rules

The service applies these; know them so your proposals can pass.

- A result off the lane's selection split, or with no finite value, is **invalid**.
- With no incumbent, the first valid result is the **baseline**.
- delta = value - incumbent when the metric goes up (incumbent - value when it goes down); with
  two or more results, value is their mean.
- delta >= min_delta: **promote** (the move becomes the incumbent).
- min_delta / 2 <= delta < min_delta with one result: **replicate** (one more result, then decide).
- Otherwise **reject**. A reject or invalid adds one to the plateau; promote and baseline reset it.
  At plateau = patience the lane stops; at moves_used = max_moves it is exhausted.
- **Eval contract.** Once a lane has an incumbent, its results must come from the same eval set
  (the same bench tasks and limit, the same RLT suite). A result from another eval set is invalid,
  so reuse the run id of the eval set the incumbent used.
- One open move per lane. Tiers are offline < loopback < gpu; a recipe above `max_tier` is
  refused. Do not ask for a higher tier; report what the next tier would run and stop.

## Data flow between lanes

- agonist_evals: verified dev-split world pairs are new eval families; train-split candidates go to
  curation.
- agonist_rollouts: verified teacher rollouts (`sft.jsonl`, `repairs.jsonl`, `aleph.jsonl` from
  rollout_score) go to curation, then to looped-module and adapter training.
- self_adapter: exact bench answers, verified teacher answers and clean app transcripts
  (export_driver_sft) go to curation, then to adapter training on this model.
- curation feeds training. Training on a GPU is the gpu tier (adaptergym launches through
  vramorama leases); with a lower tier, stop at a curated set and report it.

## Custody and honesty

- Train only on train rows. Never put dev, calibration or test outputs into a training input;
  curate drops them, and the gate invalidates results off the selection split.
- A model's proposal or explanation is a claim. Values count only when a tool returned them.
- Bigger-model outputs are candidates until a verifier accepts them. Fixture replies are wiring
  checks; the service refuses them as results.
- Keep failures. Do not rerun a rejected configuration unchanged.
- GPU: read `suite_status.gpu` first; one queued GPU job per session; never start, stop or reuse a
  model server that this session did not start, except to send requests to an endpoint listed in
  `suite_status.endpoints`.

## Other DataGoblin skills

These are operator documents in the DataGoblin checkout, not GDK skills: `Agonist/SKILL.md`
(taste-to-world gym), `Agonist/skills/agonist02/SKILL.md` (student uplift),
`training/rlt_gym/hypergame_taste/skills/research_hillclimb/SKILL.md` (RLT repair controller).
