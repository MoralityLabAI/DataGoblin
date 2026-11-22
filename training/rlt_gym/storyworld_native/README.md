# DataGoblin source-native Storyworld gym

This gym connects three existing GPTStoryworld implementations through a read-only, pinned sibling checkout. It produces deterministic CPU reference controls and training/evaluation packets for compact models.

| Mode | Native implementation | First task |
| --- | --- | --- |
| builder | codex-skills/storyworld-building/scripts/sweepweave_validator.py | Repair a missing SweepWeave reaction successor in JSON; check native validity and exact original edge. |
| mini | codex-skills/storyworld-building/scripts/json_to_swmd.py and codex-skills/small-storyworld-builder/scripts/swmd_encounter_index.py | Repair a single SWMD-0-MIN ORX target inside an 8K encounter card, then re-export and parse with upstream code. |
| hypergame | verifiers_envs/storyworld-hypergames-env/src/spiral_ln/hypergames_single_player.py | Play the HG-02 hidden-bit, 140-dimensional game against its source-owned transitions and exact Bellman reference. |

Source inspected: MoralityLabAI/GPTStoryworld commit 5cd0b480e9a01d07418cdf8cb3bedbc9d38ef906. Every archive captures the actual GPTStoryworld HEAD and SHA-256 of the relevant Python tools and source world. Pass --expect-source-commit to refuse another revision. The existing DataGoblin hypergame_taste collector remains the separate synthetic K-depth research-taste proxy.

The first builder and mini tasks specifically evaluate bounded successor repair. Whole-world authoring, prose quality, secret endings, playable reader checks, trained-model performance, and current hypergame multiplayer runtime remain separate evaluation targets. HG-02 provides exact synthetic mechanism results.

## First offline run

Use sibling DataGoblin and GPTStoryworld checkouts. Python 3.11+ and Git suffice for the native module. Generated traces stay outside both source trees.

    export PYTHONPATH=src
    export GPTSTORYWORLD_SOURCE=../GPTStoryworld
    python -m datagoblin.training_gym.rlt_gym.storyworld_native \
      --source "$GPTSTORYWORLD_SOURCE" \
      --expect-source-commit 5cd0b480e9a01d07418cdf8cb3bedbc9d38ef906 \
      --seed 23 --per-mode 6 --out runs/storyworld-native-23
    python -m datagoblin.training_gym.rlt_gym.storyworld_native \
      --source "$GPTSTORYWORLD_SOURCE" --replay runs/storyworld-native-23

The default valid source world is storyworlds/diplomacy/forecast_backstab.json (10 encounters and 42 reactions). Use --world with another valid path relative to GPTStoryworld. The collector requires three distinct encounter IDs and a sufficient number of repairable reactions.

For private held-out roots, generate a private 128-bit or longer seed file outside Git and supply --private-seed-file PATH. Only its SHA-256 commitment enters the manifest. The same file is needed to replay. Public fixed seeds are engineering smoke controls.

## Streams and custody

- public_tasks.jsonl: public observations, response grammar, opaque root, shared lineage, mode and split.
- private_sealed.jsonl: evaluator-only exact successor, source-specific edge, scenario seed and hidden bit.
- baseline_rollouts.jsonl: gold, first-choice and invalid repair controls, plus HG-02 Bellman, blind, proxy and metric controls with source grades.
- train_targets.jsonl: source-native oracle supervision restricted to train.
- manifest.json: source revision, per-file hashes, profile, seed commitment, splits, row counts and SHA-256.

All repairs originating from the same source encounter share a split, including builder and mini versions and distinct reaction targets. Paired builder/mini views also share their exact reaction lineage. Both private-bit HG twins share a scenario lineage and split. The train-only source-target export gives at most four training presentations per lineage across formats or paired hidden worlds. All baseline labels stay explicitly labeled source controls.

## Model response contract

For builder and mini, return exactly PICK followed by a candidate successor ID. Submit a JSONL row with root and action fields:

    {"root":"<public root>","action":"PICK enc_turn_2"}

For HG-02, the learner must receive the current public observation after each simulator action. Submit the resulting ordered legal action sequence under an actions array. A sample syntax row is:

    {"root":"<public root>","actions":["audit","route_0","verify_1","authorize_2","publish"]}

This is an illustrative grammar, not a solved scenario. The native simulator checks every action and all true terminal rewards.

Score submissions through:

    python -m datagoblin.training_gym.rlt_gym.storyworld_native \
      --source "$GPTSTORYWORLD_SOURCE" \
      --grade runs/storyworld-native-23 \
      --submissions runs/model_submissions.jsonl \
      --graded-out runs/model_grades.jsonl

A native structural validator can accept a wrong but syntactically valid edge. Semantic successor restoration therefore has a separate exact score. Hypergame proxy points are kept separate from true utility, safe success and false release.

## Source-native minified compiler SFT

The additional micro-compiler collector runs GPTStoryworld's actual
small-storyworld-builder script, swmd_build_qlora_examples.py, over the
source-generated SWMD-0-MIN document. It collects the original compiler's
four transformations: 40% compile, 25% compression, 20% repair and 15%
targeted edit, subject to deterministic sample rounding.

    python -m datagoblin.training_gym.rlt_gym.storyworld_micro \
      --source "$GPTSTORYWORLD_SOURCE" --seed 23 \
      --examples-per-encounter 4 --out runs/storyworld-micro-23
    python -m datagoblin.training_gym.rlt_gym.storyworld_micro \
      --source "$GPTSTORYWORLD_SOURCE" --replay runs/storyworld-micro-23

Its micro_public.jsonl hides reference outputs, micro_sealed.jsonl keeps
canonical compiler labels under evaluator custody, and micro_train_sft.jsonl
exports messages for train-only supervision. The split assignment is shared
with the repair gym at the source encounter level, preventing source
encounters from appearing in train and test through different variants.
Each input packet has an 8K byte-safe cap, tighter than the stated 8K-token
Mini budget. Source script SHA and output SHA-256 are frozen in a distinct
manifest. The original script's train/validation *row split* is reconstituted
by encounter identity here to avoid same-encounter leakage.

A learned model can return JSON to the micro grader:

    {"root":"<public root>","output":"{\"schema\":\"SWMD-MICRO-0.1\",...}"}

Then run:

    python -m datagoblin.training_gym.rlt_gym.storyworld_micro \
      --grade runs/storyworld-micro-23 \
      --submissions runs/my_micro_submissions.jsonl \
      --graded-out runs/my_micro_grades.jsonl

These are deterministic compiler targets, useful for first-pass SFT and
token-economics studies. Exact JSON equality defines the initial grader.
Independent narrative quality and runtime correctness require additional
validation. Existing GPTStoryworld authored worlds are public data, so
the source-held-out encounters are development controls rather than
unseen-world confirmation.

## Local Inky Mini model rollout (opt-in)

The turn-by-turn runner at
src/datagoblin/training_gym/rlt_gym/storyworld_local.py reuses an
**already running** loopback OpenAI-compatible llama.cpp endpoint. It never
starts a server, downloads weights or calls a remote fallback. Supply the
server's actual model alias, quant and revision manually. It restricts
HTTP requests to the localhost endpoint with an explicit port, rejects
redirects, caps response size and prompts, and forces temperature zero.

First collect a native suite as documented above, then run a small paired
HG-02 evaluation. The calibration split is useful for wiring and debugging.

    python -m datagoblin.training_gym.rlt_gym.storyworld_local \
      --source "$GPTSTORYWORLD_SOURCE" --archive runs/storyworld-native-23 \
      --model inky-mini --model-revision <actual-revision> --quant <actual-quant> \
      --endpoint http://127.0.0.1:8080/v1/chat/completions \
      --mode hypergame --split calibration --max-roots 2 \
      --context-budget 8192 --output-tokens 96 --out runs/storyworld-inky-hg02

At each HG turn, the runner sends the current model-visible native
observation, reads one legal action, passes it to the source-owned simulator
and sends the next observation. The hidden bit remains sealed until the
native audit action reveals its coordinate. The runner stores every raw
output, action parse, visible state and token usage. True utility and false
acceptance are scored only after the episode. Both secret-bit twins of a
scenario remain grouped and are either selected together or left out.

Use separate output directories for builder and Mini:

    python -m datagoblin.training_gym.rlt_gym.storyworld_local \
      --source "$GPTSTORYWORLD_SOURCE" --archive runs/storyworld-native-23 \
      --model inky-mini --mode mini --split calibration --max-roots 1 \
      --context-budget 8192 --out runs/storyworld-inky-mini

The builder and Mini response grammar is exactly PICK <candidate ID>.
Parsing failures count as failures and remain in the trace. The server
may report its token counts and actual model name, which are included
in each event. A recording of actual runs requires a supplied live
endpoint; the GitHub Actions tests exercise only a fake policy.

A network interruption leaves a partial events.jsonl. Resume it without
repeating completed turns using the exact original settings and --resume:

    python -m datagoblin.training_gym.rlt_gym.storyworld_local \
      --source "$GPTSTORYWORLD_SOURCE" --archive runs/storyworld-native-23 \
      --model inky-mini --mode hypergame --split calibration --max-roots 2 \
      --context-budget 8192 --output-tokens 96 \
      --out runs/storyworld-inky-hg02 --resume

Run a fully offline regrade after completion:

    python -m datagoblin.training_gym.rlt_gym.storyworld_local \
      --source "$GPTSTORYWORLD_SOURCE" --archive runs/storyworld-native-23 \
      --replay-run runs/storyworld-inky-hg02

The replay reconstructs public observations from the sealed simulator,
checks actions against the raw response, recomputes native grades and
verifies the output-file digests. The recorded model trace is distinct
from source-native oracle controls and should be compared on matched
root lineages before training claims.

## Verification and next experiment

    GPTSTORYWORLD_SOURCE=../GPTStoryworld PYTHONPATH=src \
      python -m unittest discover -s tests/unit -p 'test_rlt_gym_storyworld_native.py' -v

The dedicated GitHub Actions job checks out the pinned source Python modules and world, runs the three-mode suite and independently reproduces the collection bytes. Keep hidden seeds, output archives and model weights outside Git.

For learned-model promotion, freeze held-out worlds before development, record full model transcripts and failures, compare paired completion under identical budgets, and report false accepts. Native gold controls and synthetic GF(2) policy ceilings are separate from neural generalization claims. For production-quality builder experiments, also run upstream storyworld_quality_gate.py, ending reachability and the playable Storyworld reader.
