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

## Verification and next experiment

    GPTSTORYWORLD_SOURCE=../GPTStoryworld PYTHONPATH=src \
      python -m unittest discover -s tests/unit -p 'test_rlt_gym_storyworld_native.py' -v

The dedicated GitHub Actions job checks out the pinned source Python modules and world, runs the three-mode suite and independently reproduces the collection bytes. Keep hidden seeds, output archives and model weights outside Git.

For learned-model promotion, freeze held-out worlds before development, record full model transcripts and failures, compare paired completion under identical budgets, and report false accepts. Native gold controls and synthetic GF(2) policy ceilings are separate from neural generalization claims. For production-quality builder experiments, also run upstream storyworld_quality_gate.py, ending reachability and the playable Storyworld reader.
