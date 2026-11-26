# MOLT-GOBLIN-02: delayed evidence episodes

Development-only exact host. This extends the static array gym without altering its schema.
Each independent generated root produces a paired relevant-change and unrelated-change episode,
with the same candidates, starting receipts, capacity and distractors. The task is to retain
current evidence, request an updated source receipt if needed, and submit the optimal admissible
array. Old receipts never become current merely by being retained.

The host supplies starting observations, then one event per advance. A relevant source edit
changes the frame and replaces one exclusion, potentially restoring a formerly eliminated
candidate. An unrelated metadata edit leaves the frame unchanged. The new exclusion is visible
only after a charged request. The host independently checks the issued receipt roster, current
frame, eligibility, resource cap and exact optimum (lexicographic tie rule). A valid but suboptimal
array has verified=true and optimal=false. An abstention, stale frame or invalid action is
retained as a terminal failure. This fixture uses one request and bounded turns, including
advances. It has no actual model, tool execution, checkpoint loading or promotion authority.

```sh
python -m pytest -q tests/test_molt_episode.py
python -m datagoblin.training_gym.molt_episode --seed 71 --roots 128 --train-out /tmp/molt-episodes-train.jsonl
```

The export writes 256 exact-policy TRAIN episodes for 128 roots. Rows are typed observation/action
traces, not fabricated private reasoning. Use a separate steward-held seed and the in-memory
Episode API for evaluations; no evaluation trace export is supplied. The exact oracle is a
structural reference, not a teacher-model benchmark. Variant IDs differ, root IDs are shared,
and split namespaces differ; audit semantic overlap before population claims.

Next: independent adapter via the existing RLT/Verifiers environment, restricted history access,
model checkpoint comparison and per-root aggregation. The exact reference policy has full
host-side problem access to solve the final knapsack; its actions request only the changed
receipt. For a learned model the new exclusion is provided by the request observation and
all other required input by the initial observation. The adapter must keep the private World
object inaccessible to the proposer.
