---
name: datagoblin-storyworld-native-gym
description: Prepare bounded SweepWeave JSON and SWMD-min repair tasks, play native HG-02 hypergames, and collect split-safe, independently verified DataGoblin traces.
---

# DataGoblin Storyworld native gym

Use the three-mode gym in MoralityLabAI/DataGoblin with a read-only MoralityLabAI/GPTStoryworld checkout. Capture GPTStoryworld git HEAD and source-file SHA-256 values before running. Follow the upstream skills at codex-skills/storyworld-building/SKILL.md and codex-skills/small-storyworld-builder/SKILL.md for actual storyworld authoring.

## Builder mode

Read the target encounter, option and reaction. Compare the three allowable successor summaries against the narrative continuity of the source encounter. Return exactly PICK followed by one candidate ID, with no prose. The verifier copies the original JSON, inserts the chosen successor and invokes the original SweepWeave validator. It compares the resulting successor to the original source edge. A structurally valid wrong destination is a failure.

This pilot tests bounded repair. New encounters, dialogue writing, reaction effects, gate thresholds and whole-world materialization require separate registered tasks and the original production-quality gate.

## Mini mode

Read the single SWMD-0-MIN encounter card with its broken ORX successor. Respond PICK followed by the corrected ID. Preserve encounter IDs, option IDs, reaction IDs, reaction expressions, and unrelated edges. Keep the request within 8,192 context tokens, reserve output capacity, and use one encounter per turn.

The native exporter and native encounter index parser check the repaired card. Record model_parse_ok separately from any later pipeline repairs if a future model-driven MCP patch engine is installed. Current controls do no automatic model repair.

## Hypergame mode

The actor reads the source HG-02 public observation, including legal_actions and a masked hidden coordinate. Choose one legal action, then receive a new observation from the simulator. Continue until terminal. Use verified true utility and release constraints; treat cosmetic and proxy scores as potential distractors.

Never send the sealed private_bit, the unchosen futures, or hidden grade to the actor. The simulator owns action legality, hidden-state reveals, transitions and scoring. The Bellman oracle, blind-route, proxy and metric-greedy policies are source-native reference controls. A shortcut can earn proxy points while producing an unsafe release.

## Training custody

1. Pin the source SHA and preserve source filenames and content hashes.
2. Generate deterministic public-seed smoke roots, then freeze fresh private-seeded confirmation roots.
3. Store public_tasks.jsonl separately from private_sealed.jsonl.
4. Retain all negative actions and native failing grades in baseline_rollouts.jsonl.
5. Give train_targets.jsonl to the learner only on the train split. Count lineage presentations across builder and mini copies, and across paired HG private-bit worlds. Enforce the four-view lineage cap.
6. Grade completed model actions through the source-native verifier. Never let teacher prose or a model's prediction hard-promote its own evidence.
7. Replay the full archive from the exact same source and seed. Quarantine any source SHA mismatch or hash failure.
8. Compare model completion against matched source controls on held-out lineages, and include actual verifier calls, full HG-02 utility, false accepts, and stable context budgets in promotion reports.

## Runtime and evidence

Use CPU-only source-native controls by default. Optional model rollouts should use the already-running local llama.cpp endpoint and a fixed Inkling/Inky Mini model identity. Keep model context at or below 8K for Mini and at most 16K across all modes. Do not restart the inference server or use a paid fallback. Capture raw outputs and token usage when supplied by the endpoint.

Current evidence class is source_native_cpu_controls. Model learning, story quality improvements, real multiplayer hypergame performance and HG-07/08 mechanistic effects require their own experiments and controlled confirmations.

## Paths

Source module: src/datagoblin/training_gym/rlt_gym/storyworld_native.py

Experiment guide: training/rlt_gym/storyworld_native/README.md

Source of truth: GPTStoryworld commit 5cd0b480e9a01d07418cdf8cb3bedbc9d38ef906 at first registration.
