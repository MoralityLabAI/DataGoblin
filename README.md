# DataGoblin

DataGoblin is an offline-first research orchestrator for synthetic data aimed at non-language-centric learners. It turns a typed learner snapshot into competing falsifiable hypotheses, a bounded data recipe, adversarial critique, narrow builder tickets, independently verified experiment records, and future research-model training examples. Its “constitution” is an executable research-integrity and control policy; it is not proof of alignment.

```text
LearnerSnapshot
      |
    Scout --------> competing hypotheses
      |
  Alchemist ------> DataRecipe + predictions
      |
     Rat ---------> critique + minimal repair
      |
ResearchMembrane -> constitutional decision
      |
 BuilderTickets -> trusted toy builder
      |
causal_switches_v1 -> independent verifier -> NumPy learner
      |
ExperimentRecord -> SQLite + SHA-256 artifacts -> report/JSONL exports
```

Scout identifies unexplained capability gaps and competing mechanisms. Alchemist proposes a recipe that describes worlds, trajectories, interventions, objectives, curricula, splits, solvers, and verifiers. Rat attacks leakage, confounds, metric gaming, permissions, unsupported claims, and missing negative-result interpretations. ResearchMembrane admits only typed monotone updates with appropriate evidence authority; model and experience claims remain useful but cannot silently become verified findings.

## Quickstart

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
python -m datagoblin demo --seed 7 --out runs/demo
```

The demo needs no API, weights, GPU, network, or upstream checkout. It writes the required manifests, tickets, receipts, SQLite ledger, content-addressed artifacts, experiment record, and Markdown report beneath `runs/demo`.

Repository-local caches can be inspected or removed without touching run evidence:

```bash
python scripts/clear_local_caches.py
python scripts/clear_local_caches.py --apply
```

Removing `runs/` requires the explicit `--include-runs` flag. The cleanup script resolves every target beneath the DataGoblin root and never traverses `.git`.

Useful commands:

```bash
datagoblin validate --recipe examples/recipe.good.yaml
datagoblin propose --snapshot examples/learner_snapshot.delayed_causality.json --out runs/plan.json
datagoblin emit-tickets --plan runs/plan.json --out-dir runs/tickets
datagoblin synth-episodes --count 32 --seed 7 --out runs/synthetic.jsonl
datagoblin export-sft --input runs/demo/research_episode.json --out runs/sft.jsonl
datagoblin export-preferences --input runs/demo/research_episode.json --out runs/preferences.jsonl
datagoblin upstream-status
datagoblin gym-demo --seed 7 --out runs/training-gym-demo
datagoblin etiquette-demo --seed 7 --out runs/etiquette-demo
datagoblin etiquette-build-training-dataset --seed 17 --out runs/ldt-fit/dataset
python -m datagoblin.adapter_verifier drift-confirm \
  --out runs/adapter-schema-drift-confirmation
python -m datagoblin.adapter_verifier tournament-confirm \
  --out runs/four-game-adapter-tournament
python -m datagoblin.adapter_verifier skill-tournament-succession \
  --skill-before-fingerprint path/to/skill_before.json \
  --skill-after-fingerprint path/to/skill_after.json \
  --candidate-validator path/to/validate_adapter_candidate.py \
  --out runs/curation-skill-succession
```

The architecture-neutral training gym evaluates LoRA, HRM-Text, and bounded looping-transformer candidates through a
frozen split and metric firewall. The default command runs mock controls only and never loads weights; see
[`docs/training_gym.md`](docs/training_gym.md).

The collection-etiquette pipeline classifies public metadata envelopes into reject, quarantine, discovery, verified,
and evaluation custody lanes. Its LDT is currently a deterministic plumbing control and has no authority to release
training data. See [`docs/data_collection_etiquette.md`](docs/data_collection_etiquette.md).

The four-game curation verifier also has a learned, shape-only schema-drift gate. It fits a small
CPU reference adapter, compares it with the frozen exact-key policy, and requires a different-surface
confirmation before emitting checkpoint-ready LoRA and HRM-Text recipe packs. See
[`docs/curation_adapter_verifier.md`](docs/curation_adapter_verifier.md).
Repeated boundary, authority, and robustness rounds can also produce a split-safe pairwise LDT
selection curriculum; see
[`docs/curation_skill_tournament_succession.md`](docs/curation_skill_tournament_succession.md).
The bounded array-sorter gym fits linear, latent, and weight-tied looped heads to that
curriculum under WSL2 cgroup caps and scores them through a separate steward join; see
[`docs/ldt_array_sorter_gym.md`](docs/ldt_array_sorter_gym.md).
Its fresh hard-surface successor separates abstention from genuine tradeoffs and reduces
orientation-balanced pair predictions into five-candidate Pareto arrays; see
[`docs/ldt_array_hard_surface_v2.md`](docs/ldt_array_hard_surface_v2.md).

## Optional HRM-Text cycle milestone

The optional milestone asks whether recurrent update summaries rank and calibrate constitutional-verdict errors. It does
not train a model, and the DataGoblin validator remains the label authority. Check optional runtime availability without
loading weights or touching the network:

```bash
datagoblin hrm-status
datagoblin make-verdict-dataset --seed 7 --out runs/hrm-v1/dataset
datagoblin steward-sealed-labels --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --out ../datagoblin-hrm-v1-steward
datagoblin trace-hrm --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --out runs/hrm-v1/mock --mock
datagoblin evaluate-cycle-signal --run-dir runs/hrm-v1/mock \
  --sealed-labels ../datagoblin-hrm-v1-steward/sealed_labels.jsonl --calibration logistic
datagoblin trace-recurrent-baselines --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --sealed-labels ../datagoblin-hrm-v1-steward/sealed_labels.jsonl --out runs/hrm-v1/baselines
datagoblin compare-recurrent-baselines --primary-run runs/hrm-v1/mock \
  --baseline-root runs/hrm-v1/baselines --out runs/hrm-v1/recurrent_comparison.md
datagoblin role-ablation --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --out runs/hrm-v1/roles --mock
datagoblin hrm-preflight --local-model-path D:/models/HRM-Text-1B \
  --out runs/hrm-v1/preflight --ram-mb 4096 --cpu-percent 50 --io-mb-s 50
```

`--mock` validates the pipeline and is never evidence about HRM-Text. A real run requires the optional `hrm-text` extra,
an explicit local path or explicitly opted-in Hugging Face ID, and acknowledgement that the process is already under an
operator-provided hard resource-cap wrapper. Downloads are disabled unless `--allow-download` is passed. Full activations
are not persisted by default; checkpoint-provided Python additionally requires `--trust-remote-code`. `--device cpu` is
the default, while CUDA must be selected explicitly. The repository's WSL launcher supplies a reproducible one-example
hard-capped path; see [`docs/hrm_execution_safety.md`](docs/hrm_execution_safety.md) and
[`docs/hrm_text_milestone.md`](docs/hrm_text_milestone.md).

## Example recipe

The complete executable example is [`examples/recipe.good.yaml`](examples/recipe.good.yaml). Its essential split and authority boundary is:

```yaml
split_contract:
  train_seed_namespace: causal-switches/train/example
  evaluation_seed_namespace: causal-switches/eval/example
  held_out: true
reference_solver_requirements:
  required: true
  component: causal_switches_exact_solver
independent_verifier_requirements:
  required: true
  component: causal_switches_independent_verifier
builder_permissions:
  network: false
  subprocess: false
```

The deliberately invalid leakage example is `examples/recipe.invalid_eval_leakage.yaml`; validation returns a nonzero exit code and the `evaluation_leakage` rule.

## Upstream integration

`datagoblin discover-upstreams` checks environment variables, sibling paths, and `C:/projects` in the documented order. `configs/upstreams.example.yaml` selects include globs and excludes secrets, weights, checkpoints, caches, private material, and large generated trees. Ingestion is read-only with copy or symlink mode; symlink creation falls back to copying when unavailable. Neither upstream is a runtime dependency. Exact inspected commits and license cautions are in [`docs/upstream_inventory.md`](docs/upstream_inventory.md).

## Non-claims and roadmap

Version 0.1 does not establish constitutional alignment, autonomous scientific competence, safe arbitrary code generation, transfer to real game agents, superiority of HRM/TRM/LDT, effectiveness of a 1B specialist, or capability gains beyond the toy environment. See [`docs/non_claims.md`](docs/non_claims.md).

The path from the deterministic heuristic to a roughly 0.8B–1.3B specialist uses supervised research episodes, chosen/rejected constitutional pairs, violation-vector and outcome prediction, calibrated abstention, ledger retrieval, and held-out environment families. Model size alone is not a research capability. See [`docs/model_training_roadmap.md`](docs/model_training_roadmap.md).
