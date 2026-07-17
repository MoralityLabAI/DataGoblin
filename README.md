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
```

## Optional HRM-Text cycle milestone

The optional milestone asks whether recurrent update summaries rank and calibrate constitutional-verdict errors. It does
not train a model, and the DataGoblin validator remains the label authority. Check optional runtime availability without
loading weights or touching the network:

```bash
datagoblin hrm-status
datagoblin make-verdict-dataset --seed 7 --out runs/hrm-v1/dataset
datagoblin trace-hrm --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --out runs/hrm-v1/mock --mock
datagoblin evaluate-cycle-signal --run-dir runs/hrm-v1/mock --calibration logistic
datagoblin role-ablation --dataset-manifest runs/hrm-v1/dataset/dataset_manifest.json \
  --out runs/hrm-v1/roles --mock
```

`--mock` validates the pipeline and is never evidence about HRM-Text. A real run requires the optional `hrm-text` extra,
an explicit local path or explicitly opted-in Hugging Face ID, and acknowledgement that the process is already under an
operator-provided hard resource-cap wrapper. Downloads are disabled unless `--allow-download` is passed. Full activations
are not persisted by default; checkpoint-provided Python additionally requires `--trust-remote-code`. See
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
