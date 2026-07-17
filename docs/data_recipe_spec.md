# DataRecipe specification

`DataRecipe` v1 is the executable experiment proposal. It binds hypotheses and capabilities to an environment, generator, trajectory mixture, objectives, curriculum, split contract, transforms, solver, independent verifier, builder permissions, resources, controls, ablations, predictions, provenance, constitution version, and claim boundary.

Trajectory shares must sum to one within `1e-6`. Predicted outcomes are intervals with confidence and evidence rather than unsupported exact gains. Every executable task requires a solver, invariant certificate, or explicit impossibility/uncertainty label. Every accepted recipe has a falsifier and at least one control or ablation.

The split contract freezes metric and namespace definitions before inspection. Builder-facing fields may include the training namespace and public split contract; they may not contain evaluation seed values, expected answers, hidden metrics, evaluator source, or hidden mechanics.

The canonical human-readable example is `examples/recipe.good.yaml`, and the machine-readable contract is `schemas/data_recipe.schema.json`.

