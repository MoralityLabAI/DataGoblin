# EvalMath AIRIS-07 candidate intake

Upstream: MoralityLabAI/EvalMath commit [3ef134f](https://github.com/MoralityLabAI/EvalMath/commit/3ef134f0d880a81ad56592487d951ac2287237d5).
The upstream exact CPU run [45285351276](https://github.com/MoralityLabAI/EvalMath/actions/runs/45285351276) passed all tests and deterministic replay. The training artifact contains 856 synthetic presentations, 428 roots and 214 observation groups. Its SHA-256 is pinned by this adapter.

This namespace is separate from DataGoblin's existing MeTTa AIRIS curation gym. The input histories expose action schedules, observed bits and declared drift rate. Teacher values supervise the next experiment, not hidden mechanism labels. Two chronological permutations share a stationary control.

`python -m MeTTa.evalmath_airis07_intake /path/to/training-bundle`

The adapter reads only train_manifest.json and train_candidates.jsonl. It checks exact schema, pin, split, canonical group/row identity and paired controls. It independently reconstructs every teacher value by enumerating raw hidden-driver paths and performing rational chronological filtering, without importing the upstream oracle. The candidate-only result grants no training approval, native MeTTa semantics, model validation or generalization claim.

The integration workflow checks out the pinned upstream revision in a separate directory, generates the deterministic fixture, copies only the two training files into a dedicated input directory, and invokes the read-only validator. The development artifact is never passed to the adapter. Research code and public development data are not sealed; prospective evaluation requires separately stewarded families and roots. Any later training use requires normal DataGoblin review/admission.
