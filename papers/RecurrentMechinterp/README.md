# RecurrentMechinterp

A deliberately short index of research on the internal computations of weight-tied or recurrent-depth models. The larger general interpretability literature lives in the chronological folders.

| Year | Paper | Focus |
| --- | --- | --- |
| 2025 | [Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer](https://arxiv.org/abs/2507.02199) | Tests whether Huginn's iterates are decodable as reasoning |
| 2025 | [Two-Scale Latent Dynamics for Recurrent-Depth Transformers](https://arxiv.org/abs/2509.23314) | Step geometry and stopping |
| 2026 | [A Mechanistic Analysis of Looped Reasoning Language Models](https://arxiv.org/abs/2604.11791) | Layer-specific cyclic fixed points and repeated computation |
| 2026 | [Understanding Dynamic Compute Allocation in Recurrent Transformers](https://arxiv.org/abs/2602.08864) | Per-token allocation and generalization |
| 2026 | [From Growing to Looping: A Unified View of Iterative Computation in LLMs](https://proceedings.mlr.press/v306/kapl26a.html) | Depth-wise signatures and retrofitting |
| 2026 | [Beyond the Training Horizon: Mechanisms and Limits of Length Generalization in Looped Transformers](https://arxiv.org/abs/2609.33144) | Causal progress cues, state propagation and extrapolation |

## Closely related methods

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) and [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html): feature and circuit foundations.
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944): reusable feature representations.
- [Is This the Subspace You Are Looking for?](https://arxiv.org/abs/2311.17030): controls for activation-patching claims.
- [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html): graph-based mechanism attribution.

## Our work (research manuscripts and experimental reports)

- [Shared weights / dark fibers](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/papers/shared_weights_dark_fibers): shared parameters versus visit-specific causal roles.
- [Recurrent mechanistic interpretability manuscript](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/papers/rmp_looped_mechinterp): tied dictionaries and recurrent operation surrogates.
- [LoopedTransformers mechinterp experiments](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/mechinterp): interventions, latent transport, fixed points and Jacobian diagnostics.
- [BPD](https://github.com/MoralityLabAI/BPD): parameter decomposition and intervention research.

Reading questions: which features persist across visits? Which operations change causal roles with recurrent depth? How do measured predictions compare with interventions and matched controls? Treat in-house manuscripts as work in progress unless their source explicitly establishes otherwise.

[LoopedTransformers](../LoopedTransformers/README.md) · [Library](../README.md)
