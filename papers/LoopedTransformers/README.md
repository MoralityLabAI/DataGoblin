# LoopedTransformers

Priority reading for Aleph and the HybridTRMLDT research program. Separate repeated latent computation over the same token sequence from temporal recurrence, explicit generated reasoning, and untied dynamic depth. The related methods below are useful comparisons even when they use different mechanisms.

| Year | Paper | Research theme |
| --- | --- | --- |
| 2016 | [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) | Halting and ponder cost |
| 2018 | [Universal Transformers](https://arxiv.org/abs/1807.03819) | Shared depth and per-position stopping |
| 2019 | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) | Cross-layer parameter sharing |
| 2019 | [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) | Fixed-point implicit depth |
| 2023 | [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196) | Constructive computational theory |
| 2024 | [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models](https://arxiv.org/abs/2404.02258) | Dynamic depth without requiring tied recurrence |
| 2024 | [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) | Coconut, adjacent continuous-latent reasoning |
| 2025 | [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach](https://arxiv.org/abs/2502.05171) | Huginn and prelude/core/coda |
| 2025 | [Reasoning with Latent Thoughts: On the Power of Looped Transformers](https://arxiv.org/abs/2502.17416) | Theory and controlled reasoning |
| 2025 | [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734) | Hierarchical state and updates |
| 2025 | [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871) | Tiny recursive reasoning |
| 2025 | [Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/abs/2507.10524) | Routing and shared KV |
| 2025 | [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741) | Ouro, pretraining and depth allocation |
| 2026 | [Tiny Recursive Reasoning with Mamba-2 Attention Hybrid](https://arxiv.org/abs/2602.12078) | SSM recurrence |
| 2026 | [LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation](https://arxiv.org/abs/2602.11451) | Variable-length trajectories |
| 2026 | [Think-at-Hard: Dynamic Looped Transformers for Improved Reasoning](https://proceedings.mlr.press/v306/fu26t.html) | Token-level selective iteration |
| 2026 | [From Growing to Looping: A Unified View of Iterative Computation in LLMs](https://proceedings.mlr.press/v306/kapl26a.html) | Depth growth, loop retrofitting |
| 2026 | [Tiny Autoregressive Recursive Models](https://arxiv.org/abs/2603.08082) | Matched-work comparisons |
| 2026 | [Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers](https://arxiv.org/abs/2604.07822) | Generalization and overthinking |
| 2026 | [Attention Residuals](https://arxiv.org/abs/2603.15031) | Depth aggregation; precursor to ARLoop |
| 2026 | [Closing the Loop: Practical Training Recipes for Looped Language Models](https://arxiv.org/abs/2610.00673) | Practical training and conversion |
| 2026 | [ARLoop: Fully-Open Pre-trained Looped Transformers with Attention Residuals](https://arxiv.org/abs/2611.05234) | Recurrent-depth residual design |
| 2026 | [Recurrent Looped Transformer](https://arxiv.org/abs/2610.07591) | Token-to-token recurrence; distinguish temporal from depth recurrence |
| 2026 | [Trading Depth for Time in Recurrent Transformers](https://www.microsoft.com/en-us/research/publication/trading-depth-for-time-in-recurrent-transformers/) | Temporal thought tokens versus physical depth |
| 2026 | [Looped Models are Fast Multi-Token Predictors (Floodgates)](https://arxiv.org/abs/2611.33896) | Shallow parallel drafts and deep verification |

## Broader bibliographies

- [Looped Language Models survey and claims ledger](https://github.com/ArminAzizi98/looped-lm-survey), an October 2026 snapshot of 557 papers and 1,211 claims. Use its categories to expand architecture, theory, optimization, cache, exit, and stability reading.
- [Awesome Looped Transformers](https://github.com/HJSang/awesome-looped-transformers), annotated discovery index.
- [Awesome Tiny Recursive Models](https://github.com/kuoyang1999/awesome-tiny-recursive-models), adjacent TRM and HRM literature.
- [RecurrentDepth research library](https://recurrentdepth.org/research).

## Research companions

- [MoralityLabAI/LoopedTransformers papers](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/papers), including [Floodgates](https://github.com/MoralityLabAI/LoopedTransformers/tree/main/papers/floodgates).
- [HybridTRMLDT](https://github.com/MoralityLabAI/HybridTRMLDT): recursive skills, routing, and native gym experiments.
- [RecurrentMechinterp](../RecurrentMechinterp/README.md): causal mechanisms and diagnostics.

Use original papers for claims. Keep publication versions, model checkpoints, budgets and comparison regimes explicit. This is a link library; no full-text licensing or training admission is asserted.
