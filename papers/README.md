# DataGoblin research paper library

Curated AI research bibliography for Aleph's research-oriented pretraining and the associated GoblinBench gyms. Snapshot: 28 November 2026. The chronological library contains 181 paper references, with cross-indexes for recurrent models and their mechanisms. These are reading lists with links to primary sources; the same paper may appear in both an era and a specialty index. Each work should be counted once when used as a source.

## Chronological reading

| Era | Subject | References |
| --- | --- | ---: |
| [1982-2006](1982-2006/README.md) | Foundations: connectionism, learning and early deep networks | 12 |
| [2007-2016](2007-2016/README.md) | Deep learning revival: representations, optimization and adaptive computation | 23 |
| [2017-2019](2017-2019/README.md) | Attention, language-model pretraining and weight-tied depth | 18 |
| [2020-2022](2020-2022/README.md) | Scaling laws, efficiency, reasoning and circuits | 31 |
| [2023](2023/README.md) | Open models, agentic reasoning, formal methods and interpretability | 28 |
| [2024](2024/README.md) | Efficient architectures, test-time compute and research agents | 25 |
| [2025](2025/README.md) | Reasoning models, recurrent depth and AI-assisted discovery | 23 |
| [2026](2026/README.md) | Research frontier through 28 November 2026 | 21 |

## Specialized reading

| Index | Scope |
| --- | --- |
| [LoopedTransformers](LoopedTransformers/README.md) | 25 focused papers on shared weights, recursive reasoning, pretraining, routing and inference, plus extended survey links |
| [RecurrentMechinterp](RecurrentMechinterp/README.md) | 6 focal papers, adjacent interpretability methods, and our recurrent-mechanism manuscripts |

## Accompanying formal mathematics

- [DataGoblin formal-game catalog](../training/data/formal_game_catalog.json) and [owned corpus catalog](../training/data/owned_corpus_catalog.json): existing data sources, rights notes and verifier requirements.
- [MathCan](https://github.com/patrickdugan/MathCan), [EvalMath](https://github.com/MoralityLabAI/EvalMath), [mathlib4](https://github.com/leanprover-community/mathlib4), [LeanDojo-v2](https://github.com/lean-dojo/LeanDojo-v2): Lean statements, proof and tactic traces, and proof replay. See the era lists for LeanDojo, DeepSeek-Prover, STP, Goedel-Prover, AlphaProof and the 2026 formal-research literature. These are companion sources, not a claim that each AI paper has a corresponding verified Lean formalization.

## Expanding the collection

Use [OpenAlex](https://openalex.org/), [arXiv](https://arxiv.org/), [PMLR](https://proceedings.mlr.press/), [ACL Anthology](https://aclanthology.org/), and [OpenReview](https://openreview.net/) to locate original versions and citation families. Prioritize actual methods, derivations, controlled experiments, negative findings and independent verification. The looped-transformer survey linked in the specialty index is a larger discovery reservoir; the yearly index is curated, not comprehensive.

## Source custody

This directory contains only Markdown bibliographies and original reading notes. It does not contain copies of third-party PDFs, extracted text, model outputs, or code. A citation or open-access page alone is not evidence of redistribution or training permission. Check paper-level terms and retain bibliographic identity, source URL, version, and license before adding full text to any corpus. Use the canonical paper identifier across editions and cross-indexes to avoid duplicate training material. Preserve evaluation-only tasks and solutions outside training.
