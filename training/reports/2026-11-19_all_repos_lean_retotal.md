# Complete Lean repository inventory, 19 November 2026

Scope: **all 36 repositories returned by GitHub organization search for MoralityLabAI and all 188 returned for patrickdugan**, scanned on their default-branch `HEAD` using the recursive Git tree endpoint. Every nonempty returned tree reported `truncated=false`. Two personal repositories (`Aleph`, `my-near-rust-contract`) returned GitHub HTTP 409 because their Git histories are empty.

Count unit: tracked file whose path ends in `.lean`, including test fixtures, generated proofs, certificate instances and source copied from elsewhere. Bytes are Git tree uncompressed blob sizes. This is a complete **default-branch path/byte inventory** for the enumerated repositories, not a count of verified proof declarations or accepted, deduplicated training tokens. Other refs, local unpushed files and content-addressed external archives remain outside scope.

## MoralityLabAI

| Repository | Files | Bytes |
|---|---:|---:|
| [EvalMath](https://github.com/MoralityLabAI/EvalMath) | 34 | 802,097 |
| [LoopedTransformers](https://github.com/MoralityLabAI/LoopedTransformers) | 93 | 187,772 |
| [BPD](https://github.com/MoralityLabAI/BPD) | 20 | 48,213 |
| [HybridTRMLDT](https://github.com/MoralityLabAI/HybridTRMLDT) | 10 | 17,646 |
| [complicency-bench](https://github.com/MoralityLabAI/complicency-bench) | 2 | 9,416 |
| [Compactification-Theory](https://github.com/MoralityLabAI/Compactification-Theory) | 4 | 7,362 |
| **Total** | **163** | **1,072,506** |

The remaining **30 organization repositories** had zero `.lean` files in their default-branch tree. The former inventory included only EvalMath from this organization.

## patrickdugan

| Repository | Files | Bytes |
|---|---:|---:|
| [OntologicalProofs](https://github.com/patrickdugan/OntologicalProofs) | 399 | 2,176,267 |
| [leanp2p](https://github.com/patrickdugan/leanp2p) | 236 | 2,639,097 |
| [semaev-index-calculus](https://github.com/patrickdugan/semaev-index-calculus) | 124 | 77,074,795 |
| [MathCan](https://github.com/patrickdugan/MathCan) | 122 | 560,412 |
| [btc-verified](https://github.com/patrickdugan/btc-verified) | 79 | 495,226 |
| [neoplatonic-lean](https://github.com/patrickdugan/neoplatonic-lean) | 21 | 83,118 |
| [TradeLayer-Rust](https://github.com/patrickdugan/TradeLayer-Rust) | 17 | 301,949 |
| [bitcoin-rs](https://github.com/patrickdugan/bitcoin-rs) | 16 | 81,643 |
| [beyond-bayes](https://github.com/patrickdugan/beyond-bayes) | 10 | 31,564 |
| [bitcoin-liquidity-formal](https://github.com/patrickdugan/bitcoin-liquidity-formal) | 7 | 21,812 |
| [UTXO-Ref](https://github.com/patrickdugan/UTXO-Ref) | 7 | 277,212 |
| [tensor-lab](https://github.com/patrickdugan/tensor-lab) | 7 | 37,624 |
| [BitAgent](https://github.com/patrickdugan/BitAgent) | 7 | 18,532 |
| [SystemicRiskBench](https://github.com/patrickdugan/SystemicRiskBench) | 3 | 15,295 |
| [cryptanalytic-discovery-lab](https://github.com/patrickdugan/cryptanalytic-discovery-lab) | 1 | 2,119 |
| **Total** | **1,056** | **83,816,665** |

The other **173 personal repositories** contained no tracked `.lean` files in the scanned default branch; two of these are empty Git histories.

## Source organization and proof relevance

A directory-level inspection confirms that file counts cover different types of material:

- **EvalMath:** `lean/FeedbackChannel` has 22 files / 620,535 bytes; `fixtures/bpd_lean_curriculum_v2` has 6 files / 130,108 bytes; `fixtures/external` has 4 files / 38,034 bytes. Some generated fixtures may repeat proofs held in other repositories.
- **LoopedTransformers:** `experiments/lean_ram03` has 67 files / 115,123 bytes and `experiments/lean_ram02` has 12 / 23,315 bytes. Research and activation-funnel experiment fixtures account for other files. Whole-repository file counts should not be construed as 93 independent theorems.
- **BPD:** all 20 files / 48,213 bytes are under `mechinterp/bpd`.
- **HybridTRMLDT:** all 10 files / 17,646 bytes are under `formal/CCA`.
- **OntologicalProofs:** `Neoplatonic/LTR` holds 382 files / 2,111,181 bytes.
- **leanp2p:** 81 files / 820,403 bytes under `packages/libp2p`; the top-level `Tests` directory contains 34 files / 769,618 bytes, plus further nested test files. The repository combines implementation and test proofs.
- **btc-verified:** consensus, serialization, transaction, crypto and chain-state modules appear alongside test and fuzz files.
- **TradeLayer-Rust:** one `formal/` Lean file contributes 219,438 bytes, with 16 other files under `formal/bitcoin_core_36629`.

The path breakdown supports corpus-lane assignment. It does not establish which declarations are kernel checked, how many `sorry` terms remain, or whether generated files share theorem ancestry.

## Corrected totals

| Measure | Result |
|---|---:|
| Repositories enumerated | 224 |
| With Lean files | 21 |
| Lean source files | **1,219** |
| Source bytes | **84,889,171** |
| Files larger than 1,000,000 bytes | **7** |
| Bytes in seven Semaev certificate files | **74,515,825** |
| Source files excluding those seven | **1,212** |
| Source bytes excluding those seven | **10,373,346** |

The previous four-repository inventory reported **679 files and 80,613,571 bytes**. Scanning all enumerated repositories adds **540 files and 4,275,600 bytes**. Among normal-sized source, the expansion is proportionally much larger because the seven Semaev files dominated the earlier byte total.

### Training accounting

The current tree scan cannot establish a token total without a frozen Aleph tokenizer and a content/split audit. In particular, large Semaev certificate blobs should be counted apart from theorem statements, tactic traces, and proof-repair examples. Repositories may contain imported upstream code, duplicate lemmas, generated test fixtures or held-out material. A valid DataGoblin corpus intake must deduplicate by source/statement ancestry, preserve reserved test sets and run Lean kernel checks against pinned toolchains.

Source eligibility categories to derive next: (a) human-authored proof declarations, (b) generated verified proofs, (c) tactic-state traces, (d) checked negative refutations, (e) certificate payloads, (f) source fixtures/tests, and (g) evaluation-only or unverified material. Report unique roots, verified train-eligible proofs, actual Aleph tokens and repeated exposures separately.

This report supersedes the four-repository Lean inventory in `training/reports/2026-11-19_data_landscape.md` and `training/data/owned_corpus_catalog.json`.
