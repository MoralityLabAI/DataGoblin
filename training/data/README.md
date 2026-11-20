# Aleph data planning

The [19 November 2026 landscape report](../reports/2026-11-19_data_landscape.md) evaluates the owned corpus receipts and public sources for a proposed 20B-token run with a conditional 30B continuation.

| File | Contents |
| --- | --- |
| [public_pretraining_catalog.json](public_pretraining_catalog.json) | 24 public pretraining and solution sources, including access, ancestry and license notes. |
| [formal_game_catalog.json](formal_game_catalog.json) | 16 formal-math/game datasets, repositories and generator families. |
| [owned_corpus_catalog.json](owned_corpus_catalog.json) | Eight owned lanes, four pinned Lean source inventories and 13 primary receipt links. |
| [aleph_20b_30b_mix.json](aleph_20b_30b_mix.json) | Proposed quotas, repeat limits, fallback rules, objectives and accounting requirements. |
| [aleph_20b_30b_estimates.json](../estimates/aleph_20b_30b_estimates.json) | Reproducible analytical compute cases for 10B, 20B and 30B tokens. |

## Meaning of the records

These files are planning metadata. `training_ready` remains false. Public source sizes use the publishers' counting methods; owned receipts also use several different tokenizers and serialization formats. Aleph's tokenizer revision and accepted unique counts remain unresolved.

Source cards were read on the audit date. Pin data-file revisions at intake and retain original document, problem and episode IDs. The corpus export should contain achieved quotas after filtering, repeat caps and shortfall substitution. Keep the desired quotas beside those achieved values.

The proposed static-root cap is four presentations total across all views of a root. Up to four distinct verified views can share that allowance. Preserve original split custody during extraction, grouping all descendants of a held-out root with that root.

## Objective integration

Reference text and source code need a declared language-modeling objective. Checked tasks use response supervision with claim-specific receipts. The proposed pretraining adapter must represent provenance and split eligibility separately from truth or outcome verification. This planning update leaves the current [`data_etiquette.yaml`](../../configs/data_etiquette.yaml) policy unchanged.

Research predictions and model critiques remain attributed proposals. A proof checker, executed test suite or environment replay can support a narrower outcome claim. Admit each record with the claim type and receipt that establish its allowed supervision.

## Generation measurement

The proposed first yield measurement is 10,000 examples across 12–20 task families. Record accepted unique Aleph tokens, loss-bearing tokens, canonical roots, replay pass rate, duplication and resource cost. This measurement has not run.

Store raw training content, generated runs and evaluation labels outside this directory. The repository retains catalogs, plans and aggregate analytical estimates.
