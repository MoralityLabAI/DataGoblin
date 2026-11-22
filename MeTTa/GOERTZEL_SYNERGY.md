# MeTTa-03: Pattern-Miner-style curation and the wider Goertzel stack

**Status:** original CPU research control on top of [MeTTa-SFT-02](LOOPED_SFT.md). All generated sources have `training_ready=false`; proposed motifs have advisory authority only. The miner runs deterministic Python and does not claim to execute the external Hyperon Pattern Miner, MeTTa-MOSES, NARS or MetaMo packages.

## Why begin with the miner

The existing curriculum has 3,072 synthetic roots, including 1,960 training roots. Its correction traces teach what action to take. The research opportunity is to identify **which kinds of initial curation states systematically need correction** and propose cases that disambiguate their causes. The upstream Hyperon Pattern Miner searches frequent and surprising graph structures, with optional partition-based surprisingness. Its active PeTTa implementation is [iCog-Labs-Dev/hyperon-miner](https://github.com/iCog-Labs-Dev/hyperon-miner), AGPL-3.0. This prototype is independent original code, imports no upstream miner source and writes no third-party source snapshots.

[MeTTa/pattern_miner.py](pattern_miner.py) mines bounded one-/two-feature conjunctions from the **public** initial states of existing `train_candidate` roots. Features cover budget pressure, source-pool lineage/key collisions, prerequisite density, selected-count, topic diversity and source revision structure. The outcome is a separately computed bounded Python-oracle validity result.

A single root contributes once, irrespective of its up-to-four SFT presentations. Discovery roots nominate motifs; a disjoint internal subset of **training** roots supplies diagnostic confirmation. Pair surprisingness compares observed frequency with independent marginals. Proposed ranks use only discovery enrichment of invalid initial states. Candidate ranking and confirmation statistics are exploratory, with no causal authority.

The script produces `motifs.json` with root-stamp digests and `motifs.metta` with advisory atoms. They are **new views of existing train roots** and create **zero additional SFT presentations**.

    python -m MeTTa.pattern_miner --roots-per-family 512 --min-support 20 --top-k 24 --out runs/metta/pattern_miner
    pytest -q tests/unit/test_metta_pattern_miner.py

## Goertzel components to incorporate next

| Component and source | Concrete DataGoblin/looped-transformer use | Gate |
| --- | --- | --- |
| [Hyperon Pattern Miner](https://github.com/iCog-Labs-Dev/hyperon-miner) | Turn recurring failed-episode motifs into counterfactual research tickets for Scout and Alchemist | Root-level non-overlap, independent verifier of synthesized worlds |
| [MeTTa-MOSES](https://github.com/iCog-Labs-Dev/metta-moses) | Evolve compact gate/repair rules on mined features, then distil verified actions to a small recurrent student | Exact synthetic-oracle score, program-length cap, fresh task-family OOD, deterministic-policy control |
| [PLN and NARS](https://hyperon.dev/primer) | Attach evidence-source stamps to curation claims, distinguish strength and confidence, and reject correlated evidence reuse | Independent lineage checks; calibration against adjudicated outcomes only |
| [AIRIS / causal induction](https://wiki.hyperon.dev/Glossary) | Search for interventions separating competing explanations of failed edits | Explicit paired do-action, independent verification and no hidden evaluator labels |
| [MetaMo](https://github.com/iCog-Labs-Dev/MetaMo) and [OpenPsi](https://github.com/iCog-Labs-Dev/hyperon-openpsi) | Budget research effort among coverage repair, novelty, computation and source-risk constraints | Homeostatic controller proposes requests only; DataGoblin membrane retains release authority |
| [ECAN/MeTTa attention](https://github.com/iCog-Labs-Dev/metta-attention) | Token-rent and source-handle prioritization in Pocket Goblin; prototyped in HybridTRMLDT MCP-05 | Exact upstream token counts, repeated-read metrics and matched native task scores |
| [HiBaCaML](https://github.com/iCog-Labs-Dev/HiBaCaML) | Shared vs task-specific recurrent expert columns; test whether isolated columns preserve prior skill | Forgetting and matched-compute sweeps on held-out source families |

The upstream MeTTa-MOSES and Pattern Miner implementations use **PeTTa/MeTTaLog** while DataGoblin's existing native query replay uses `hyperon==0.2.10`. A Python motif report or native `match` result does not establish that other runtimes support the same functions or produce equal results. Runtime compatibility needs a separate experiment.

## Worklist and evaluation protocol

1. **Propose:** use motifs with discovery/confirmation support as candidate `BuilderTickets`, retaining root stamps and synthetic labels. Filter near-duplicate objectives by canonical feature tuples. Pass tickets through existing `ResearchMembrane`. Private/evaluation roots stay absent.
2. **Intervene:** synthesize paired worlds that toggle one proposed causal feature. Have the environment-owned oracle check both outcomes. Preserve failed interventions, too. This is where AIRIS-style learning becomes testable.
3. **Compress:** constrain a MOSES-style rule search to a small auditable grammar and evaluate it against a deterministic baseline on separate train-family roots. Record verifier error, program size, CPU time and downstream loop counts.
4. **Distil:** train a small recurrent student on approved synthetic paired traces under the existing four-presentations-per-root accounting. Measure test-family transfer, premature STOP, invalid admission, token cost, calibration and variation across seeds.

A mined motif's predictive lift is an exploratory association in a synthetic corpus. Native MeTTa checks establish matched query bindings; the separate Python oracle defines the admissibility objective. The next hard evaluation is whether motif-guided counterfactual data improves a trained looped model against a matched quantity of uniform synthetic data on untouched source families.

**Integrity:** original project code only. No AGPL miner code copied. No upstream content downloaded automatically, no reserved test labels accessed, and no promotion to `VERIFIED_TRAIN`.
