# MeTTa-AIRIS-01: paired counterfactual synthetic curation worlds

**Status:** original AIRIS-inspired, bounded CPU research gym. Produces research-only paired receipts; it runs neither native AIRIS nor native MeTTa inference and gives no authorization to admit data.

## Purpose

The existing Pattern Miner ranks observable motifs associated with failed initial synthetic evidence selections. The MOSES-style control predicts invalidity. MeTTa-AIRIS-01 asks a narrower, causal question: **Does changing one world fact cause a change in the collection verifier's recorded constraint violation or canonical next repair?** Every original and counterfactual world is regenerated from the existing synthetic curation task generator and checked by its bounded exact Python oracle.

One root supplies three counterfactual attempts against **the same initial state**. The motif-guided selector uses only public features and discovery-ranked motifs; a seeded uniform selector samples among the same root's admissible intervention mechanisms. A third pre-registered control chooses the **token-budget intervention whenever available**, exposing how easily a deterministic threshold-crossing mutation can flip its targeted constraint. The verifier runs after all three interventions have been frozen.

A resulting change is an intervention on a toy deterministic world. It is not evidence that a similar causal relation exists in a real document corpus, agent rollout, or external graph.

## Interventions

| One public fact changed | Candidate mechanism | Independent target check |
| --- | --- | --- |
| Token budget scalar | Cross the selected set's cost boundary | Selected cost exceeds budget |
| Selected record's eligibility Boolean | Flip one eligibility flag | An ineligible selected record exists |
| Selected record's lineage ID | Assign another selected record's lineage | Selected lineages collide |
| Selected record's prerequisite list | Add/remove one dependency edge to an earlier record | A required predecessor is missing |
| Eligible competing record's epoch | Make a newer competing version observable | A selected revision is stale |

The chooser reads the task's public fact store and selected IDs. Every proposed world is a deep copy with **exactly one structured fact change**, followed by regeneration and validation of the MeTTa fact string. A separate direct target predicate must agree with the existing collection verifier on the presence of the affected constraint before and after intervention.

Some changes fail to alter the target violation because another fact sustains the original failure. Those negative results remain in the study. Every root also receives an identity-sham replay.

## Selection firewall

Only the old synthetic task families lineage, coverage, dependency, revision and budget are eligible. The conflict family is fully OOD. Every row is checked against the original immutable root-hash train/dev/test split. Within those existing **train_candidate** roots, a second deterministic split yields approximately 70% discovery and 30% internal confirmation.

During motif ranking, the discovery world's exact verifier label is available. Internal confirmation labels are explicitly masked and unused for choosing rules. A stratified and hash-ordered slice of 96 internal confirmation roots is frozen before outcome evaluation. The selected intervention and uniform intervention are compared on identical source roots. The root and both modified worlds receive SHA-256 receipts.

This internal confirmation is a split of known synthetic training roots and **not a new sealed benchmark**. No new source root, SFT presentation or preference pair is released by this experiment.

## Outcome contract

Report the target-violation flip rate for each arm, full violation-signature change rate, changed canonical oracle next edit, mutation type distribution, exact-source hashes, sham outcomes, and matched-root wins/ties. The budget-first control intentionally crosses the selected-cost threshold whenever possible and may attain 100% targeted flips by construction. A higher flip rate alone has little meaning in that setting. The meaningful downstream question is whether motif guidance finds varied and independently useful oracle-repair changes beyond this deterministic threshold shortcut. None of these metrics establishes model improvement.

The experiment keeps independent claims for:
- valid single-fact mutation and synthetic-world replay;
- the environment-owned Python oracle's constraint and optimal-subset verdict;
- an AIRIS-inspired selection heuristic over Pattern Miner motifs;
- eventual native runtime comparison and learned-model transfer.

The first two can pass under CPU tests. The latter two require an external native or trained-model experiment.

## Reproduce and audit

    python -m MeTTa.airis_counterfactual --roots-per-family 512 --cases 96 --seed 20261122 --min-support 20 --top-k 24 --out runs/metta/airis_counterfactual
    python -m MeTTa.airis_receipts --bundle runs/metta/airis_counterfactual --out runs/metta/airis_counterfactual/audit.json
    python -m pytest -q tests/unit/test_metta_airis_counterfactual.py tests/unit/test_metta_airis_receipts.py

The auditor verifies the receipts' SHA-256, root split, paired control identity and exact source-root replay. A fast hash/custody check is available with the audit command's --fast flag. GitHub Actions archives the complete synthetic diagnostic receipt, marked internal_confirmation_only. It should never be copied into the canonical SFT or preference export.

## Next phase

Feed successful **discovery-only** interventions to a shadow Scout/Alchemist ticket adapter with registered verifier and world-source hashes. The ResearchMembrane remains the sole admission boundary, and a new ticket requires a separate source-rights review. Compare genuinely new motif-guided counterfactual worlds against a matched volume of uniformly generated worlds when training a compact recurrent model. Score retained evidence validity, completed curation tasks, premature STOP, compute cost and transfer to a fresh steward-held root pool.

Implementation ticket: [DataGoblin issue #17](https://github.com/MoralityLabAI/DataGoblin/issues/17). Goertzel source inspirations: [Hyperon Pattern Miner](https://github.com/iCog-Labs-Dev/hyperon-miner), [MeTTa-MOSES](https://github.com/iCog-Labs-Dev/metta-moses), and [MetaMo](https://github.com/iCog-Labs-Dev/MetaMo). This experiment uses original code and vendors none of their implementations.
