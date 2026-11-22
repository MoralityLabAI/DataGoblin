# MeTTa-05: bounded MOSES-style rule control for curation failures

**Research status:** original CPU genetic search. Upstream [MeTTa-MOSES](https://github.com/iCog-Labs-Dev/metta-moses) uses PeTTa/MeTTaLog and has independent licensing considerations. This experiment imports no upstream code and makes no native equivalence claim.

## Hypothesis

Pattern-mined features of a synthetic research corpus may support a compact interpretable rule predicting when an initial evidence selection violates the collection oracle. The fitness target is an **invalid initial selection**, independently computed by the existing bounded synthetic Python verifier. The resulting rule is advisory and cannot release source material or certify MeTTa semantics.

One observation is drawn per canonical training root from the existing [MeTTa-SFT](LOOPED_SFT.md) curriculum. All development, test and conflict OOD roots remain excluded. An internal hash split of the TRAIN roots supplies discovery and confirmation partitions. Evolution reads discovery labels only; confirmation results are calculated after selecting the program.

## Program grammar and evaluation

The finite grammar has constants, signed observable features, and conjunctions/disjunctions of two signed features. Each program has at most **three syntax nodes**. Genetic mutation and crossover are bounded by a fixed population size and at most 64 generations. Fitness is balanced accuracy penalized by program size. This is an original tiny symbolic control rather than full MOSES.

On the same internal confirmation roots, report a majority-class baseline chosen on discovery roots and a fixed cost-pressure threshold. Store confusion matrices, balanced accuracy, false-positive/false-negative rates, selected program size, number of evaluated programs and disjoint source-lineage digests. Keep each generation's best fitness to reveal selection effects.

Run from the repository root:

    python -m MeTTa.moses_control --roots-per-family 512 --seed 20261122 --population 40 --generations 12 --out runs/metta/moses_rule_control
    python -m pytest -q tests/unit/test_metta_moses_control.py

Outputs include a JSON receipt and an unverified MeTTa-shaped teaching expression for the chosen rule. The Python synthetic policy defines its semantics; native interpretation requires a separate replay. All examples retain `training_ready=false`. No weights are trained and zero additional SFT presentations are created.

## Next experimental lane

Feed the best rule into a shadow-only Scout/Alchemist selector that nominates counterfactual curation tickets. Verify any new world with the existing independent bounded oracle. Compare against equal-volume random interventions and preserve negative outcomes. Evaluation on genuinely fresh steward-held roots is necessary before a generalization claim. Keep the fully OOD conflict family excluded from training.

The evolved program must never be installed as an authority inside EtiquettePolicy or ResearchMembrane without independent verification and data-rights review.
