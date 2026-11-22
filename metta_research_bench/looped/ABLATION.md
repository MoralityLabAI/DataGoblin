# One-rule MeTTa ablations

Each ablation changes one clause of the original good curator. The closure,
eligibility, budget, lineage, revision, dependency, and conflict variants
drop the named guard. The coverage variant ranks evidence mentions in place
of distinct goals. The other eligibility predicates, rank fields, canonical
tie-break, and one-rollout-per-root quota stay fixed.

The original 10,662 candidates and all 240 training roots come from the
locked benchmark. The runner checks both original selection vectors against
the prior receipts and independently replays every ablated selection.
Equivalent selection vectors share the same deterministic fit. A rule that
selects identical examples is unidentifiable in this corpus.

The good and sloppy controls use their original hash-verified checkpoints.
Every distinct ablation trains the same three paired initializations, the
44,417-parameter pinned Looped Transformer, and the original 600 AdamW
updates with 28,800 supervised tokens per fit. The runner checks seed
initialization and budget fields against the original controls. The
three-pass policy, typed membrane, and independent Prime Verifiers grader
stay fixed.

The primary comparison is good minus each ablation on 200 new in-family
roots. The separate transfer split uses 32 new conflict roots. Each root is
paired over all three seeds. The lock excludes the earlier study's training,
development, primary, and transfer sources and checks that no new source
hash overlaps. The registered seed ranges are disjoint from the earlier
MeTTa conflict reserves.

Root-clustered intervals describe each contrast. A two-sided root-level
sign-flip test with Holm correction covers distinct comparisons for rules
that differed between the original curators. Closure, eligibility, and
budget were shared by the original rules, so their ablations are
diagnostics. These are conditional single-rule effects within the good
curator. They cannot be added to reconstruct the full difference.

## Run locally

Set up the same Linux or WSL environment and pinned LoopedTransformers
checkout as in the [paired benchmark](README.md). Use the original run
directory, which contains its locked development and frozen reports, and
its scratch checkpoint directory.

    python -m metta_research_bench.looped.ablation train \
      --reference runs/metta-looped --reference-checkpoints /tmp/metta-looped-weights \
      --out runs/metta-looped-ablation --checkpoints /tmp/metta-looped-ablation-weights
    python -m metta_research_bench.looped.ablation frozen \
      --reference runs/metta-looped --reference-checkpoints /tmp/metta-looped-weights \
      --out runs/metta-looped-ablation --checkpoints /tmp/metta-looped-ablation-weights

The frozen command requires the committed source, unchanged rule and
protocol hashes, and every checkpoint hash from the training lock.
Keep the generated traces and all checkpoint directories outside Git.
