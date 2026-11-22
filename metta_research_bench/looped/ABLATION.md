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

## Frozen results

The full good curator yielded 178 verified completions in 600 paired trials
versus 117 for the sloppy curator. The gain was 10.17 percentage points
(paired root bootstrap 95% interval 5.33 to 14.67 points). The gain was
positive for all three seeds. On the 32 fresh conflict roots, the controls
completed 28 and 24 of 96 trials respectively (4.17 points, interval −6.25
to 14.58).

| Training curator | Changed selections / 240 | In-family complete / 600 | Gap from good, pp (95% interval) | In-family Holm p | Conflict complete / 96 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Good | 0 | 178 | | | 28 |
| Sloppy | 133 | 117 | 10.17 (5.33 to 14.67) | | 24 |
| Drop closure | 0 | 178 | Identical training data | | 28 |
| Drop eligibility | 40 | 155 | 3.83 (0.17 to 7.83) | Shared guard | 28 |
| Drop budget | 118 | 172 | 1.00 (−3.17 to 5.17) | Shared guard | 17 |
| Drop unique lineage | 18 | 157 | 3.50 (−0.50 to 7.50) | 0.3844 | 23 |
| Drop current revision | 27 | 161 | 2.83 (−1.33 to 6.83) | 0.6027 | 27 |
| Drop dependencies | 25 | 174 | 0.67 (−3.00 to 4.50) | 0.9674 | 25 |
| Drop conflict guard | 0 | 178 | Identical training data | | 28 |
| Rank by repeated mentions | 71 | 169 | 1.50 (−2.17 to 5.00) | 0.9674 | 22 |

Unique lineage had the largest overall estimate among the rules that
differed between the controls. The revision guard had the largest drop
within its own family, from 70 to 48 of 120, alongside an increase from
32 to 44 in the dependency family. These family counts are descriptive.
No identifiable rule passed the preregistered Holm correction on the
primary split or on conflict. The overall control gap reproduced while
the contribution of any single differing rule remained uncertain.

Good had 405 premature stops and 2,401 policy verifier calls versus 440
and 2,920 for sloppy in-family. Dropping the shared budget guard reduced
premature stops to 359 while raising calls to 3,655 and lowering conflict
completions from 28 to 17. All frozen arms used the same inference verifier
and made zero unsafe additions. Identical selections for closure and
conflict leave their effects unidentifiable in this corpus.

The run at `a1f7496bc1b767c8d5d850180e5de6bf2e350798` audited 240 training
roots and 10,662 candidates, trained 18 ablated fits, and reused six
verified control checkpoints. The audit and training took 252.47 seconds.
Peak RSS was 871.51 MiB in the recycled native audit workers, 391.95 MiB in
the training process, and 473.69 MiB in frozen evaluation. The receipt
audit verified 4,800 in-family and 768 conflict trials against the Prime
Verifiers traces and confirmed disjoint source hashes from 90 archived
conflict roots across prior studies.

- Training lock SHA-256 `7cb93a0dda84678e1cae0eb2c5c6895cc0cfedb7a11608e622c729a74ee20384`
- Frozen report SHA-256 `c087bebc2771652116474cce2148e62d1ed23f746479f63c5d8c408fc984538c`
