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

## Lineage × revision follow-up

The factorial comparison holds every other clause fixed and scores the
full curator, the two single-rule omissions, and the joint omission on
the same fresh worlds. The existing nine control checkpoints are reused
after their selections, weights, initialization, and budgets are checked.
Three joint-omission fits use the same 10,662 candidates, 240 training
roots, initialization seeds, 44,417 parameters, 600 updates, and 28,800
supervised action tokens per fit. No conflict roots are used in training.

The primary estimate on the verified completion-rate scale is joint
omission minus the sum of the two single omissions plus the full curator.
A positive value means the joint omission completes more repairs than
predicted by adding the single-omission changes. The analysis pairs all
four arms within each seed and root, averages across seeds, then uses a
root bootstrap interval and a two-sided root-level sign-flip test. The
primary split uses 200 fresh in-family roots and the separate transfer
split uses 32 fresh conflict roots. Neither split overlaps the earlier
paired benchmark or one-rule ablation reserves.

Run `python -m metta_research_bench.looped.interaction train` followed by
the same command with `frozen`. Both stages use `--reference` and
`--reference-checkpoints` for the original paired benchmark,
`--prior-ablation` and `--prior-checkpoints` for the one-rule run, and
`--out` and `--checkpoints` for the new run. The fresh split is opened only
after all twelve checkpoint hashes are locked. Keep generated receipts
and checkpoints outside Git.

### Frozen interaction results

| Training curator | Changed selections / 240 | In-family complete / 600 | Premature stops | Policy verifier calls | Recurrent passes | Conflict complete / 96 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 0 | 172 | 414 | 2,348 | 8,289 | 28 |
| Drop unique lineage | 18 | 173 | 398 | 2,532 | 8,430 | 23 |
| Drop current revision | 27 | 182 | 389 | 2,460 | 8,070 | 24 |
| Drop both | 45 | 181 | 377 | 2,801 | 8,649 | 22 |

The single omissions predicted 183 in-family completions under an
additive model. Joint omission yielded 181. The interaction estimate
was −0.33 percentage points (root-bootstrap 95% interval −5.00 to +4.17,
two-sided sign-flip p = 0.941). Seed estimates were −4.50, −0.50, and
+4.00 points. The five family contrasts summed to −2 completions. On the
revision family alone, both omissions yielded 50 completions versus an
additive prediction of 61 out of 120. That family contrast is descriptive.

On conflict, the joint omission produced 22 completions and the additive
prediction was 19. The transfer interaction was +3.13 points (95% interval
−7.29 to +13.54, p = 0.692). The intervals leave room for either direction
of interaction. All arms had zero unsafe additions under the shared
inference verifier.

The joint rule chose the same 18 lineage and 27 revision replacements as
the single omissions. The training-data audit yielded 191 independently
verified complete trajectories, versus 236 with both guards present,
218 after the lineage omission, and 209 after the revision omission.

The run at `867372836561b9a42f4800cc096b0d957d6e2592` took 269.03 seconds
for audit and training and 82.75 seconds for frozen scoring. Peak RSS was
530.89 MiB per native worker, 379.32 MiB for training, and 444.87 MiB for
frozen scoring. The final audit matched 2,784 Prime Verifiers 0.4.1 traces
to receipts and checkpoints, and confirmed fresh source hashes against
122 archived conflict roots.

- Training lock SHA-256 `336af481a214adc26dcfcb9c2f77cbf6783843b91bbf20cc8247622cb27821cb`
- Frozen report SHA-256 `bb37f90a381bed89419ea0e1c40bf641589a8f897e97aa0dc242d44515a4f4f0`

## Revision-family replication

The earlier 40 revision roots showed a descriptive interaction of
−9.17 percentage points, with 50 joint-omission completions versus 61
predicted by the additive model. This independent check uses 400 fresh
revision roots and the exact same twelve archived checkpoints across
the four arms. The architecture, initializations, 600 training updates,
28,800 supervised action tokens per fit, inference policy, and verifier
are carried forward. There are zero new optimizer updates.

The primary contrast is the paired factorial interaction on verified
completion. The secondary contrast is the full curator minus the joint
omission, to distinguish overall policy loss from nonadditivity. The
runner uses two-sided sign flips and 95% root-bootstrap intervals as in
the original factorial study. A negative primary estimate with a
two-sided p-value below 0.05 is the criterion for replicating the drop.
The estimate uses only the new roots.

Run `python -m metta_research_bench.looped.revision_replication` with
`--reference`, `--reference-checkpoints`, `--prior-checkpoints`,
`--interaction`, `--checkpoints`, and `--out`. The interaction directory
contains the original factorial lock and frozen report. `--checkpoints`
points at the archived joint-omission weights. The runner verifies the
original archive digests and every checkpoint hash before opening the
reserve. Keep the generated report, receipts, traces, and checkpoints
outside Git.

### Fresh revision results

| Training curator | Verified complete / 1,200 | Rate | Premature stops | Policy verifier calls |
| --- | ---: | ---: | ---: | ---: |
| Full | 701 | 58.42% | 456 | 5,625 |
| Drop unique lineage | 670 | 55.83% | 497 | 5,572 |
| Drop current revision | 488 | 40.67% | 592 | 7,049 |
| Drop both | 481 | 40.08% | 575 | 7,721 |

The new additive prediction was 457 completions and the joint omission
yielded 481. The interaction was +2.00 percentage points
(95% root-bootstrap interval −1.59 to +5.75, two-sided p = 0.321). The
interval excludes the discovery estimate of −9.17 points. The three seed
estimates were +9.00, −0.25, and −2.75 points.

The full-to-joint completion drop did replicate. The full curator led the
joint omission by 18.33 points (95% interval 14.58 to 22.17, p = 0.00005).
Removing the revision guard cost 17.75 points with lineage present and
15.75 with lineage absent. The smaller lineage losses were 2.58 and 0.58
points with revision present and absent, respectively. All arms made zero
unsafe additions under the shared verifier.

The run at `987648d3338f9fa17dfa8050042f3e9b0fa93cb6` took 55.83 seconds
with peak RSS of 471.21 MiB. The independent receipt audit verified all
4,800 trials against the twelve archived checkpoints and Prime Verifiers
0.4.1 traces, with no source overlap across the 996 locked training and
evaluation roots or the 154 archived conflict roots.

- Replication lock SHA-256 `fe42fed6c99a242a00e017b36c9ae4c61c016cf79b2394e727cb961940f782f2`
- Frozen report SHA-256 `e85d2f8be56beb454525dc73dcce5c05b6519c228f70e63c86402f03e21a0f37`

## Lineage-family replication

On the 40 lineage roots in the factorial study, keeping the lineage guard
yielded 25 versus 22 completions with revision present and 24 versus 23
with revision absent, each out of 120. The mean paired effect was +1.67
percentage points. The earlier single-rule study had a separate 40 lineage
roots and estimated +5.00 points with revision present.

The replication uses 400 fresh lineage roots and the same four arms and
twelve checkpoints. Each fit used a 44,417-parameter Looped Transformer,
600 optimizer updates, and 28,800 supervised action tokens. The replication
uses zero new optimizer updates. The primary contrast is the mean of
`good − without_lineage` and `without_revision − without_lineage_revision`
within each root and seed. The two conditional effects and the factorial
interaction are secondary. A positive primary estimate with a two-sided
root-level sign-flip p-value below 0.05 is the replication criterion.
Intervals use a 95% bootstrap over roots, keeping the three seeds paired.

Run `python -m metta_research_bench.looped.lineage_replication` with the
same paths used by the revision replication and `--prior-revision`
pointing to its archived report directory. Use a fresh `--out` path. The
runner verifies the original archives and weights before opening the
reserve. Its exclusion manifest includes the earlier revision replication.
Keep weights and generated output outside Git.
