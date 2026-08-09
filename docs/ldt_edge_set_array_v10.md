# Edge-conditioned set-coupled LDT gym v10

## Result

V10 tested whether explicit candidate-difference messages and a genuine
selected-set loss could close the whole-array gap left by v9. The matched
soft-Dice head improved status macro-F1 over the factorized edge head in all
three frozen seeds, by 0.0106 on average. It improved exact status-vector and
selected-set accuracy in one seed and tied in two, for a mean delta of 0.0139.
Worst-fold selected-set accuracy did not improve.

This is directional mechanism evidence, not a passing adapter. Every selected
diagnostic missed the frozen whole-array thresholds. The strongest selected
exact result was the recurrent factorized head at 0.5417 exact status vectors
and selected sets, below the required 0.6500 and 0.7500 means and with only
0.2500 in its worst fold. The selector therefore recorded
`no_qualified_primary: true` and `confirmation_join_authorized: false`.

The private pair and array labels were not staged, resolved, or read. The
confirmation evaluator was not run, and the one-reveal confirmation budget
remains unspent. There is no retained primary, checkpoint or corpus promotion,
deployment claim, or larger-model authorization.

## Frozen evidence contract

The v10 fixture uses six fresh development lineages and four fresh sealed
confirmation lineages from Centauri, They Sing, Court, and Diplomacy. It has
552 development and 350 target-free confirmation ordered pairs, 24 development
and 16 confirmation arrays, and array sizes 4, 5, and 6. All 54 lineages from
v1-v9 are forbidden. The generated fixture had zero prior-lineage overlap,
zero cross-split overlap, complete critical-class support, and both
orientations for every unordered pair.

Before model training, the two exact metrics passed an independent,
environment-owned held-out robustness audit: 20 of 20 probes across 10
counterexample families agreed with a separately coded reference scorer. The
tournament directly calls that candidate-ID scorer for each array. Candidate
features remain target-independent, and confirmation inputs contain no target
fields.

The edge encoder is permutation-equivariant. It combines 23 candidate-local
public primitives with mean signed, mean absolute, minimum, and maximum
differences for each of the 22 non-bias primitives, producing 111 features.
The matched edge heads have identical 1,041-parameter architectures and differ
only in the preregistered selected-set soft-Dice weight: 0.0 versus 0.5.

## Selected diagnostics

The frozen selector retained seed 59 for all four architectures using the
preregistered worst-fold-first rule.

| Diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact status | Exact selected | Worst exact selected | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair reducer | 245 | 0.7842 | 1.0000 | 1.0000 | 0.8667 | 0.7480 | 0.4583 | 0.5833 | 0.5000 | 0.0564 |
| recurrent factorized | 409 | 0.7842 | 1.0000 | 1.0000 | 0.8667 | 0.8980 | 0.5417 | 0.5417 | 0.2500 | 0.0564 |
| edge factorized | 1,041 | 0.7842 | 1.0000 | 1.0000 | 0.8667 | 0.8669 | 0.4167 | 0.4167 | 0.2500 | 0.0564 |
| edge + soft Dice | 1,041 | 0.7842 | 1.0000 | 1.0000 | 0.8667 | 0.8796 | 0.4167 | 0.4167 | 0.2500 | 0.0564 |

All selected diagnostics passed safety and swap invariance, and their ECE was
well below the 0.20 ceiling. Calibration is not the current bottleneck. The
edge representation also did not beat the smaller recurrent candidate-local
head on exact recovery. The supported local statement is only that the
set-coupled surrogate produced a small, consistent status-F1 lift relative to
its matched edge control on these development lineages. It did not establish
reliable exact-set improvement or close the target gap.

## Evaluator supersession

The first complete run used the mathematically equivalent index-based exact
comparison already present in the trainer. Before accepting that run as the
durable evidence package, the evaluator was changed to invoke the independently
validated candidate-ID scorer directly. The entire unchanged 12-cell matrix
was rerun under the same frozen contract.

An offline comparator verified identical candidate receipts, selected seeds,
primary decision, and byte-identical merged and selected pair/array prediction
hashes. The second run supersedes the first for evaluator provenance. This
equivalence creates no additional model sample, label reveal, or promotion
authority.

## Resource and custody receipts

All 12 candidate/seed cells completed under isolated verified caps: 2,048 MB
RAM, zero swap, 50% CPU, 50 MB/s I/O, and 900 seconds per cell. All exit codes
were zero, all PID-scoped cleanup audits passed, and no private labels were
staged. The superseding matrix completed 889,200 training steps in 422.51
trainer seconds. Its observed maxima were 74.31 MB RAM, 50.22% CPU, and 11.055
MB/s I/O.

No generated run, prediction, checkpoint, weight, private label, or hidden seed
is committed to the repository.

## Bound receipts

- preregistration: `4ef45893704c7cd40f73b3822b082301d0f98f3e0dab4fe0e109268d8662d78a`
- public dataset manifest: `584d52e01460b935b9f49fef17bf7ae0fc58a8f9827b5c5d8e8614d19fabfa9b`
- private pair-label custody hash: `72e64a2fed1551808a5e3d50c7e0a7cfbcd26b252adf1d9be5db2f4ed471822c`
- private array-label custody hash: `6dad1820bdd6d90078d67a85d152447608c8c5fc0aa7766cea7161a52f41926b`
- trainer-plan preflight: `e7eff31c86a50e11a98a05edc18c04552078242ef4f776f2bdbb2ac06a710753`
- held-out metric audit: `2ff88579e010953378c160ec10c25e94f5093585e697a1fabb726fec72899087`
- superseding shard merge: `0eef527d707b44c252b3a30467803762a933bff93c85775c1ea6f3eb3d24d4de`
- superseding trainer summary: `4133b232724f2995973ca6ce232fad46109a05b3520f60008f9ffe8396a42648`
- merged pair predictions: `a5c641e15d9d634a52818997a6396d1dad542d8a608a67f5ce89aa1fad95533d`
- merged array predictions: `45470ac6a5c6f37902333ceaef105e41a215005a97480268ad1a8901c12a0a74`
- superseding development selection: `a50c2cfb0aa9e2d51b37c17e752ce68dd8d70684ebfd1561d4db96804162d46d`
- superseding selected trainer summary: `eafb51ce4dbc0c9b1e6ea483c7cd83a98b4861be41e2fd05afee134c707ec07b`
- evaluator-equivalence receipt: `b2d311df3d70238cb0e98f52ec9597964098adf558e789b5be47241afa047a25`

## Next gate

Do not tune the Dice weight against these same lineages. A future, separately
frozen development contract should test a constrained structured decoder or
explicit subset/cardinality model against the recurrent factorized control.
It should preserve the candidate-ID exact metrics, fresh lineages, seed
ensemble, worst-fold gates, and sealed confirmation firewall. Only a small
sorter that clears both whole-array development gates should be allowed one
independent confirmation join; LoRA or HRM-scale training remains downstream
of that evidence.
