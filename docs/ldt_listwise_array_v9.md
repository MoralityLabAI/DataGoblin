# Listwise LDT array gym v9

## Result

V9 tested whether a direct, permutation-equivariant array head could close the
whole-set gap left by v8. It improved the selected diagnostic's mean
array-status macro-F1 from the pair reducer's 0.7668 to 0.8951 with one pooled
loop and 0.9103 with three recurrent loops. The three-loop head also improved
mean exact-array and exact-selected-set accuracy to 0.5833, with 0.5000 in its
worst lineage-heldout fold.

That is useful architecture evidence, but it is not a pass. The frozen
development contract required 0.6500 mean exact-array accuracy and 0.7500 mean
exact-selected-set accuracy. No candidate qualified, so the selector recorded
`confirmation_join_authorized: false`. The evaluator was not run, the private
pair and array labels were not read, and the one-reveal confirmation budget
remains unspent.

There is no retained primary, corpus or checkpoint promotion, deployment,
shadow integration, or larger-model authorization.

## Dataset and custody

The public fixture contains six development lineages and four fresh
confirmation lineages. It has:

- 552 development and 350 target-free confirmation ordered pairs;
- 24 development and 16 confirmation arrays;
- 126 development and 82 confirmation tournament reports; and
- array sizes 4, 5, and 6.

Every development lineage contains every critical preference, rationale, and
array status. Minimum lineage support is therefore six, and every
leave-one-lineage-out training and validation fold remains class-complete.
Development and confirmation contexts are disjoint, all 10 lineages are fresh
relative to the 44 forbidden v1-v8 lineages, and all unordered pairs have both
orientations.

The pair model receives the 45 target-independent primitives from v7. The
listwise head receives 23 candidate-local primitives: public metric interval
bounds, widths and missingness flags, the safety-qualified bit, and parameter
bucket rank. It receives no game, pressure, policy decision, rationale,
lattice field, target, or verifier-policy execution result.

Confirmation inputs contain no targets. The environment-owned pair and array
labels were sealed outside all candidate workspaces before training. Only
their hashes and custody receipt are public.

## Objective ablation

All setups retain paired-orientation batches and exact antisymmetric pair
inference. Pairwise calibration uses only out-of-fold development predictions.
The pair reducer is the control. The other heads encode each candidate, pool
the array by a permutation-invariant mean, exchange tied recurrent messages,
and predict the four candidate statuses directly.

Selection retains one seed per architecture using the frozen
worst-fold-first rule:

| Selected diagnostic | Params | Pair F1 | Safety | Swap | ABSTAIN | Status F1 | Exact arrays | Selected set | Worst selected set | OOF ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pair reducer, seed 97 | 245 | 0.7758 | 1.0000 | 1.0000 | 0.8000 | 0.7668 | 0.4167 | 0.5000 | 0.2500 | 0.0493 |
| pooled listwise, seed 29 | 377 | 0.7808 | 1.0000 | 1.0000 | 0.8000 | 0.8951 | 0.5000 | 0.5000 | 0.2500 | 0.0474 |
| recurrent listwise, seed 29 | 409 | 0.7808 | 1.0000 | 1.0000 | 0.8000 | 0.9103 | 0.5833 | 0.5833 | 0.5000 | 0.0474 |

Across all nine runs, maximum mean status macro-F1 was 0.9103 and maximum
mean selected-set exactness was 0.5833. Thus the direct head learned materially
better per-candidate statuses, while complete-set recovery remained the
limiting factor. Mean ECE remained about 0.047-0.050 for the selected seeds;
calibration, safety, orientation, and development support were not the
bottleneck.

The factorized per-candidate status cross-entropy should therefore not be
described as a set-coupled objective. It operates over a whole array and is
permutation-equivariant, but correlated candidate errors still defeat exact
set recovery.

## Confirmation boundary

The selector wrote one seed and one target-free prediction group per
architecture before any possible verifier join. All three diagnostics missed
at least the mean selected-set threshold; the strongest recurrent diagnostic
also missed the mean exact-array threshold.

Because the public authorization bit is false, the evaluator refuses before
resolving or hashing either private label path. A unit test exercises this
ordering with nonexistent label paths. There are deliberately no v9 held-out
quality numbers in this report.

## Resource and recovery receipts

The original all-candidate invocation correctly aborted at the frozen
900-second cap. It had reached only a partial matrix, so none of its predictions
entered selection. Its receipt records 77.52 MB peak RAM, 74.85 MB average
RAM, 49.88% CPU, 16.25 MB/s peak I/O, successful stage and PID-scoped cleanup,
and no staged private labels.

The unchanged candidate matrix was then run as two architecture shards and
three recurrent seed shards. Pair and pooled each covered all three frozen
seeds. Recurrent used one shard per frozen seed because its complete
architecture shard was projected to approach the same wall-clock limit. Every
shard ran under its own verified WSL cgroup:

| Shard | Seconds | Steps | Peak RAM | CPU | Peak I/O | Exit | Cleanup |
|---|---:|---:|---:|---:|---:|---:|---:|
| pair reducer, seeds 29/53/97 | 298.91 | 125,100 | 32.67 MB | 50.00% | 0.045 MB/s | 0 | passed |
| pooled listwise, seeds 29/53/97 | 474.06 | 254,700 | 32.96 MB | 49.98% | 0.045 MB/s | 0 | passed |
| recurrent listwise, seed 29 | 279.47 | 84,900 | 32.05 MB | 49.97% | 0.041 MB/s | 0 | passed |
| recurrent listwise, seed 53 | 216.09 | 84,900 | 31.41 MB | 49.79% | 0.044 MB/s | 0 | passed |
| recurrent listwise, seed 97 | 227.87 | 84,900 | 32.80 MB | 49.47% | 0.046 MB/s | 0 | passed |

Each unit enforced 2,048 MB RAM, zero swap, a 50% CPU quota, a 50 MB/s I/O
limit, and a 900-second wall clock. The merge rejects a shard unless its cgroup,
cleanup, artifact hashes, candidate/seed coverage, public prediction coverage,
and target-free claim boundary all validate. The operational sharding changed
neither data, objectives, seeds, thresholds, nor ranking.

No weights, checkpoints, private labels, predictions, or generated runs are
committed to the repository.

## Skill evolution

The run was bound to curation-skill fingerprint
`4a601e0dfa9989fef1492674fd0fb5355d6be6d39152f0de21c8c00318fe266a`.
V8 and both v9 listwise heads repeat the same deterministic distinction between
high per-candidate status quality and weak whole-set recovery. Following the
skill's mutation rule, its tournament-succession reference now requires mean
and worst-fold exact selected-set gates, distinguishes a direct array model
from a set-coupled loss, and preserves sealed confirmation after this failure.
The post-run skill fingerprint is
`b74de9417f153a47d34d9e94413510f186af1d7ce52e30c9fd8fc813f1dc4e7f`.

This mutation is guidance for future frozen contracts. It does not alter the
v9 preregistration or retroactively change its result, and its procedural value
still requires validation in the next fresh gym.

## Bound receipts

- preregistration:
  `6020e647afd12ef5478065ba92156f07824098db6d5837188da1251aeaebafdd`
- public dataset manifest:
  `3e78e6f15fd23f1b968344b123ce847886633de84301cf7ffa0ebedcaf60c150`
- private pair-label custody hash:
  `af97a7fed35aba85c4ac21cb6265cf26d35bb4a045fb2f4923505f94732506a8`
- private array-label custody hash:
  `9cc7f60382cd558d18d7abe061532d488668821d48c96d77a65bb3214f0519b0`
- discarded capped monolithic run:
  `a153366d98a731cd762a83e3a9062ed7cf7d53d9eee0c1ceba87d541ea8dc9c0`
- capped pair-reducer shard:
  `5d785c864f9ac741b646f53b5ba943491b17c9db153cae6a8aeafe39db986a9a`
- capped pooled-listwise shard:
  `4ff464816efc9f0fcf4e59cec1536a13ba44878edae0a8ef03f3c17f73e0e8df`
- capped recurrent-listwise seed-29 shard:
  `c2d4a729a553ac3301eaed6e6a91e6f67a63af4b8b4973765d3913e75acc140f`
- capped recurrent-listwise seed-53 shard:
  `4bf3c94941cf44f0b65206c6fb41d95cf230340f7efae239e304b504f3f66e93`
- capped recurrent-listwise seed-97 shard:
  `2a39728a9ecf2592b3101deddee47fcfb0b06687f897410fad39bda7917b6b03`
- complete shard merge:
  `30b267e9ba2fb7ca3598c84c69a17dafdd4ffdfabcf16805e51bcd14f10709d9`
- merged trainer summary:
  `17bbc360265c1c5b50518535af40117be3925a5db37f2282a356ee9ca1e7fa8e`
- merged pair predictions:
  `a501a050320d20066a8866de0e77a8ea8df38960cccefedcb1939675daa9b1b5`
- merged array predictions:
  `ab7c87987e3a1443061dba037f1f0e39bc4a92a3e44eca8b6b15db3ed67b8da5`
- development selection:
  `d6e3a00d8b0686e611558e22ea015a9952494a36bf13969ca2d73ee4eeb47c68`

## Next gate

A future, separately frozen development contract should compare the recurrent
mean-pooled head with an edge-conditioned message-passing head that explicitly
represents pairwise candidate differences. It should add a set-coupled
selected-set loss or constrained status decoder, retain the pair reducer as a
control, and keep exact antisymmetry and the independent primitive views.

The next fixture should use fresh development and sealed confirmation lineages,
select only from mean and worst-fold development evidence, and preserve the
same confirmation firewall. A larger checkpoint remains out of scope until a
small sorter first clears both whole-array gates and passes one independent
confirmation without post-reveal tuning.
