# Final external reasoning allocation tournament

**Keep solver allocation symbolic.** At the registered 4.98112-second deadline, the best fixed portfolio found 26 of 48 held proofs, matching the coverage ceiling of an oracle that could choose the cheapest successful tested order for each equation. The train-selected fixed portfolio solved 23. The learned MLP solved 23, 17 and 24 across three seeds. The repository's one-pass width-four LDT head solved 23 on every seed. Both learned promotion gates failed.

The strongest observed fixed order was two-frontier search, congruence decomposition, verified state feedback, then normalization. It was one of 24 fixed orders specified before the experiment. Its rank was determined on held data. The independently train-selected order was congruence, two-frontier search, normalization, then feedback. Keep that distinction in any follow-up deployment comparison. This external tournament leaves the dense direct Aleph reference and the paid-run decision as established in CONFIRM14.

## Measured comparison

Each policy received 24 unchanged positive destinations, their 24 altered-but-provable counterparts and 24 false mutations. A complete intervention set requires both positives to obtain kernel-accepted proofs in time and the false mutation to be rejected. All policies rejected all 24 false mutations through the common exact semantic gate. A separate Lean challenge below tested stale proof terms against changed goals.

| Policy | Proved /48 | Complete sets /24 | Deployment seconds | Training + deployment seconds | Kernel calls |
|---|---:|---:|---:|---:|---:|
| Normalization | 9 | 4 | 188.90 | 188.90 | 42 |
| Congruence | 13 | 0 | 35.26 | 35.26 | 14 |
| Two-frontier search | 15 | 5 | 40.25 | 40.25 | 16 |
| Verified state feedback | 21 | 9 | 213.60 | 213.60 | 80 |
| Historical congruence → normalization order | 19 | 6 | 194.09 | 194.09 | 50 |
| Train-selected fixed portfolio | 23 | 10 | 181.92 | 747.69 | 50 |
| Train-selected structural portfolio | 23 | 10 | 183.56 | 749.39 | 50 |
| Best fixed portfolio on held | 26 | 13 | 180.59 | 180.59 | 61 |
| MLP seed 401 | 23 | 10 | 196.34 | 762.84 | 66 |
| MLP seed 409 | 17 | 8 | 205.35 | 771.57 | 53 |
| MLP seed 419 | 24 | 11 | 183.71 | 749.75 | 55 |
| LDT seed 401 | 23 | 10 | 181.90 | 748.50 | 50 |
| LDT seed 409 | 23 | 10 | 181.82 | 748.48 | 50 |
| LDT seed 419 | 23 | 10 | 181.82 | 748.18 | 50 |
| Retrospective per-equation oracle | 26 | 13 | 176.30 | 176.30 | 53 |

Each fixed order has zero fitting expense if named in advance. Choosing among them from training outcomes incurs their acquisition cost. The held winner's 180.59 seconds is its observed deployment cost under this convention. Selecting the winner from this table would require a new confirmation cohort.

Deployment time includes inference, prior-state acquisition when requested, all earlier failed attempts and checks, and the final attempted action. The primary table caps each charge at the common deadline. Completing indivisible actions past the deadline would instead cost the held winner 284.80 seconds, the train-selected fixed control 347.77 seconds, and MLP seed 419 285.16 seconds. The verified-feedback standalone incurred 325.81 uncensored seconds. Each replay used the same measured singleton cost for an identical emitted source and recorded every attempt. Different policy orders were evaluated offline over those measurements.

Building the training labels cost 564.34 summed kernel wall seconds for 195 unique submissions and 1.42 seconds of measured solver, semantic-gate and feature work. MLP fitting took 0.28–0.73 seconds and LDT fitting 0.60–0.89 seconds. Every learned seed had lower coverage than the best fixed portfolio at the primary deadline. Amortizing training therefore cannot establish a same-coverage learned cost advantage on this sample. Against the train-selected fixed control, MLP seed 419 solved one additional positive at 1.79 additional deployment seconds. The other two MLP seeds supplied no consistent gain. LDT matched that control's coverage with differences of about a tenth of a second in deployment.

## Composition, interventions and expensive checks

The training set had six lineages in each of four single-equation Nat/List families, with 48 closed, kernel-accepted positive examples exported. The held panel had six fresh lineages per family. The generator excluded 1,511 historic equation/reversal keys and excluded exact training goals and their reversals. The shared external solver knew how to lift proofs through product construction. The product and nested-product obligations tested its ability to allocate leaf-solving effort on a combination absent from training.

| Held family | Positives | Best fixed portfolio | MLP 401 / 409 / 419 | LDT 401 / 409 / 419 |
|---|---:|---:|---:|---:|
| Crossed Nat expansion | 12 | 4 | 4 / 4 / 4 | 4 / 4 / 4 |
| List reversal over nested unequal maps | 12 | 6 | 6 / 6 / 6 | 6 / 6 / 6 |
| Nat × List product | 12 | 8 | 7 / 3 / 7 | 6 / 6 / 6 |
| Nat × (List × Nat) product | 12 | 8 | 6 / 4 / 7 | 7 / 7 / 7 |

The best fixed portfolio solved 13 unchanged and 13 edited destinations. The train-selected order solved 13 unchanged and 10 edited. Verified feedback alone solved 10 unchanged and 11 edited destinations. Its certificates begin from an acquired, kernel-checked source trace, lift the trace through a public wrapper and spend up to eight expansions repairing the remaining discrepancy. Its 21 successes exceed the 15 from standalone two-frontier search and cost 80 kernel calls versus 16.

Of 48 forced stale transports, **35 produced an explicit Lean type mismatch after a separately accepted source context**. This comprised 18/24 changed-but-provable target types and 17/24 false target types. Eight challenges lacked a kernel-verified antecedent within the given caps, and five invalid queries timed out. Those 13 remain unresolved. The verifier returned no unexpected acceptance of a submitted stale proof. The registered requirement for complete stale-transport rejection therefore also failed.

Across train and held, 364 distinct Lean 4.19.0 singleton sources produced 229 accepted declarations, 81 expected type mismatches and 54 timeouts. They consumed 1,539.05 summed wall seconds and 1,031.62 summed child CPU seconds. The 169 distinct held sources alone had a median 3.71 seconds and a 90th percentile of 12.11 seconds. The longest accepted held source was 37,425 bytes; the largest attempted source was 129,526 bytes. The subprocess execution cap was 12 seconds, and termination overhead raised the longest observed wall charge to 17.99 seconds.

The primary deadline was four times the 1.24528-second median of nine earlier archive singleton checks. At 2.49056 and 9.96224 seconds the same held-winning fixed order solved 14 and 30 positives, respectively, while the per-equation oracle solved 16 and 31. Solver order, source-term length and kernel startup all affected which proofs arrived in time. These are small synthetic families with six independent lineages per family and shared primitives. The reconstruction seed is intentionally public; exclude this released diagnostic from future private tests.

The cost accounts cover solver execution, verification, feature construction, routing and fit. Historical-panel reconstruction, process setup outside the timed solver/checker, and evidence packaging are shared experiment overhead. Identical sources share a measured duration across alternatives; independent wall-time repetitions of every policy order would include additional machine variation.

## Evidence and replay

The verified evidence archive is `ALLOCATION_EVIDENCE.zip` with SHA-256 `21c22612c863261d4483b1fc721dd4ce85eb54c7a83c9f28e4c0c977f40bdbef`. Its internal manifest covers 740 members. The verifier checks the six frozen models, all 364 proof-source hashes, independent denotations and equation fingerprints, and exactly replays all 2,592 primary policy decisions. The archive includes redacted diagnostic paths and the original local log digests. The train-only export is separate inside the release. The model-selection freeze SHA-256 was `58b153bf8f03a90ae57c8056eabbd7a803def64d13c652c2a1d9a035294b0394`.

A training-only pilot exposed a Lean layout ambiguity for one-step nested `calc` proofs. Its errors and source hashes were retained locally. The final renderer emits top-level component proofs, and the failing example passed kernel replay after the change. The final cohort started in a fresh directory. The final train phase used at most four one-core workers and the held phase at most two. Sampling the last two workers recorded 202–203 MiB peak Lean working sets and at least 7.28 GiB available system memory. No paid compute was used.

From the repository root, set `PYTHONPATH=src` in a Python environment with CPU PyTorch, NumPy and psutil. Supply the pinned Lean 4.19.0 executable through `--lean` if it is installed elsewhere. A fresh run measures local wall times again and can change the measured policy ranking. Replaying the released archive reproduces the original decisions.

```text
python training/aleph_next/state01/allocation_worker.py --core 0 allocation_replay.py --split train
python training/aleph_next/state01/allocation_worker.py --core 0 allocation_routes.py freeze
python training/aleph_next/state01/allocation_worker.py --core 0 allocation_replay.py --split held
python training/aleph_next/state01/allocation_worker.py --core 0 allocation_routes.py score
python training/aleph_next/state01/allocation_evidence.py verify --evidence ../../outputs/ALLOCATION_EVIDENCE.zip
python -m pytest -q tests/test_aleph_allocation_final.py
```

For a sharded replay, pass `--shards 4 --shard I` to the replay worker with each index 0–3 and then use `--collect` once for that split. Keep train collection and fit freeze before held generation. The registration and fit details are in [the tournament protocol](ALLOCATION_PROTOCOL.md) and [the model contract](ALLOCATION_MODELS.md).
