# Edge-conditioned exact-array LDT correction v15

## Result

V15 is the first retained LDT sorter in this succession to pass both frozen
whole-array gates on a fresh, environment-owned held-out split. The sole
primary, `exact-edge-h4-global-v15` seed 167, reached 0.8125 exact status-vector
accuracy and 0.8125 exact selected-set accuracy. The gates were 0.6500 and
0.7500 respectively. Every safety, orientation, pair-quality, uncertainty,
status, and calibration gate also passed.

The learned Pareto set contains only the retained primary. This confirms the
content-free synthetic sorting architecture for one downstream use: a
separately frozen, architecture-neutral adapter tournament. It does **not**
authorize corpus promotion, deployment, direct LoRA/HRM training, or a
universal adapter champion.

## Corrective architecture evidence

V14's single held-out result failed at 0.6250 on both exact metrics, with its
largest descriptive weakness on near-tolerance arrays and size five. V15 did
not tune or reuse those rows. It froze six new development and four new sealed
confirmation lineages, added all ten v14 lineages to a 104-lineage firewall,
and increased fresh size-five support from four to six held-out arrays.

The primary keeps v14's exact selected-count likelihood and global MAP decoder
but replaces the 23-feature independent candidate representation with a
target-free 111-feature edge-difference representation. For each candidate it
adds mean signed, mean absolute, minimum signed, and maximum signed deltas to
all other candidates. Arrays remain bounded at six candidates, so exact subset
normalization enumerates at most 2^6 assignments.

The official 12-cell matrix isolated both intervention axes:

- edge-global versus independent-global changes only the candidate encoder;
- edge-global versus edge-factorized changes only the decoder;
- all candidates use the same pair model within a seed;
- the matched edge decoder pair has identical local status probabilities.

The public auditor observed 48 target-free probability changes and seven
status changes from the encoder intervention, plus 17 status changes from the
global decoder. The edge-global exact lift over independent-global was positive
at every development seed: +0.2083, +0.1250, and +0.2500, for a +0.1944 mean
lift. Its lift over the matched edge-factorized decoder was +0.2500, +0.2083,
and +0.2917, for a +0.2500 mean lift.

## Gate-first development selection

Gate-first selection found two qualified edge-global seeds, 167 and 269, then
applied the frozen worst-fold-first rank and retained seed 167. No other
architecture had a qualified seed.

| Architecture and retained seed | Params | Pair F1 | Status F1 | Exact status | Exact selected | Worst status | Worst selected | OOF ECE | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| pair reducer, seed 269 | 245 | 0.7924 | 0.7803 | 0.5000 | 0.5833 | 0.2500 | 0.2500 | 0.0667 | no |
| independent exact-global, seed 223 | 416 | 0.7764 | 0.8885 | 0.6250 | 0.6250 | 0.5000 | 0.5000 | 0.0517 | no |
| edge exact-factorized, seed 167 | 768 | 0.7658 | 0.8895 | 0.5417 | 0.5417 | 0.2500 | 0.2500 | 0.0547 | no |
| edge exact-global, seed 167 | 768 | 0.7658 | 0.9372 | 0.7917 | 0.7917 | 0.5000 | 0.5000 | 0.0547 | yes |

Development evidence authorized only one primary-only confirmation packet.

## Independent held-out confirmation

The public packet passed a no-label evaluator traversal before the one real
join. The environment then opened the sealed labels once. Post-reveal tuning
is false, no alternate seed may be substituted, and the final auditor bound
the prediction, trainer, dataset, and label-custody hashes.

| Frozen held-out gate | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| safety-gate accuracy | 1.0000 | 1.0000 | yes |
| swap consistency | 1.0000 | 1.0000 | yes |
| preference macro-F1 | 0.7500 | 0.8148 | yes |
| ABSTAIN recall | 0.7500 | 0.8000 | yes |
| status macro-F1 | 0.8000 | 0.9733 | yes |
| expected calibration error | at most 0.2000 | 0.0889 | yes |
| exact status vector | 0.6500 | 0.8125 | yes |
| exact selected set | 0.7500 | 0.8125 | yes |

Exact status-vector accuracy by size was 1.0000 for size four, 0.6667 for size
five, and 0.8571 for size six. All sizes were preregistered and supported.

| Game | Arrays | Pairs | Exact status | Exact selected | Support-aware status F1 | Support-aware pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| Centauri | 4 | 90 | 0.7500 | 0.7500 | 0.9641 | 0.8099 |
| Court | 4 | 90 | 0.7500 | 0.7500 | 0.9641 | 0.8099 |
| Diplomacy | 4 | 120 | 0.7500 | 0.7500 | 0.9699 | 0.7930 |
| They Sing | 4 | 66 | 1.0000 | 1.0000 | 1.0000 | 0.8691 |

| Pressure | Arrays | Pairs | Exact status | Exact selected | Support-aware status F1 | Support-aware pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| boundary Pareto/tradeoff | 4 | 82 | 1.0000 | 1.0000 | 1.0000 | 0.9508 |
| controlled missingness | 4 | 120 | 1.0000 | 1.0000 | 1.0000 | 0.9630 |
| near-tolerance dominance | 4 | 82 | 0.5000 | 0.5000 | 0.9111 | 0.8575 |
| near-tolerance equivalence | 4 | 82 | 0.7500 | 0.7500 | 0.6559 | 0.3656 |

The slices remain descriptive. In particular, low support-aware pair/status F1
on near-tolerance equivalence did not prevent three of four exact arrays, but
it is not evidence to weaken those diagnostics or tune v15 after reveal.

## Evaluation review

The scoped deterministic-development policy returned `warn` with no blocking
finding; its only warning was the generic adapter's recommendation for
subjective dimensions that are outside an exact environment-target claim. The
base policy returned a release-blocking `fail` because the 24 development
arrays are not held out, no subjective A/B ordering protocol exists, and the
development item count is below its generic minimum. Those facts preserve the
boundary between development selection and held-out confirmation.

The final auditor's conclusion layers are:

- metric robustness: `supported_on_declared_independent_v10_holdout`;
- task result: `development_qualified`;
- measurement reliability: `single_primary_heldout_join_bound_and_reproduced`;
- development claim support: `supported`;
- held-out claim support: `supported`;
- operational decision: `confirmed_for_downstream_preregistration_only`.

## Resource receipts

Every official shard completed offline in an owned WSL systemd unit under
2,048 MB RAM, 50% CPU, 50 MB/s I/O, zero swap, and a 900-second wall-clock
cap. All 12 PID-scoped cleanup audits passed. No steward label was staged into
candidate custody, no external checkpoint was loaded, and no run-owned process
remained.

| Candidate family, three seeds | Trainer seconds | Steps | Peak RAM | Peak CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|
| pair reducer | 69.38 | 125,100 | 73.86 MB | 50.11% | 14.315 MB/s |
| independent exact-global | 164.81 | 254,700 | 33.20 MB | 50.03% | 0.128 MB/s |
| edge exact-factorized | 165.37 | 254,700 | 33.53 MB | 50.06% | 0.128 MB/s |
| edge exact-global | 164.89 | 254,700 | 33.41 MB | 50.00% | 0.128 MB/s |

The official matrix completed 889,200 steps in 564.45 trainer seconds. The
earlier capped smoke is discovery-only and excluded from every official claim.
No generated run, prediction, label, checkpoint, or model weight is committed.

## Bound receipts

- frozen harness commit: `1b54f81d743136b6cc3e550259c1edea7a6db8df`
- preregistration: `d28c52cca0c1bf4b904ed708b9b0c2df7db11d32ae82109ca809f816f87da905`
- eval-review manifest: `80eb5c63218eab00449704cfb58b0ff1aac0d3e0deea2187ceb9d39ac3007e2e`
- deterministic-development policy: `08a055fd70159e142140acbd2c7a62081d1968cb8489352fce9d4144ea6f0a01`
- scoped development preflight: `d303772ac0abf3416f3a27378dde374a855c254fcbd342f0e4b832a1c370bba9`
- base-policy preflight: `01725e5cc2dd719a7b012b19931780006d9ebe682aa7d237fc2fab064eaf5db1`
- public dataset manifest: `5ef4f03ce8c220b97c74fcb77be80b92ab3a8865687bd4aa6032752db0f0d3cc`
- generation report: `b344ae1edff479660f37a67970d487b99c0e7260ffd0dca4db830c19f0b4c74b`
- trainer plan: `6c7c8b9cb7e8d845aaaf3e462a70fc0bfdd0a3b55f8b38d106479fa6a4830252`
- complete shard merge: `0cf5331e4858824a3fb8b68134c71a824d74efa0a99233f00347dafad906b32c`
- merged trainer summary: `ea8510b8df122d98ff722b378401032c0cf5feb4ee36e684d4fb95bbc727ba1b`
- merged pair predictions: `79d996b18e9914129bcae0156391b772d18fefe9593ffa78038c9533c2eea69e`
- merged array predictions: `2a52a6c8dae4d4646bc5045906511446047b9c0e3de7a658162dbce240004c64`
- development selection: `e81b8bed1b8081c61411674c1551349a49ceb775266362f7e9b573f3855961d4`
- primary trainer summary: `d0a3b31339d9c783b2cf7a4381864bf1a75c07ce3dbcbd14d31eb900c3e92ce3`
- primary pair predictions: `5c9c0cd569c5e40fb02a947e2dd84b6482e7285a20c963c5e5e039bd34cf4cc5`
- primary array predictions: `6f5e88d76a223852427e5eb410e0cafd8fc6e6c3353886de8332326d9a49c560`
- development-only audit: `716b94edeec060408a1ba3f68f6b55044eb19fa7181b2401dedd254c7b2d3826`
- public auditor implementation: `34ff271a5a424830d00ce6983f3b5aa6d64ee8c9dc8a3dcc8efffcb3dc53bbfa`
- environment pair-label custody: `a67258253b0eff10fcbe1e81c46309fe6b48d536724d6b8aa190227005be283c`
- environment array-label custody: `f3cc67f30e13a03f6f0dc4a6fa3787aa82eacbff12220fcc874f8a25eccd6d60`
- held-out evaluation: `45a44bce52bc30e08f752c9ceae833c5eff4ae8c23fa8ac4a1a9847b1c8cbc43`
- held-out evaluation receipt: `ef2bd4844a550c57aa2eb16ff43e6c86823d082a991d9f167f2d562ccd4588db`
- final independent audit: `165f346c1120e301190d111932fca8acf113f765a457c69c91edf6e34e0b9227`

## Next gate

Freeze a downstream adapter tournament before generating its task data. The
confirmed sorter may provide shadow recommendations over content-free adapter
receipts, but the environment verifier remains the decision owner. The matrix
should compare an unchanged deterministic control, a low-rank LoRA-like
adapter, and an optional HRM-Text-style recurrent adapter under equal data,
seed, resource, and checkpoint contracts. It must not download weights or
train a real checkpoint unless an operator separately supplies a capped local
runtime and approves that scope.

Adapter promotion must require per-game safety qualification, no critical
false release, no lineage contamination, and exact whole-array recovery under
a newly preregistered held-out split. Mean ECE, status F1, or the sorter alone
cannot promote an adapter.
