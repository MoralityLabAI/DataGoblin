# Gate-first exact-array LDT replication v14

## Result

V14 corrected v13's rank-then-qualify ambiguity by filtering seed cells
through every frozen development gate before ranking them within each
architecture. The fresh-lineage matrix trained four candidates across seeds
61, 107, and 149. Only `exact-array-h4-global-map-v14`, seed 107, qualified.
Its development means were 0.7500 exact status-vector accuracy and 0.7500
exact selected-set accuracy, with 0.5000 for both worst-fold exact metrics.
The gate-first selector therefore emitted a one-candidate confirmation packet.

The single environment-owned held-out join did **not** confirm it. Exact
status-vector accuracy was 0.6250 against a frozen 0.6500 gate, and exact
selected-set accuracy was 0.6250 against a frozen 0.7500 gate. The learned
Pareto set is empty. The reveal count is one, post-reveal tuning is forbidden,
and neither shadow integration nor larger-model training is authorized.

This is useful negative evidence: the model retained perfect held-out safety
and orientation behavior and good candidate-level classification, but those
metrics did not imply reliable whole-array recovery. The exact v10 metrics
correctly blocked promotion.

## Frozen contract and development selection

The harness was frozen and pushed as `65afacd` before official dataset
generation. It used six fresh development and four sealed confirmation
lineages across Centauri, They Sing, Court, and Diplomacy. All 94 prior
lineages were forbidden. Generation found zero prior overlap, zero cross-split
overlap, complete critical-class support, and no confirmation targets in
candidate custody.

The four candidates shared the target-independent 23-feature candidate
encoder. The exact factorized/global pair had identical 416-parameter training
and local status probabilities; only the global selected-count MAP decoder
differed. It changed 17 target-free held-out array predictions. Gate-first
selection retained qualified seeds before applying the frozen
worst-fold-first diagnostic rank; an unqualified rank winner could not
authorize confirmation.

| Architecture and retained seed | Params | Pair F1 | Status F1 | Exact status | Exact selected | Worst status | Worst selected | OOF ECE | Qualified |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| pair reducer, seed 149 | 245 | 0.7678 | 0.7258 | 0.3750 | 0.5833 | 0.2500 | 0.5000 | 0.0530 | no |
| recurrent factorized, seed 61 | 409 | 0.7781 | 0.8980 | 0.5417 | 0.5417 | 0.2500 | 0.2500 | 0.0401 | no |
| exact factorized, seed 61 | 416 | 0.7781 | 0.8955 | 0.5417 | 0.5417 | 0.2500 | 0.2500 | 0.0401 | no |
| exact global selected-count MAP, seed 107 | 416 | 0.7606 | 0.9467 | 0.7500 | 0.7500 | 0.5000 | 0.5000 | 0.0590 | yes |

These are development diagnostics, not held-out architecture evidence.

## Independent held-out result

The steward evaluated only the frozen primary packet. All non-array gates
passed, but both whole-array gates failed.

| Frozen held-out gate | Threshold | Observed | Pass |
|---|---:|---:|:---:|
| safety-gate accuracy | 1.0000 | 1.0000 | yes |
| swap consistency | 1.0000 | 1.0000 | yes |
| preference macro-F1 | 0.7500 | 0.8356 | yes |
| ABSTAIN recall | 0.7500 | 1.0000 | yes |
| status macro-F1 | 0.8000 | 0.8852 | yes |
| expected calibration error | at most 0.2000 | 0.1027 | yes |
| exact status vector | 0.6500 | 0.6250 | **no** |
| exact selected set | 0.7500 | 0.6250 | **no** |

Array-size diagnostics localize the largest failure to size five: exact
status-vector accuracy was 0.6000 for size four, 0.2500 for size five, and
0.8571 for size six. These slices are descriptive and cannot be used to tune
v14.

| Game | Arrays | Pairs | Exact status | Exact selected | Support-aware status F1 | Support-aware pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| Centauri | 4 | 82 | 0.5000 | 0.5000 | 0.8128 | 0.8348 |
| Court | 4 | 102 | 0.7500 | 0.7500 | 0.9375 | 0.8157 |
| Diplomacy | 4 | 66 | 0.7500 | 0.7500 | 0.9062 | 0.9065 |
| They Sing | 4 | 100 | 0.5000 | 0.5000 | 0.8730 | 0.8106 |

| Pressure | Arrays | Pairs | Exact status | Exact selected | Support-aware status F1 | Support-aware pair F1 |
|---|---:|---:|---:|---:|---:|---:|
| boundary Pareto/tradeoff | 4 | 74 | 1.0000 | 1.0000 | 1.0000 | 0.8860 |
| controlled missingness | 4 | 120 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| near-tolerance dominance | 4 | 74 | 0.2500 | 0.2500 | 0.7778 | 0.7842 |
| near-tolerance equivalence | 4 | 82 | 0.2500 | 0.2500 | 0.6121 | 0.4444 |

The pressure slices show a boundary: the sorter handled safety and explicit
missingness but did not reliably compose near-tolerance pair relations into an
exact array decision. High mean status F1 and acceptable ECE therefore remain
diagnostics, not substitutes for the two exact metrics.

## Pre-reveal evaluator correction

The first confirmation command stopped before label access and before creating
an output directory. Its public packet validation exposed two legacy v9
assumptions: the evaluator allow-listed only old prediction-source names and
required every emitted status to be its local probability argmax. The latter
is intentionally false for a globally coupled selected-count decoder.

Commit `5aa969e` corrected only the verifier. It binds each prediction source
to the frozen candidate specification. For the global decoder, it validates
the best non-selected status and stable top-k selected assignment implied by
the local probabilities; factorized sources still require local argmax. A
regression test rejects both source substitution and a tampered selected set.
Predictions, labels, models, seeds, thresholds, selection, and packet hashes
were unchanged. A public-only precheck then traversed the complete packet and
stopped at a deliberately absent label path. The real join followed once and
only once.

## Evaluation review

The scoped deterministic-development preflight returned `warn` with no
blocking finding. Its only warning was the generic adapter's preference for a
subjective dimension; exact environment targets exhaust this claim. The base
policy returned a release-blocking `fail` by design because development arrays
are not held out and no subjective A/B judge protocol exists. It prevents
development evidence from being mislabeled as release evidence.

The public auditor independently reproduced the gate-first selection, all
packet bindings, the fresh-lineage firewall, matched decoder ablation, and all
12 cap/cleanup receipts before the reveal. After the join it also reproduced
the exact held-out decision and label-custody hashes. Its conclusion layers
are:

- metric robustness: `supported_on_declared_independent_v10_holdout`;
- task result: `development_qualified`;
- measurement reliability: `single_primary_heldout_join_bound_and_reproduced`;
- held-out claim support: `not_supported`;
- operational decision: `heldout_not_confirmed_no_promotion`.

## Resource receipts

Every shard completed offline in its own WSL systemd unit under 2,048 MB RAM,
50% CPU, 50 MB/s I/O, zero swap, and a 900-second wall-clock cap. All 12
PID-scoped cleanup audits passed. No label was staged into candidate custody,
no external checkpoint was loaded, and no run-owned process remained.

| Candidate family, three seeds | Trainer seconds | Steps | Peak RAM | Peak CPU | Peak I/O |
|---|---:|---:|---:|---:|---:|
| pair reducer | 84.73 | 125,100 | 73.15 MB | 50.01% | 12.575 MB/s |
| recurrent factorized | 183.84 | 254,700 | 33.25 MB | 50.04% | 0.128 MB/s |
| exact factorized | 237.09 | 254,700 | 33.02 MB | 50.02% | 0.127 MB/s |
| exact global selected-count MAP | 294.79 | 254,700 | 33.02 MB | 50.02% | 0.121 MB/s |

The official matrix completed 889,200 steps in 800.45 trainer seconds. The
earlier capped smoke remains discovery-only. No generated run, prediction,
label, checkpoint, or model weight is committed.

## Bound receipts

- frozen harness commit: `65afacdb788f45fb9fc50e861b23faf81ca022ed`
- pre-reveal evaluator correction commit: `5aa969e8f2229f9614eab279a79864414447605a`
- preregistration: `a3960f7ebf82bbbeaba0ad36819e68e548524c3f1bff178bcd9bba4a8116f159`
- eval-review manifest: `84b4c51b43f34b855435ffac807f8d59ef94d552e38c8e5cc0eb5c4da4011bf6`
- deterministic-development policy: `08a055fd70159e142140acbd2c7a62081d1968cb8489352fce9d4144ea6f0a01`
- scoped development preflight: `8c2d66fc477f2276a69231caf0523c1eaf0a482aaa842511018e7f7b1f259b50`
- base-policy preflight: `3a83227052979eb2b9c22d5208da2fb1619a5b50e21dab6f2f3a4d2c4be7a86b`
- public dataset manifest: `c7ff56bba188d265c8938b8722cecdd9cb715a06d34a45f40039be2f64ee5c49`
- generation report: `8b4d35c90a523311dc1303a6dd5a365c43f2151f81f667ef252d6121325716c0`
- trainer plan: `c8b1aeb15872568362d642aa4cea74577d8f0492d87f1a761fb1acee933a1664`
- complete shard merge: `af6d86a50ead462b0121025ae805eccf06364bae5a3df86b2053ce49ac001718`
- merged trainer summary: `da301ff731746bd9a226b7c61ded2a71939abcded172308ddeae9f7fcc1e5f18`
- merged pair predictions: `39f87fda0348347b38e8cb0824f2508b481be257e18aae7a7c2a87faa5bbd30c`
- merged array predictions: `9681a66e17a70c8eb49a680a19310c94ea9d1980f2e369444c0fe20cd5127fa7`
- development selection: `c6bd7aa4a9837acf22ca5db3ac22dd3ac394dbe62d856ad6f5cdb3f117e8106b`
- primary trainer summary: `706642f2d6dfa1864205261b8c19603105afba964aaaa5c4b0398723241a6668`
- primary pair predictions: `2b1aa7fe2d7dac9560eb639bee3b86ac11b1426792c0a5dde93a146880cc1aaf`
- primary array predictions: `b1521e09bff7456d7107a43f8e123975f3d2f851b0ecc75a8733538507fd1d6e`
- development-only audit: `eac06671410c87c0c5d35f645e78f48d93895f386655702f531fd62f5903c17b`
- corrected evaluator implementation: `20cfabe61e793fd342c7670ab930e2a1be82282811de842061e72f28dc802058`
- environment pair-label custody: `4b5968deb5137ac8f853eab7277f798b5c9580132bd2899066084d2e2c0ef66c`
- environment array-label custody: `eac2366ef54e9ccea115f1c6c8ef504b9535baf82620b3c948880715218e11fa`
- held-out evaluation: `a5e85549f22d81fe908078157caad1435d1212ed2de407bf26a40cfa42b7591b`
- held-out evaluation receipt: `ac5c6a9fb6d735e1b026210c84353184019d1b85b1e88943d4ff0d73cbde762d`
- final independent audit: `d475fdcceef848094fb4944d7181ca4345923f756f61efac0ba786d4e2882883`

## Next gate

Do not tune on v14 confirmation rows, substitute another v14 seed, or lower
either exact threshold. Do not start the LoRA/HRM adapter tournament with this
sorter. A successor must freeze new development and held-out lineages before
generation and treat v14 only as aggregate mechanism evidence.

The next bounded experiment should preserve the exact selected-count control
while testing an explicitly edge-conditioned, set-coupled representation of
near-tolerance dominance and equivalence. It should stratify arrays by size,
preregister size-five diagnostics without making them a post-hoc gate, retain
gate-first seed selection, and require a fresh single-primary held-out pass
before any larger adapter comparison.
