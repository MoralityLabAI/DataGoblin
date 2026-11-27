# ALG-DESCENT-02 development receipt (2026-11-26)

Synthetic public-witness polynomial certificate routing. No factor discovery, discrete-log solving, or cryptographic security inference. This is a development experiment; the evaluation seeds are now disclosed. The original v1 tree argument is independent of Cogentic's technical proof.

## Reproduce

From repository root, with PyTorch installed:

    python src/datagoblin/research/alg_descent_v2.py > receipt_v2.json
    python src/datagoblin/research/audit_alg_descent_v2.py

Local execution: Python 3.13.5, torch 2.10.0+cpu, one torch CPU thread. Source SHA256 e8472b83d09a89e5eac270cec050ed46d811c2a795b07199ef5a9882d8fde6a5. Git blob 59c544774ac5aabef39f21376853cdcdb01a7b70. Generated full JSON SHA256 a3a2455dcc00ab20e54e0f30da1e295faebf2766934e7c2bcec6fcbf347273f4 (byte-level reproducibility may depend on software versions). The source was read back from GitHub and its blob matched the locally executed source. Checkpoints and raw JSON are not committed; the script regenerates them/receipts from seeds, although it does not persist model weights.

## Design

40 training roots, seeds 1000..1039, degrees 8..16; 1,817 teacher examples (initial candidate selection and selection after one simulated invalid proposal). Teacher uses exact no-cache DP over the generated banks. 18 IID roots, seeds 20000..20017, degrees 8/12/16; 18 OOD roots, seeds 30000..30017, degrees 24/32/48. Three model initializations (11/23/37), MLP 865 parameters and GRU 881, both six epochs. 1,332 evaluation rollouts. Model features are ten public candidate descriptors, including two inexpensive evaluation fingerprints. Candidate arities 2/3/4/8, balanced and random contiguous partitions, guaranteed valid binary fallback, and polynomial-coefficient corruption. All accepted actions checked by exact multiplication. Every child must close. Model hidden state resets at each node and updates after invalid actions. Cache scoped per root, keyed by exact canonical polynomial, used only after full closure, one work unit per hit. Cost budget 3*n^2 coefficient operations, call cap 8*n. Every attempted verification is charged; generation, fingerprint construction, DP preparation, model inference, dictionary lookups on misses, assertion checks and terminal handling are off-meter. Logical cached bytes = len(polynomial)+16 per entry; not a bound on RSS or the public candidate bank. Utility eviction uses observed subtree work/byte without predicting reuse probability. This is a proxy for resource-aware routing, not native BPD or MMS.

## OOD no-cache results

| Arm | Closed | Mean spent work | Rejections |
|---|---:|---:|---:|
| Binary | 18/18 | 1273.56 | 144 |
| Wide | 15/18 | 1645.50 | 94 |
| Fingerprint | 18/18 | 1075.56 | 32 |
| MLP | 50/54 | 1546.61 | 139 |
| GRU | 51/54 | 1458.07 | 103 |
| Offline DP | 18/18 | 912.22 | 0 |

All-run work includes incomplete rollouts and is secondary to closure. Offline DP has privileged validity and downstream cost information and is a no-cache comparator, not a lower bound on cache-enabled policies. On the 50 jointly closed seed/root pairs, GRU minus MLP work has mean -98.08, median +5, 16 wins, 5 ties, 29 losses. No significance claim; the 54 model evaluations share 18 roots. GRU seed means 1478.39, 1424.72, 1471.11; MLP 1541.44, 1596.89, 1501.50. IID no-cache closure GRU 51/54 and MLP 51/54; spent work 203.67 versus 221.24; fingerprint 18/18 at 162.94. The neural arms fail the heuristic comparison on both splits.

## OOD cache ablation: all-run mean work

| Arm | Off | 128 B LRU | 128 B utility | 512 B utility |
|---|---:|---:|---:|---:|
| Binary | 1273.56 | 1109.89 | 1122.83 | 1075.33 |
| Wide | 1645.50 | 1586.39 | 1590.17 | 1585.28 |
| Fingerprint | 1075.56 | 895.06 | 903.33 | 878.61 |
| MLP | 1546.61 | 1400.87 | 1404.11 | 1398.48 |
| GRU | 1458.07 | 1336.20 | 1337.41 | 1334.81 |

Closure unchanged within each arm for these caps. All budget assertions held. The 128-byte LRU GRU arm has 333 cache hits and max logical peak 127 bytes across 54 rollouts. Utility eviction fails to beat LRU at matched 128 bytes for every arm. The best public policy is fingerprint + 512 B utility in the tested arms; 512 B LRU was not run, so no inference about its relative performance.

## Audit

An independent coefficient convolution on 10 additional roots of degree 16 (seeds 40000..40009) checked 2,144 candidate identities against oracle validity, including 681 invalid candidates; 225 invalid candidates passed both cheap fingerprints. The audit checks each DP choice's recurrence and root oracle rollout against its stated value. This checks algebraic identities and local optimality over the candidate bank, not the learned model's training trajectory or an independently reconstructed full inference replay.

## Decision: FAIL as learned-router improvement

The GRU helps relative to the MLP mean but loses most jointly solved comparisons, and both underperform the strong deterministic heuristic. The tiny closure memo helps all arms; heuristic utility eviction is weaker at matched cap. No BPD steering, native MMS integration or wall-time claim is established.

Next registered candidate (fresh roots, equal action and feature access): mask known fingerprint failures with an exact necessary-condition check; train a residual score over the strong heuristic using cost-to-go/regret and validity auxiliary targets; supervise multi-attempt recurrent state; compare reset/shuffled history to normal recurrence, and use shared candidate banks. Give learned and heuristic policies identical measured preprocessing and RAM budgets. For cache utility, estimate reuse probability times expected saved work per byte; include matched LRU at every cap. Require matched closure and root-paired cost improvement over the strongest heuristic before scaling. Treat this as proposed, not preregistered before the development pilot.
