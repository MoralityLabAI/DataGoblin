# ALG-DESCENT-08 / BPD-RISK-02: frozen pre-proposal recurrence (2026-11-27)

[Prospective protocol](https://github.com/MoralityLabAI/DataGoblin/commit/e534bcd4975df9a68282af5b28a97304c068c51b). Synthetic public-witness polynomial certificate routing only. No factor discovery, DLP/security inference, or full native BPD/MMS integration.

## Design and reproduction

64 training roots, 48,172 candidate/history examples; deterministic 30,000-example train subset; 16 calibration roots, 14,174 examples. 16 IID roots (16,292 examples), 24 OOD roots (142,403 examples). All fresh seeds/families follow the registered protocol. V7 generator, two extra necessary-condition fingerprints and exact product verifier unchanged. OOD 114,288 candidates, 38,410 invalid, 25,305 masked, 13,105 surviving invalid. IID 11,344 candidates, 1,899 invalid, 634 masked, 1,265 surviving invalid. Simulated rejection histories are not on-policy history samples.

Logistic 42 params, MLP 517, dense recurrent router 409, three seeds 19/41/67, ten epochs. History enters as mean rejected-feature vector through the same encoder; logistic/MLP receive separate mean and count. Feature access shares public candidate and history data; encodings and model sizes differ. Dense router initializes h0=tanh(inp) with inp=enc(x)+linear(mean_history,enc.weight), then performs three tied updates tanh(inp+W h). Reset control returns to h0 before each update, retaining encoder/history. Train-only smoothed bucket prior, same structural risk policy, exact verifier and work/call caps. No cache. All 1,720 routes close.

Freeze dense weights. SVD W=U S V^T, factors U sqrt(S) Q and Q^T sqrt(S) V^T with Q=exp(A-A^T), exact unmasked reconstruction up to floating-point rounding. Optimize rotation and shared state-only sigmoid gates on 12,000 training cases for eight epochs; per-pass squared logit discrepancy + .1 hidden-state MSE + .02 mean max-over-pass activity. This is a simplified behavioral sparsity proxy, not native BPD adversarial CI/PGD, and does not establish atom identifiability. Select one atom by greatest calibration root-average loss damage per seed (indices 1,2,8). Targeted deletion/enhancement applies same rank-one edit across all three passes. Eight sham shared random rank-one directions have per-pass scaling fitted to clean calibration activation-write RMS; not physical shared parameter-norm controls. Held-out scale mismatch reported, no OOD adjustment.

From repo root:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/datagoblin/research/alg_descent_v8/experiment.py
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python src/datagoblin/research/alg_descent_v8/audit.py

Python/PyTorch CPU, one torch thread. Experiment creates checkpoint and receipt adjacent to source; rerun overwrites them. Separate replay ZIP contains saved checkpoints, v1-v4/v7 dependencies, receipts and audit. Source remote blob hashes match executed files:
experiment.py bc39ce6be8ef88f6159f54e787bb8d329eb75e38
audit.py d6e58f91a54ca7ce6a3466c0aefe6a5fade82e60
SHA256:
experiment.py c39f744bfade5e2e418c26693c2630d794906f78510b0a31089b43797e3826c5
audit.py 497824a420d369e25a6191ebb0307612b0883c61391f5fa485ce5fadf818ca0a
models.pt 81203d5218c50d22dc185a1894ced0298982685dc967cddd36f4a908606d9569
receipt.json 5e48938248920e915b8c22f9ffc1b3c847959ab4fc965a0155ce1e512c3041a4
audit.json ae2fa9e64f8e2c1fa3f9506b05efe2f1eafaa4e80e1261fbaf53fd773e96e577

## OOD results

| Arm | Log loss | Brier | Mean work | Mean warmed route ms |
|---|---:|---:|---:|---:|
| bucket | .321007 | .089588 | 990.2083 | 1.101 |
| logistic | .257810 | .080585 | 984.5278 | 2.161 |
| MLP | .271302 | .088554 | 985.2639 | 2.491 |
| dense | .260888 | .085215 | 984.0694 | 3.337 |
| reset | .248869 | .077045 | 986.6111 | 3.465 |
| ablate | .351392 | .110155 | 987.6528 | 3.696 |
| enhance | .357115 | .115027 | 985.0417 | 3.452 |

Root averages pool three seeds per root, not independent seed/root tests. All 24 or 72 routes per arm close. Timing single randomized-order warmed executions, excludes bank preparation and not end-to-end. Dense vs logistic loss +.003079, 9 wins/15 losses, descriptive bootstrap [-.03485,+.04102]; dense vs MLP -.010413, 13/11, [-.02246,+.000668]. Reset vs dense -.012019, [-.06527,+.03526]. Prediction gate fails. Dense vs logistic work -.4583, 10 wins/6 ties/8 losses, [-3.1944,+2.3472]; vs MLP -1.1944, 6/14/4; vs bucket -6.1389, 8/5/11. Routing gate fails. Neural routing latency higher.

Second-difference family log loss: bucket .316046, logistic .228099, MLP .212050, dense .201903, reset .223771, ablate .310358. Two-child: bucket .325967, logistic .287520, MLP .330553, dense .319874, reset .273968, ablate .392426. Dense does not beat strongest controls on both families.

## Scoped causal pass

Targeted deletion damage +.09050369 nats, 20 of 24 roots worse, paired bootstrap [.0468214,.1412758], both families worse (+.108455 and +.072552). Median pooled sham damage +.03064541; excess above median +.05985828 > .01 registered threshold. Sham damages [.02141081,.18499448,.14501144,.03988001,.00188296,-.02690369,.00531380,.09753406]; three of eight exceed targeted pooled damage. Enhancement also damages loss (+.09623), so no monotonic steering benefit. Calibration shams match clean per-pass RMS to relative error <=1.74e-7; OOD worst mismatch 71.11%, across seeds 36.62..71.11%. No claim of mechanism specificity or stronger null superiority. OOD selected-atom write nonzero at all 266,949 zero-history candidate/checkpoint pairs. This exposure is a necessary opportunity, not proof of decision relevance. First root choice changes 15/72, full trace 59/72, work 58/72, so behavioral gate passes. Ablation mean work +3.5833, bootstrap [-2.0694,+10.875]; no reliable work damage or improvement demonstrated. Only two natural rejections in intact dense OOD routes. Distinguish pre-proposal recurrent computation from multi-attempt memory. 

OOD gated reconstruction mean gate .6943/.7036/.7978, max-over-pass pooled activity .7297/.7294/.8323; per-pass logit RMS .02962/.04143/.05388, max discrepancy .22074/.27626/.38744. Unmasked decomposition logit max error <=1.91e-5. Soft gates retain substantial activity; this is not evidence of a sparse discrete circuit.

## Independent replay

All 40 roots, 99,693 eligible polynomial identities checked by independent convolution; 3,440 log-loss/Brier fields reproduce exactly on this host. Iterative DFS and independent checkpoint inference replay all 1,720 routes: 10,320 fields (closure/work/calls/rejects/first root proposal/trace hash) match, zero discrepancies. Shared generator, structural scores and prior functions are reused; prior independently regenerated from training roots. Double-precision rotation reconstruction max weight error 9.12e-7; selected atom edit error 4.58e-7. All budgets satisfied. This is an independent inference/routing implementation on shared generated banks, not independent training or generator replay.

## Decision

PASS registered causal exposure/damage/median-sham gate with substantial held-out RMS mismatch and sham-tail weakness. FAIL predictive and routing improvement gates. Next prospective experiment should use stronger controls (physical shared rank-one edit vs per-pass matched sham; predeclared matched RMS tolerance, report calibration and held-out mismatch; norm and Jacobian/readout-sensitivity controls, larger null roster), and compare frozen learned rotation with SVD/Haar-rotated bases under identical selection and edit budgets. Test whether the decomposition adds value above basis selection, whether recurrence adds value above a matched trained one-pass model, and whether a calibrated improvement translates into exact closure/work and latency. Native BPD trainer integration remains outstanding. Preserve sealed ARG-03-N.
