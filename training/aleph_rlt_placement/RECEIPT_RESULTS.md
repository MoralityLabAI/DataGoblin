# One-bit receipt gym results

## Tournament decision

The frozen rule nominated neither C nor D for a 10B-token run. All nine models kept the first bit across the entire ten-prior parity grid. After LDT adaptation D sent a constant bit in 24 of 24 seed–frame combinations, C in 21 of 24. Both spoiled more decisions than they corrected. D scored better than C on two seeds despite transmitting less information.

## Delayed parity

Each pair of integers identifies the learned eight-world truth tables at priors 0.07 and 0.08. Bit x of an integer is the upload in world x, after fixing the upload at world zero to zero. 170 stores the first bit and 232 stores majority. All value-equivalent codes receive full credit.

| Model | Seed | Codes at .07 / .08 | Optimal at .07 / .08 | Learned Brier at .07 / .08 | Bayes Brier for these codes |
|---|---:|---|---|---|---|
| C | 137 | 170 / 170 | 0 / 1 | 0.1424 / 0.1408 | 0.1423 / 0.1408 |
| D | 137 | 170 / 170 | 0 / 1 | 0.1423 / 0.1408 | 0.1423 / 0.1408 |
| MLP | 137 | 170 / 170 | 0 / 1 | 0.1429 / 0.1414 | 0.1423 / 0.1408 |
| C | 163 | 170 / 170 | 0 / 1 | 0.1424 / 0.1409 | 0.1423 / 0.1408 |
| D | 163 | 170 / 170 | 0 / 1 | 0.1425 / 0.1410 | 0.1423 / 0.1408 |
| MLP | 163 | 170 / 170 | 0 / 1 | 0.1424 / 0.1409 | 0.1423 / 0.1408 |
| C | 191 | 170 / 170 | 0 / 1 | 0.1424 / 0.1408 | 0.1423 / 0.1408 |
| D | 191 | 170 / 170 | 0 / 1 | 0.1423 / 0.1408 | 0.1423 / 0.1408 |
| MLP | 191 | 170 / 170 | 0 / 1 | 0.1425 / 0.1410 | 0.1423 / 0.1408 |

Side-only Brier is 3/14 at every prior. The optimal gain over side-only is the larger of 1/14 + λ/56 and 3/49 + 15λ/98. They are equal at 4/53. The exact expected scores enumerate the finite population, so even a small code regret at the two bracket points is observable. Initialization uncertainty remains. Brier is lower-is-better throughout.

### Why a useful bit can miss this switch

For the store-first code, flip its output in exactly one of the eight worlds and let the receiver recalibrate optimally. The change in Brier risk is (15λ − 1)/196 for every choice of world. This follows by enumerating the eight neighbors at λ = 0 and 1 and using the fact that risk is affine in λ. Thus for 1/15 < λ < 4/53 the first-bit receipt is strictly locally optimal under a one-world edit, while a nonlinear receipt has lower global risk. At 0.07, a one-world edit costs 0.000255 and a coordinated change to majority gains 0.000740. The neural optimizer can change several worlds together. This calculation identifies a measurable difficulty of the objective, while the observed first-bit solutions even at λ = 0 show that the local barrier alone cannot explain the pilot.

## Fresh LDT transfer

A low-cost check compares one measured axis with a calibration reference. The model sees one result per goal. Training offers pair clearance, interval abstention and one-opponent array membership. The held-out families add a second rival and ask whether the focal entry survives or must remain unresolved. Labels and side observations come from the repository interval policy and array-status verifier. Four new frames per held family yield 384 world/query pairs in total, with unequal public query probabilities. The frames vary bounds and deployment weights within two fixed logical structures.

| Model | Parity checkpoint, zero-shot | Adapted, familiar families¹ | Adapted, fresh families | Ideal readout of adapted bit² | Corrected / spoiled³ |
|---|---:|---:|---:|---:|---:|
| C | 0.2648 | 0.1493 | 0.3335 | 0.1499 | 10.6% / 41.5% |
| D | 0.2597 | 0.2065 | 0.3037 | 0.1536 | 5.8% / 24.9% |
| MLP | 0.3155 | 0.1655 | 0.3175 | 0.1244 | 10.8% / 30.5% |

¹ New frames of the three training structures. Zero-shot parity and LDT use different feature meanings. Adaptation gives each frozen parity checkpoint 600 updates on LDT training families before transfer.

² The ideal readout knows the game distribution and may decode each frozen bit optimally. Its gap to realized Brier measures readout headroom. The gap from that ideal to the unrestricted oracle measures lost information.

³ Probability mass of errors corrected and correct decisions spoiled against the calibrated side-only Bayes decision. Probability ties at 0.5 predict one for both. Each family, frame and training seed has equal weight within its group.

| Fresh family | C | D | MLP | Side only | Best linear receipt² | Fixed majority² | Best one-bit receipt² |
|---|---:|---:|---:|---:|---:|---:|---:|
| array_two_opponents | 0.3122 | 0.2391 | 0.2446 | 0.1532 | 0.0891 | 0.0396 | 0.0396 |
| array_unresolved_after_challenge | 0.3548 | 0.3682 | 0.3905 | 0.1541 | 0.0953 | 0.0402 | 0.0402 |

The best linear baseline may choose any of seven nonzero GF(2) parities independently for each frame. The majority baseline uses the same three-input threshold in every frame. Both get the same single check.

| Model | Seeds bracketing switch | Positive parity gain at both points | Beats best linear on LDT | Beats MLP on LDT | Beats other recurrent arm on LDT |
|---|---:|---:|---:|---:|---:|
| C | 0/3 | 3/3 | 0/3 | 1/3 | 1/3 |
| D | 0/3 | 3/3 | 0/3 | 2/3 | 2/3 |

### What transferred

The two-opponent point label simplifies to “majority OR the shared metric.” The point-plus-interval unresolved label simplifies to “majority AND the isolated metric.” The verifier separately checks one calibration axis. Majority with a calibrated downstream receiver attained the exact optimum on every fresh frame. A useful small-model contribution here would be to discover this reusable Boolean statistic from public training receipts and demonstrate it on an unseen structure.

The three LDT training families never activate the second-rival mask. The test introduces a real second opponent and its input slot together. This tests a difficult deployment shift, but cannot isolate relational extrapolation from an input channel whose LDT semantics were never trained. For a follow-up, use a fresh split that exercises both rival slots during training, holds out their relation, and includes a supervised majority-sender control. Keep the old scores frozen. Require the sender to beat side-only with a refitted receiver before spending further compute on the learned readout.

## Training and replay

C and D each have 6,880 trainable scalars, two width-32 slots and two visits to the same dense residual block. C compares scores on two options. D lets a proposal read an evidence slot on each visit. These are small controlled wiring prototypes. A full self-attention language model would be a separate experiment. The ordinary MLP has 1,601 parameters. Each seed and arm saw the same ordered games for 1,200 parity and 600 LDT updates. The 18 training phases took 217.9 summed wall seconds on one capped, low-priority CPU worker, with 3 GiB and a 1,200-second training cap. The run used zero accelerator credits.

The [protocol](RECEIPT_PROTOCOL.md) fixes the objective, split and promotion rule. The [gym instructions](README.md#one-bit-receipt-gym) give the training and scoring commands. The run receipt records checkpoint hashes, optimizer traces, software versions, per-prior truth tables and per-frame scores. The holdout seed is operator-local. A new seed regenerates independent bounds and weights in the same families.
