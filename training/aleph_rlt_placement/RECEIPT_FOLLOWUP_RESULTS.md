# Coordinated receipts and two-rival transfer

## 10B decision

Neither C nor D earned a 10B-token run. Coordinated gym supervision taught both senders the switch on all three seeds. Their learned receivers paid more Brier loss than the original outcome-only models on every seed. After sequential LDT adaptation, zero of nine checkpoints still passed parity. On the fresh relation task, the MLP scored best among learned arms and side-only beat all three on Brier.

## Did the sender learn the switch?

A seed passes when its hard eight-world code is optimal at all four bracketing priors and its learned receiver beats side-only Brier at each. The ordered four priors are .070, .080, .072 and .079, straddling the exact 4/53 transition. Only whole-code supervision uses the external comparison of 128 candidate uploads. The sender receives the world and deployment prior. The fresh receiver sees one hard bit, a verifier result and public query context. The threshold and its evaluation labels are absent from the training grid.

| Arm | Original outcome only | Exposure-matched outcome only | Coordinated | After LDT adaptation | Mean realized Brier at bracket, coordinated |
|---|---:|---:|---:|---:|---:|
| C | 0/3 | 0/3 | 3/3 | 0/3 | 0.16711 |
| D | 0/3 | 0/3 | 3/3 | 0/3 | 0.18777 |
| MLP | 0/3 | 0/3 | 2/3 | 0/3 | 0.20151 |

Canonical truth tables use bit x for world x and fix the message in world zero to zero. The training enumerator selected masks 46 and 170. They differ in worlds 2 and 7. The teacher provides a risk margin across all eight training priors, a secant constraint between the observed bracketing priors, and hard teacher messages for receiver training. The second outcome control matches sender exposure, receiver game order and learning-rate annealing. The experiment therefore tests a supervised recipe as a bundle. The external solver interpolating the two endpoint risk tables had worst held-prior regret 2.8e-17.

| Arm / seed | Codes at .070 / .080 | Codes at .072 / .079 | Code intervals on the .050–.100 diagnostic grid |
|---|---|---|---|
| C-137 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07575, 170 at 0.07600–0.10000 |
| D-137 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07500, 174 at 0.07525–0.07525, 170 at 0.07550–0.10000 |
| MLP-137 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07450, 174 at 0.07475–0.07475, 170 at 0.07500–0.10000 |
| C-163 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07600, 170 at 0.07625–0.10000 |
| D-163 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07525, 170 at 0.07550–0.10000 |
| MLP-163 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07450, 42 at 0.07475–0.07475, 170 at 0.07500–0.10000 |
| C-191 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07575, 42 at 0.07600–0.07600, 170 at 0.07625–0.10000 |
| D-191 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07400, 42 at 0.07425–0.07525, 170 at 0.07550–0.10000 |
| MLP-191 | 46 / 170 | 46 / 170 | 46 at 0.05000–0.07450, 174 at 0.07475–0.07475, 170 at 0.07500–0.10000 |

All nine supervised senders got the correct code on all four bracket points. One MLP receiver missed the side-only criterion. Before LDT adaptation the original C/D models averaged about 0.1416 Brier at the brackets; the coordinated C/D models averaged 0.1671 and 0.1878. The comparison shows why a code-only metric can reward a policy that is less useful to its deployed partner.

The diagnostic grid spacing is 0.00025. A threshold between adjacent grid points is unresolved. An intermediate code identifies worlds changing their message at different priors.

## Fresh two-rival LDT cases

The two rival slots both carry measured evidence in every training frame. Training exposes disjoint masks in both orders and overlapping doubletons. Held cases use strict containment in both orders under the native point SELECTED and interval UNRESOLVED rules. Four frames per status and orientation produce 16 held frames and 768 world/query pairs. Each world is equiprobable; late queries are weighted. Both native statuses simplify to the singleton metric on these particular held constructions. The four groups exercise two orientations and two status contexts of this one logical problem.

| Arm | Familiar Brier, primary | Held Brier, primary | Held Brier, original initialization | Held Brier, majority-supervised | Clamped majority + learned readout | Majority bit accuracy |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.10469 | 0.22753 | 0.26234 | 0.22630 | 0.22095 | 98.2% |
| D | 0.15338 | 0.22841 | 0.24295 | 0.24431 | 0.23640 | 96.4% |
| MLP | 0.09491 | 0.22000 | 0.24917 | 0.25245 | 0.25026 | 99.0% |

Primary and majority-supervised branches start from the same coordinated checkpoint. The original-initialization branch starts from the frozen v1 outcome checkpoint. Each gets the same 600 ordered LDT training games and a fresh optimizer. Clamping feeds the exact majority of the three measurements to the trained downstream network.

| Primary arm | Ideal readout for its learned bit | Constant bit, seed–frames | Corrected / spoiled vs side-only | Seeds beating linear | Seeds beating each rival |
|---|---:|---:|---:|---:|---|
| C | 0.11815 | 0/48 | 23.8% / 16.3% | 0/3 | D 2/3, MLP 1/3 |
| D | 0.14560 | 16/48 | 22.6% / 20.5% | 0/3 | C 1/3, MLP 0/3 |
| MLP | 0.11632 | 0/48 | 25.1% / 15.6% | 0/3 | C 2/3, D 3/3 |

| Held native status and mask orientation | C | D | MLP | Side only | Majority | Best linear | Optimal one-bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| point_selected_contained_1_2 | 0.23883 | 0.24596 | 0.22300 | 0.19851 | 0.13234 | 0.07475 | 0.07475 |
| interval_unresolved_contained_1_2 | 0.22873 | 0.23058 | 0.21628 | 0.16056 | 0.10704 | 0.07235 | 0.07235 |
| point_selected_contained_2_1 | 0.22279 | 0.22411 | 0.21824 | 0.17704 | 0.11803 | 0.06688 | 0.06688 |
| interval_unresolved_contained_2_1 | 0.21977 | 0.21300 | 0.22249 | 0.16362 | 0.10908 | 0.07420 | 0.07420 |

The best linear control reached the unrestricted optimum on 16 of 16 frames. Consequently the strict beat-linear gate was mathematically impossible on this draw. The independent reasons to withhold promotion are that every arm lost to side-only on mean realized Brier, neither recurrent arm beat the MLP on every seed, and none retained the parity pass after adaptation. Across held frames, side-only scored 0.17493, majority 0.11662 and best linear 0.07204. The learned C, D and MLP receivers scored 0.22753, 0.22841 and 0.22000. Each analytic control gets an ideal receiver calibrated for each frame.

C transmitted a nonconstant bit on all 48 seed–frames. Its ideally refitted receiver scored 0.11815, versus 0.14560 for D and 0.11632 for the MLP. The useful information in the messages and the lower quality of the actual predictions coexist. At a 0.5 threshold C corrected more decisions than it spoiled, while its probability estimates still lost on Brier. Forcing the exact majority bit into a majority-trained C receiver also gave 0.22095, well above the 0.11662 available to an ideal majority readout.

If this work continues on CPU, freeze a useful sender, train the downstream receiver to predict the environment-verified conditional target frequency, and interleave parity probes during LDT learning. A new promotion panel needs demonstrable headroom above its strongest control before assigning expensive runs. This pilot gives C an edge over D as a sender prototype and leaves the MLP as the stronger end-to-end control.

### Two worlds must agree on when to change

With the world numbered x = x₀ + 2x₁ + 4x₂, mask 46 sends x₀ if x₁ = 0 and sends 1 − x₂ otherwise. Mask 170 always sends x₀. Only worlds 2 and 7 change messages. At λ = 4/53, either complete code gets the optimum. Exact enumeration gives each halfway code excess risk 1/1484, even with its own optimally refitted receiver. Independent world logits can therefore yield a worse receipt at precisely the point where the model is indifferent between two optimal policies.

For a network of small researchers, each candidate code needs just two independently checked endpoint risks to price it at every prior in this parity game. A compiler can compare two affine risks once for the public deployment prior, then apply the selected truth table in all eight worlds. That implements the two coordinated changes with a single public decision and one transmitted bit. The result follows from the specified target mixture and conditional verifier, so another family needs its own calibration.

### Why a nonlinear receipt can still help on coordinate questions

In a neutral reference game, independently choose a uniform target coordinate and a uniform calibration coordinate from the same three measured bits. The best linear receipt has Brier 1/9. A selector that stores x₀ or x₂ according to x₁ attains 11/108, the optimum over all 128 partitions. When the calibration reveals the selector, its conditional risk is 1/12; either other calibration gives 1/9. The gain over linear is 1/108. This is a separate exact calculation. The seeded LDT frames have different query probabilities, with their own ceilings shown above.


## Reproduction

The [frozen protocol and commands](RECEIPT_FOLLOWUP_PROTOCOL.md) specify the promotion rule and seed commitment. To reproduce the fine-grid diagnostic, run the summary inside the CPU worker with the additional argument `--checkpoints ../receipt-v2`. The [earlier pilot](RECEIPT_RESULTS.md) records the original outcome baseline and single-rival training results. The 45 new training phases used 450.1 summed wall seconds, three matched seeds and one capped CPU worker. The models have 6,880 parameters for C/D and 1,601 for the MLP. No paid compute was used.
