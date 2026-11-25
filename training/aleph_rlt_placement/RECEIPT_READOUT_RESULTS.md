# Fresh receivers for frozen Data Goblin uploads

On the same frozen C uploads, the fresh MLP receiver improved on the intact C receiver on all three seeds. Its mean Brier loss was 0.20975. The fresh C receiver reached 0.22093, the intact pair 0.22475 and the ideal side-only control 0.16934. An ideal receiver given C's bit reached 0.11221. The upload contained usable information that all three trained reader families left unrealized.

The preregistered C readout rescue failed. Fresh C beat intact C on one of three seeds, fresh MLP on zero of three, and side-only on zero of three. Each of the four seed-averaged families also favored side-only. The full cross evaluated 27 independently fitted receiver checkpoints on the same 16 new frames of v2's containment topology. All 27 lowered Brier from their matched random initialization. All 27 remained above ideal side-only Brier.

## Holding C's upload fixed

Each entry is a family-balanced Brier loss. Lower is better. Same-seed rows share the uploaded bits, public queries and side receipts.

| Sender seed | Random C | Intact v2 C | Fresh C | Fresh D | Fresh MLP | Side only | Ideal C bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| 137 | .25307 | .23027 | .21142 | .20986 | .20379 | .16934 | .10784 |
| 163 | .26136 | .22505 | .22637 | .21966 | .21138 | .16934 | .11499 |
| 191 | .25287 | .21893 | .22501 | .22979 | .21406 | .16934 | .11379 |
| Mean | .25577 | .22475 | .22093 | .21977 | .20975 | .16934 | .11221 |

A new cell-frequency fit reduced mean loss on C uploads by .00382 for the recurrent C reader and .01500 for the MLP relative to the intact C pair. The fresh MLP still paid .09754 above the ideal frozen-bit receiver. The finite external controls fit the exact posterior separately in each evaluation frame. The learned reader receives the public query, verifier bit and upload and must transfer its training calibration.

| Fresh containment family | Intact C | Fresh C | Fresh MLP | Side only | Ideal C bit |
|---|---:|---:|---:|---:|---:|
| Point, singleton first | .23593 | .22695 | .21330 | .17939 | .11948 |
| Interval, singleton first | .22282 | .22407 | .20982 | .15474 | .10276 |
| Point, singleton second | .22129 | .21652 | .20540 | .15884 | .10735 |
| Interval, singleton second | .21895 | .21620 | .21045 | .18440 | .11923 |

## All sender and receiver combinations

These means average the three paired seeds and weight each status-orientation family equally.

| Frozen sender | Intact | Random C | Random D | Random MLP | Fresh C | Fresh D | Fresh MLP | Ideal sender bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | .22475 | .25577 | .26565 | .25162 | .22093 | .21977 | .20975 | .11221 |
| D | .22722 | .25537 | .26065 | .25190 | .21875 | .22637 | .22747 | .14133 |
| MLP | .21215 | .25578 | .26631 | .25163 | .21940 | .22182 | .20604 | .11289 |

The best linear upload with an ideal per-frame reader scored .06652 and attained the unrestricted optimum on all 16 frames. The held family uses both rival slots and both native statuses, but on these containment constructions the target simplifies to the singleton coordinate. The fresh draw replicates the readout gap on that relation. It supplies no new topology test or basis for a 10B nomination.

## When parity changed

At the four original bracket priors .070, .080, .072 and .079, all nine coordinated parity senders learned the optimal canonical sequence 46, 170, 46, 170. The original v1 outcome-only senders reached that four-code criterion on zero seeds. After the earlier v2 joint LDT adaptation, zero of the nine coordinated descendants retained that sequence. For example, all four priors elicited code 168 from the C-137 LDT sender, 160 from C-163 and 128 from C-191. These are sender-only results. The earlier joint parity gate also required useful predictions from the receiver. [Earlier coordinated pilot](RECEIPT_FOLLOWUP_RESULTS.md)

The new fit optimized physically separate receiver parameters. Each of the 27 fits recorded the sender's 104 raw parity logits at updates 0, 100, 200, 300, 400 and 500. Across 162 observations the maximum absolute deviation from the start of the fit was exactly zero. The sender parameter digest also matched at the end of every fit. The earlier forgetting had already occurred when this assay began. Freezing that state preserved it through receiver training.

## Reproduction and receipts

The fixed fit used 24 original training frames, 500 one-frame Adam updates per receiver, a hashed source contract and three matched seeds. Scoring verified all 27 fitted receiver checkpoints before creating the new private seed. The one-core worker had a 3 GiB cap. The held sample has 16 frames and 768 world/query pairs shared across the candidate systems. Worlds and frames contribute repeatedly to paired comparisons. [Frozen protocol and replay commands](RECEIPT_READOUT_PROTOCOL.md)

| Local receipt | SHA-256 |
|---|---|
| Training manifest | `bd8a00c7001b5cd919437f18a59649fe482daa97382098dbee70ba2893be3ceb` |
| Private evaluation seed commitment | `54d4e6778b0d0a097e5df1e7bbab31d05511118f5c324f841e17fbdd901d4939` |
| Complete evaluation results | `fbcd179f23a81aed7a48d909837b02eda130cbce661a442c712371e91630adcd` |
