# Strict-1B compute and credit envelope

The 999,352,320-parameter dense model uses 14 physical blocks. Ten middle blocks execute on every reasoning pass. Two passes execute 24 blocks per position and three passes execute 34. Stored BF16 weights require 1.999 GB. A representative 16-byte-per-parameter Adam inventory needs 15.990 GB before activations, sharding workspaces and collectives. One full-context sequence requires 8 MiB for a single BF16 residual state and 192 MiB for fully materialized BF16 vocabulary logits. Microbatch size, vocabulary sharding and checkpointing determine the realized peak. The miniature worker's 3 GiB cap cannot hold that training inventory. [Frozen full-scale configuration](../aleph_rlt_placement/fullscale.json)

## Full-context cost

For a 2,048-position causal context, two passes require approximately 10.532 GFLOP per nonpadding position and three passes require 14.716 GFLOP. The count includes dense forward and backward projection products, the tied vocabulary head and ideal triangular attention. Padding, rematerialization, compiler/kernel behavior, I/O and evaluation change sustained throughput. The operator's 55-hour, 350-credit estimate for 10B positions implies 50,505 positions/s and about 6.36 credits/hour. That rate would require 532 aggregate useful TFLOP/s at two passes or 743 at three. The calibrated pass count is unknown. [Geometry and FLOP calculation](../aleph_rlt_placement/fullscale.py)

| Processed positions on one arm | Two-pass FLOPs | Three-pass FLOPs | Hours / credits at operator rate | Three-pass hours / credits if rate was measured at two passes |
|---:|---:|---:|---:|---:|
| 100M | 1.053 × 10¹⁸ | 1.472 × 10¹⁸ | .55 / 3.5 | .77 / 4.89 |
| 1B | 1.053 × 10¹⁹ | 1.472 × 10¹⁹ | 5.5 / 35 | 7.68 / 48.90 |
| 10B | 1.053 × 10²⁰ | 1.472 × 10²⁰ | 55 / 350 | 76.85 / 489.04 |
| 20B | 2.106 × 10²⁰ | 2.943 × 10²⁰ | 110 / 700 | 153.70 / 978.08 |

The last column assumes wall time scales proportionally with the 1.3973× analytical work ratio. It is a scenario pending end-to-end TPU measurement. The machine-readable scale release also converts each full-scale budget to illustrative one-core years using the three observed direct-arm arithmetic rates from the largest completed scale rung. Those rates come from short padded FP32 batches and exclude evaluation and checkpoint I/O. They describe a work equivalent on the measured core.

## A staged plan under 1,000 credits

Reserve **100 credits** for rate uncertainty, recovery and billing overhead. This leaves a working cap of 900. The 10M restart, 100M throughput, 500M stability and 1B futility checks are prefixes of each matched leg. At each stop measure actual cumulative credits divided by cumulative processed nonpadding positions, including billed overhead. Recompute what the remaining working balance buys before authorizing the next stage. Independent confirmation of a challenger and an admitted production corpus remain prerequisites for starting either leg.

Freeze the 1B futility comparison before either arm starts. The existing selection limits are 1% relative text regression, two percentage points of checked-macro regression and 10% throughput regression. Compare identical independent families at the same checkpoint positions, account for repeated interim looks, and continue both arms when an early comparison is inconclusive and the working cap permits it. One matched seed per arm supports a decision between those checkpoints.

| Matched position per arm | Cumulative hours, both arms | Cumulative credits | Check |
|---:|---:|---:|---|
| 10M | .11 | .70 | Checkpoint, optimizer and data-cursor restart |
| 100M | 1.10 | 7 | Measured inclusive TPU throughput and credit burn |
| 500M | 5.50 | 35 | Numerical stability and verifier competence |
| 1B | 11.00 | 70 | Predeclared independent-family futility comparison |
| 10B | 110.00 | 700 | Adjudicate both matched checkpoints |

If both arms reach 10B at the quoted rate, extend the winner by **5.7B**. Total consumption is 25.7B positions, 141.35 hours and 899.5 credits. The winner reaches 15.7B and 100.5 credits remain. If a preregistered futility condition eliminates the challenger at 1B, taking the winner to the full 20B instead processes 21B positions across both arms, requires 115.5 hours and costs 735 credits. That outcome leaves a matched comparison only up to 1B.

The original 10B + 10B + 10B plan needs 1,050 credits and 165 hours at the stated rate. It exceeds the balance by 50 and the working cap by 150. At the proportional three-pass rate, two 10B legs alone use about 978.08 credits and leave 21.92. The 900-credit working cap would cover about 9.20B positions per arm at that rate without an extension. All amounts are conditional planning figures. Paid expenditure remains on hold.
