# ALEPH-STATE-09 | certified operator transfer

2026-11-25. CPU-only continuation of STATE-08. A remains the reference and both paid slots remain on hold.

## Design

Reuse the identical 288 finite-domain source operators from STATE-08 (six families, 48 each). All 64 point values per operator are independently available from the source oracle. A fresh task seed 2026112515 creates 144 register programs and 96 two-turn graph tasks at each length 8, 32, 64: 720 tasks. This is held composition of known operators, not zero-shot identification of new operators or a sealed family split. Register programs vary one coordinate of the initial state across eight values; each instruction updates one register with a selected operator evaluated on both registers. Graph tasks vary all eight starting vertices and compare walks before and after editing one vertex's operator assignment (half of tasks are explicit no-op revisions). A genuine assignment edit need not change the final answer. Every execution is evaluated against the source truth-table interpreter.

The executable codec reconstructs the 31,392 distinct base/expanded grammar functions independently from formula tags and coefficients and asserts identical ordering to the source catalog. For each operator, full enumeration finds a shortest code among exact formula matches and base formulas with at most two corrections; otherwise use a raw table. A two-bit outer tag distinguishes formula, patched formula and raw; formula tag is four bits and each coefficient three bits. A patch count bit and nine bits per corrected coordinate/output suffice for one or two patches. Raw codes cost 194 bits. The grammar dictionary is not needed during execution. Code length excludes shared decoder implementation, record framing/padding and digests, not actual Python memory footprint. Every encoded operator is decoded and executed at all 64 inputs; every result must equal its independently supplied table. SHA-256 digests bind the code and verified truth values; these are integrity receipts, not proofs of oracle provenance.

Reconstruct STATE-08's eight-witness sample32 conjectures from their observed labels only (122 exact, 165 false, one abstention at source). Compare stateless oracle, fully verified raw table, fully verified compact representation, lazy pointwise oracle cache, early conjecture without guard, and early conjecture with mandatory verification before first use of each unknown coordinate (including observed-point cache from the eight initial labels). No truth-value calls are required by fully certified decoding at deployment. Raw table and oracle share reference execution control flow; the compact formula decoder is independently checked against the full source table. The guarded policy never trusts an unverified prediction.

## Results

| Policy | Full tasks /720 | Terminal tasks /720 | Correct terminal answers /8064 | Trace hits /279552 | Total oracle reads |
|---|---:|---:|---:|---:|---:|
| Stateless oracle |720|720|8064|279552|279552|
| Certified raw |720|720|8064|279552|18432|
| Certified compact |720|720|8064|279552|18432|
| Lazy cache |720|720|8064|279552|16873|
| Early conjectures |133|211|4798|171816|2304|
| Guarded conjectures |720|720|8064|279552|17058|

The exact policies solve all 278 outcome-changing and 2026 outcome-unchanged start-specific graph revision pairs. The unguarded conjectures solve both answers in 159/278 changed and 1584/2026 unchanged pairs. They solve zero of 144 complete 64-step register tasks. The guard encounters 1,795 disagreements with conjectures at newly queried coordinates. Its total includes 2,304 initial witness reads and 14,754 deployment reads. A pure lazy cache uses fewer reads; the conjecture does not add verified information and should not be deployed merely to reduce point checks. Full certification amortizes 18,432 reads over 279,552 evaluations, a 15.17x reduction relative to no caching. For the measured workload, lazy cache is the stronger oracle-cost baseline.

## Representation cost

| Family | Operators | Encoded payload bits | Raw fallbacks |
|---|---:|---:|---:|
| Base |48|864|0|
| Threshold |48|864|0|
| Diagonal |48|864|0|
| Sparse1 |48|1344|0|
| Sparse2 |48|1770|0|
| Alien |48|9312|48|
| Total |288|15018|48|

Raw payload 55,872 bits; codec 15,018 bits, 3.72x compression including fallback. For 240 representable operators alone, 46,560 vs 5,706 bits = 8.16x. Byte-aligned individual payloads occupy 2,064 vs 7,200 bytes before framing or certification metadata. A single 32-byte digest per operator adds 9,216 bytes to both. The receipt stores two hashes per operator and JSON, not a packed binary store; payload compression should not be reported as the on-disk receipt ratio. A fixed shared evaluator cost must be counted if comparing end-to-end storage.

All 64 selected single-coordinate revisions invalidate the old decoded function; re-encoding a newly read complete table restores correctness. The simulated update costs 4,096 reads under a conservative full re-verification policy. This is an integrity test of changed semantics, not a method for detecting unseen changes with no oracle access. Cache keys must bind the operator version.

Measured preparation and compilation CPU process time 1.819060301 s, including panel regeneration and conjecture acquisition; policy evaluation times in receipt. No paid credits, no neural training, no historical checkpoint changes. Python formula execution is slower than raw lookup here, so the codec improves payload and independent reuse rather than measured runtime.

## Mathematical transfer condition

If a primitive certificate establishes f_i(x)=g_i(x) for every x in its domain, then any well-typed composition formed from those primitives and exact state-update logic has identical outputs under f and g. Induct on program length: identical initial state, equal primitive outputs on the identical next input, identical state update. A graph revision changing operator assignments is another such composition. This condition covers arbitrary horizons for the certified fixed operators; it does not establish correctness for an unverified or changed operator.

For a finite workload with queried pairs (operator version, input), lazy caching is exact provided first use is verified and versions remain immutable. Its query count equals the number of distinct visited pairs, whereas eager verification charges the sum of all domain sizes. Thus eager certification buys offline coverage; it does not necessarily minimize reads for a sparse workload.

## Replay

Run python state09.py beside state07.py and state08.py with NumPy available (state07 imports torch). The source SHA-256 is 599bf34e486fa7121e4e163d0add11b718b4fd1d47499b3d144798f6b3a032ff; source operator panel hash 49485351e9bcf3ec8b8ab0428c9c060f8d6aee6994000f15c7b0de3427553159. Full local receipt SHA-256 dc3bc2bf81aaf936da88ffce9200bcfda81b2c094f5d906c91add702682c36dd (timing-dependent). Summary CSV in this directory.

## Next gate

STATE-10: investigate transferable proof search rather than merely reuse already certified finite operators. Use Lean kernel-checked obligations with a versioned environment, local context and goal fingerprint. Compare exact proof reuse, verifier-in-the-loop proposal/repair, and a cold search baseline with equal verifier budgets. Test held theorem families and negative transport conditions, distinguish compile/proof time from model calls, and never count an unverified proof as a success. No accelerator expenditure authorized by this finite CPU result.
