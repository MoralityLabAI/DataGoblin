# ALEPH-STATE-11 | Learned rewrite support versus symbolic search

2026-11-25. Exploratory CPU-only continuation of STATE-10. A remains the reference; hold paid training. The local sandbox lacks Lean, lake and elan; release download failed DNS resolution. These are synthetic Lean-core-style Nat/List equality obligations with a restricted rewrite certificate checker. No kernel-verified labels are authorized. No historical model weights or native LDT were used.

## Design

Generate 120 proof trajectories in each of four groups: Nat structural + identity, Nat distributive + identity, List append + reverse, List append + map. Reserve 32 mixed Nat and 32 mixed List trajectories, requiring both structural and distributive rewrites for Nat and both reverse and map rewrites for List in the generating path. Train and held exact ordered equation pairs are disjoint; semantically equivalent equations and shared subterms may occur, and alternative proofs can bypass the required generating-path combination. This is a held composition diagnostic, not a sealed split of real Lean theorem families.

All policies use the same 19 oriented rewrite schemas, matching subterm substitutions and emitting successor terms of size at most 55. Breadth-first priority is path depth. Symbolic priority is depth + syntactic goal distance/4, where distance sums aligned subtree mismatches and charges max subtree size for a mismatched root. Learned priority subtracts 0.8 times a learned edge score. Each newly reached expression is enqueued at most once; return on generating the exact target. This is bounded search, not a complete decision procedure or necessarily shortest-proof search.

Train three 2,833-parameter rankers, seeds 293/307/311, 240 AdamW updates, batches of 128, LR .003, on 13,934 candidate edges (1,817 positive planted-path edges). Features include expression sizes, syntactic distance, rule ID, position depth, token-count differences and exact successor equality. All arms have exact target matching and enumerate applicable rewrites symbolically. No negative edge is assumed mathematically false: it merely differs from the planted next step. Some candidates in training states use rules from other groups. No claim of entirely withheld individual lemmas.

Budgets are expanded expressions (4,16,64). The same maximal search trace supplies nested budget results. Generated unseen successor count is separately reported, not kernel calls or equal FLOPs. Cached expression matching and fixed policy order affect timing.

## Executed results

| Arm | @4 /64 | @16 /64 | @64 /64 | Expansions @64 | Edges @64 |
|---|---:|---:|---:|---:|---:|
| BFS |13|32|46|1930|14019|
| Symbolic |31|57|61|695|7125|
| Learned 293 |30|56|61|654|6949|
| Learned 307 |28|56|60|681|7126|
| Learned 311 |30|58|61|661|7046|

All policies solved all 32 List equations at 64 expansions. Nat: BFS 14/32, symbolic 29/32, learned 29/28/29 of 32. Learned arms reduce total expansions by 2.0–5.9% compared with symbolic at the cap, but coverage is never higher there. No promotion; retain symbolic as the reference. The measured evaluation CPU times (BFS/symbolic/learned) are .345/.134/.266/.163/.141 seconds, and the three fits consume .400 CPU seconds combined. Total main experiment 2.403252314 CPU seconds, including generation, search and output work up to receipt creation. Torch 2.10.0+cpu, Python 3.13.5. Timings are exploratory and cache/order-confounded.

Each found proof path is checked by replaying the declared rewrite schema and outer context at every step, using the same matching/substitution routines as the generator. Generate 353 explicit Lean declarations (64 planted + 289 found paths) and 64 invalid proof transports. A separate Lean replay mode compiles the positive file, audits the axiom sets and requires invalid transports to fail with a type mismatch. It has NOT been executed. Lean lemma availability, elaboration, and diagnostics may need toolchain-specific fixes.

## Independent denotational audit

The separate state11_normal_forms.py reconstructs the exact held panel by hash and independently normalizes the training and held generating paths. For Nat, represent an expression by a finite nonnegative coefficient map over exponent triples; addition is pointwise addition, multiplication is convolution. Induction on expressions proves evaluation preservation. Equal maps imply equality under all Nat assignments.

For List, a normal form is a sequence of (variable block, reversal parity, ordered map stack) triples. Append concatenates blocks, reverse reverses block order and toggles parity, and map appends an operator to each block's stack. Associativity of append, reverse involution, and map distribution/commutation with reversal give preservation of evaluation. Equal normal forms suffice for equality for arbitrary list and function assignments. This is a denotational check in the declared grammar, not a Lean proof object.

All 1,817 training and 245 held generating transitions preserved their respective normal form; all 480 training and 64 held endpoints agreed. All 544 deliberately modified target equations had different forms (Nat +1 or append an independent list variable). The latter mutation need not differ on every valuation, but does change the denotation as a function of assignments. Audit CPU .052223823 seconds, excluding reconstruction. The independent check validates the planted trajectories and endpoints, not each alternative search proof's intermediate states.

The useful next baseline is proof-producing normalization, followed by bidirectional rewrite search for goals where no convenient normal form exists. Charge proof generation/checking and retrieval costs separately. Use actual Lean with a pinned toolchain and prospectively held, non-synthetic families before any training-data admission or deployment decision. The current learned ranker has no advantage over the symbolic policy at matched cap.

## Replay

From this directory:
    python state11.py --out .
    python state11_normal_forms.py
    python state11.py --out . --replay-only

The final command requires Lean on PATH. The first command emits proxy receipts (lean_kernel_verified=false), generated positive and negative obligations; separate successful Lean replay is required to establish kernel certification. Source SHA256 54b35b6118e1333c84bc6bea9340d228d0c632abe0916f0e6f8c61141282333e; source blob 358d4e79dcd270aa2c6561353daaa80c9f018ab0 matches the executed code. Held panel SHA256 96024ba32e893fac2b657278a90fd3be38a209e4d0553b1029df8b68d0311ce8. Generated Lean file SHA256 9702ca64b5a04240fc6043c53ad1a71f41e164e745d725bf3e335972f48f78d6. Audit source SHA256 40a18134f89d5c2291226a0c9ec1e4837896a7ea29fb9c3578507ae1e98ad22b. No paid credits spent.
