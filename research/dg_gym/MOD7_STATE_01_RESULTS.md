# MOD7-STATE-01: finite-state learning, state survival and certified hardening

Public development only; no paid compute, sealed outcomes, family-transfer
claim or training/corpus admission. Registration commit
d82c81159d66779431107927898fed3e35337a51. Exact executed source SHA-256
1f55ae58667bbb937e9441e2c8115ee628e5fb987c58e1cf5c729d8a474c6424
(Git blob 742e3d0b32e99f2f2b7f3f42fa7bfb0917858c11).
PyTorch 2.10.0+cpu, one CPU thread. Training FP32, evaluation FP64.
Nine fits, each 2058 parameters, 600 updates, 57,600 root draws,
158,400 recurrent example-visits. Equal exposure, not equal wall time.
The 1/2/4 length curriculum includes direct one-step task labels, and the
canonical state coordinates are provided by the architecture.

## Observed results

Every fit passed 294/294 row argmax and 512/512 length-six soft accuracy.
Hard execution passed 512/512 at every length for every fit. Soft outcome
counts and mean probability mass at lengths 6/32/128/512 follow.

| Seed | Arm | Correct counts /512 | Mean target mass | Worst row TV | Train s |
|---|---|---|---|---:|---:|
|11|task|512/512/512/512|.982583/.910598/.694905/.290430|.00509325|.372792|
|11|concept|512/512/512/512|.985927/.927152/.744013/.350343|.00408409|.454163|
|11|shuffled|512/495/75/70|.568393/.163781/.143112/.142516|.16321206|.541185|
|23|task|512/512/512/512|.982273/.909764/.692326/.287407|.00437003|.369370|
|23|concept|512/512/512/512|.985679/.926468/.741758/.346921|.00352304|.449721|
|23|shuffled|512/493/70/70|.569227/.163769/.142380/.143155|.15711140|.465339|
|37|task|512/512/512/512|.982417/.910001/.692675/.287800|.00504927|.356743|
|37|concept|512/512/512/512|.985796/.926661/.742065/.347393|.00407177|.441112|
|37|shuffled|512/497/73/67|.568168/.163576/.142684/.142783|.14875713|.445157|

State supervision reduces the worst row TV error in each paired fit.
At length 512 the mean correct-state mass is .288546 vs .348219; the
gain is .059673, with equal argmax accuracy. Shuffled targets yield .142818.
Across the same three training seeds, donor/identity/wrong-donor/oracle
counterfactual correct counts at length 32, after replacing the state at
visit 16:

|Arm|Donor|Identity|Wrong donor|Oracle|
|---|---:|---:|---:|---:|
|Task|1536|0|0|1536|
|Concept|1536|0|0|1536|
|Shuffled|1483|9|11|1536|

Mean donor target mass .910067/.926720/.163442 respectively. This is
oracle-assisted coordinate replacement in an explicitly structured state;
it does not establish discovery of a concept in a general transformer.
The nine fits share the same public development roots, so pooled counts
are descriptive, not independent replicates for uncertainty estimation.

Median training wall seconds .369370/.449721/.465339. Timings include the
training loop and data generation, excluding optimizer initialization and
evaluation. Per-fit RAM peak and GPU measurements remain unavailable.
A repeated run reproduced all output metrics and checkpoint digests.
No equal-wall-clock superiority claim.

## Mathematical extension

For row-stochastic P_i approximating the exact transition P_i*, suppose
epsilon is the maximum TV error of every learned row versus the exact delta.
Stochastic maps are TV-nonexpansive. Telescoping the product proves

    TV(p_K, p_K*) <= min(1, K epsilon).

With an exact initial state the probability of following every correct
transition is at least (1-epsilon)^K. Other paths may also reach the correct
terminal state, hence p_K(correct) >= (1-epsilon)^K. The pilot checks both
inequalities numerically at each evaluated depth and root. A sufficient
condition for unique correct soft argmax is (1-epsilon)^K > 1/2.
Using the worst observed row error among three fits gives sufficient
horizons 135/169/3 for task/concept/shuffled respectively; these are
conservative bounds for those finite matrices, not confidence intervals.
The row error estimates use FP64, not certified interval arithmetic.

A complementary state-survival descriptor is the Dobrushin coefficient

    delta(P) = max_{r,s} TV(P[r,:],P[s,:]).

For any distributions p,q, TV(p P, q P) <= delta(P) TV(p,q).
Proof: decompose p-q into equal-mass positive and negative parts and
bound their normalized output mixtures by the maximum row-pair distance.
Repeated application gives the product bound. Positive softmax matrices
have delta < 1, so distinguishability of initial states decays in long
soft products. This is compatible with exact hard row argmax and with
correct terminal argmax despite low target mass.

## Certified hardening

Define H_i(r)=argmax_s P_i[r,s], with the specified deterministic tie policy.
For the finite environment, exhaustively check H_i(r)=P_i*(r) at all
42 x 7 elementary pairs. Given the canonical initial state and admissible
instructions, induction proves correctness of H_{i_K} o ... o H_{i_1}
for arbitrary finite K. This certifies the derived lookup policy, not the
soft neural computation. All nine exported checkpoints yield the same
hard table; an independent NumPy check decoded each from raw float32
weights and verified all rows. Packing 294 entries at three bits each
requires 882 bits, or 111 bytes including final-byte padding; unpacking
was checked. Table payload excludes metadata, verifier and interpreter.
It is not a generic neural compression ratio.

For a fixed instruction sequence, define composition of maps (g o f)[r]
= g[f[r]]. It is associative. A balanced reduction constructs its full
seven-state map in O(7 K) work and O(log K) parallel span with sufficient
processors and independent composition jobs. Sequential single-state
execution is O(K) lookup work and O(1) working state; materializing the
maps costs O(7 K) memory in the straightforward parallel scheme.
These are abstract work/span bounds, not a measured hardware speedup.
A chunked reduction can trade span against memory under the DG-Gym
reservation calculus.

## Checkpoint identities

The exact local receipt and raw little-endian float32 tensor payloads are
in the companion v0.4 archive. The raw payload hashes identify tensors
with shape (42,7,7), not complete optimizer checkpoints.

| Seed | Arm | SHA-256 |
|---|---|---|
|11|task|8027d5179b0af247f2c02027d63183dd09d12622e6a5ddf7b0b661aa95ec914b|
|11|concept|69358afad22fbe152b7aad1db5fd2a27d66eb982a4d63ab849254d1aa9e08047|
|11|shuffled|fdaa5f4ebb9401075923e81989448f6b19762efa0cb1e765e4c54d2d06968d22|
|23|task|940bddebc7a2239166f3bcd37d94e1ef46f4e7886144b51fdf3e15a82b856793|
|23|concept|2a048dc150f670b5b4008b8777d4e4eb287cfaa77dd279838544f954968ec593|
|23|shuffled|8bd3d9e2ed2f5adde89e7641eb4ffd43bc5e3d8adc271618a59836d0bbbf48b3|
|37|task|8b5cdcc037825935a8e9996608562b93bb7372e488475564610ecab97c680200|
|37|concept|1be9f84de72458aa2f486eedd377c1b28440eb5baeeedbb145abb908c5bfc24b|
|37|shuffled|6e75125945ffc8ac8c5cce20c736f2a907c61d2f68714b43789d7de176a72e46|

Disposition: retain the structured reference and its exact hardening
certificate. Concept supervision improves soft state survival at matched
exposure, but adds training work and does not improve measured ordinary
argmax accuracy. Next: benchmark the parallel composition schedules on
the same learned hard table under RAM and CPU caps, and hold out
instruction families when testing factorized transition architectures.
