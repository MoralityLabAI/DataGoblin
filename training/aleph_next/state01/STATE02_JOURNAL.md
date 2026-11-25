# ALEPH-STATE-02 | partition-valued causal state journal

2026-11-25. Follow-up to [STATE-01](JOURNAL.md). Zero paid credits. A remains fixed; neither paid slot authorized by this result.

## Mathematics: the quotient of an initial-state orbit

For a common deterministic instruction word, let f_t=F_t∘...∘F_1. Define the kernel partition Pi_t of the initial state space by a~_t b iff f_t(a)=f_t(b). The relation only grows: Pi_{t+1} is a coarsening of Pi_t. Proof: equal inputs have equal deterministic successors. In particular the transformation rank r_t=|im f_t| never increases. Permutations preserve every equivalence class and reset collapses all classes. A partial erasure can merge some classes while leaving others distinct.

For initial state A uniform over q values, S_t=f_t(A), the fiber distribution is p_t(b)=|f_t^{-1}(b)|/q. Conditional on the word, H(S_t)=I(A;S_t), since the execution is deterministic. H(S_t) is nonincreasing because S_{t+1} is a function of S_t. The exact word-only answer ceiling is B_t=max_b p_t(b)=2^{-H_infty(S_t)} when entropy uses bits. Rank alone is insufficient when the fibers are unbalanced. In the balanced rank-2, q=4 case H=1 bit, B=.5. A model may retain irrelevant initial-state information in hidden features even after the output partition collapses; this statistic concerns the executable state, not the entire hidden vector.

For each pair a<b, record E_t(a,b)=1[f_t(a)=f_t(b)]. Score the whole six-entry pair signature at every boundary and the exact complete orbit separately. Equal partitions permit a permutation of class labels, so partition accuracy can be high when exact states are wrong. Full trace accuracy is stricter than both. A causal activation intervention is needed to establish that a decoded state drives subsequent outputs. For arbitrary maps, unlike affine shifts, a JS orbit term should compare distributions after applying the exact pushforward, rather than imposing an unsupported cyclic shift.

Two successive rank-2 maps can compose to rank one or rank two depending on the intermediate permutation and fibers. Thus counting reset tokens or counting lossy instructions cannot determine the correct partition. This yields a cheap held-composition test for causal information survival.

## Fixed-dose CPU diagnostic

Source [state02.py](state02.py), SHA-256 e4e1228456a80ca5750cbb39d31661b338359c80416e2282bb6ca3aaffb77112. Fixed before executing this new panel; exploratory, not a sealed or independently preregistered trial.

Three permutations: shift, transposition and reversal (first two do not commute); two balanced rank-2 maps; four constants. All four starts expanded for every word. GRUCell, 24 hidden units, 2740 trainable parameters. No gold intermediate states in model input. Train on lengths 4/8; exactly one partial erasure, one reset, or only permutations. All three regimes represented equally. Held lengths 12/24/48, with a fourth regime containing two partial erasures and no resets, absent from training. Seeds 211/223/227, independent evaluation RNG 2026112502, shared deterministic panel, 128 words per cell, 960 updates, 414720 instruction positions per run. Matched streams and AdamW .005, weight decay .01. Terminal-only vs terminal+correct intermediate CE vs terminal+shuffled intermediate CE. Auxiliary loss weight one. Shuffling preserves per-position label marginals. Nine runs; summed training process time 16.946s, CPU torch 2.10.0+cpu.

|48-step regime|Terminal-only answer /1536|Verified state answer /1536|Shuffled answer /1536|Terminal-only full trace /384|Verified state full trace /384|Shuffled full trace /384|
|---|---:|---:|---:|---:|---:|---:|
|Permutation|1533|1534|1534|381|382|128|
|Partial|1534|1536|1501|205|383|1|
|Reset|1527|1532|1473|324|380|2|
|Double partial|1536|1534|1494|155|383|1|

Terminal complete-orbit count in the double partial regime is 384/384, 383/384, 357/384 respectively. Full trace partition counts are 158/384, 383/384, 9/384. Thus the effect is on complete execution and state equivalence across time; terminal accuracy was already saturated. For double partial at length 48 the 128 words comprise 34 rank-one and 94 balanced rank-two maps, so an exact suffix-only cue can get 324/512 answers and 34/128 orbits. Full-trace counts separate the models even when terminal predictions agree. The full three-seed, all-length counts are in [STATE02_RECEIPTS.csv](STATE02_RECEIPTS.csv); the full local JSON also records oracle bounds and ranks.

Replay:
    python state02.py > state02_results.json

No held cases added to training. Shared generator and finite alphabet limit external validity. No inferential confidence intervals or claimed 14M transfer. The shuffled control performed surprisingly well at final answers and is essential. State labels improve intermediate readout, but this experiment does not distinguish use of the readout from latent computation.

## Native direct A bridge

[Native adapter](native_bridge.py) consumes compact pilot IDs [REPAIR=260, SEP=262, start, op+32,...]. Reads logits for the four state classes at the position immediately after reading each instruction (i.e. logits at that instruction's token index). Gold states are detached targets supplied to CE, never actor inputs. Use existing tied vocab projection; zero added model parameters. Input IDs 32..40 are pilot-only; do not mix them into the text stream without checking namespace collisions. For production encode explicitly named instruction tokens and a validated sidecar keyed by root and instruction boundary.

The actual native trainer anchor14_train.py has a five-lane MASS budget and selects restricted choice logits at the last query; add an isolated state lane using the new loss, hold other lane masses and sample streams fixed across arms, and normalize the state lane by row counts and state loss by supervised boundaries. Do not silently use the existing checked_response packed mask for these internal labels: packing.py disallows checked target masks in the prompt; the causal query labels here are detached from the actor token sequence. If using PackedStream, version its schema and verify detached label provenance, boundary offset, segment identity and no label across a segment cut. A separate explicit trace-generation lane is a distinct experiment. Keep the existing 14M A checkpoints immutable. Prefix weights and historical pack are required to do actual matched continuation.

Executed sandbox smoke against a local direct-only transcription of the pinned gym.py Block/Aleph and study.dense(256) (network checkout unavailable). 14,179,584 parameters, 88 input positions, 8 terminal labels, 56 auxiliary labels, initial loss 3.8958284854888916, finite preclip gradient norm 131.9107208251953, all parameters got gradients, one AdamW update, max prefix-logit change after perturbing the future instruction suffix 0.0. This is a single random-initialization gradient check, not a native checkpoint run or a learning result. The committed adapter imports the original dense implementation in a real checkout:
    python training/aleph_next/state01/native_bridge.py
Native bridge SHA-256 d60b8c6e87b7455e795884600a661b434cd2f160c03e7d8ca27a56541675d0eb.

## Next gate

Freeze data, lane weights, counts and a new family-level test panel before continuation. Start 3 matched arms from each exact A checkpoint and matched optimizer initialization; test 128/512/2048 updates (or separately registered token horizons) on bounded local hardware. Report original text and checked reasoning regression and fresh full traces, partitions, terminal orbits, and exact cue thresholds on register, graph and Lean families. Add an oracle-independent verifier and causal hidden-state patch/sham controls. Reject gains confined to the readout or local dev family. Do not open a paid 10B run on this evidence.
