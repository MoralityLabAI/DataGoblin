# ALEPH-STATE-04 | externalized recurrent state

2026-11-25. A reference fixed, zero paid credits, no native checkpoint changed. Follow-up to STATE-03. Exploratory CPU diagnostic; feedback and calibration were designed after initial outcomes, not independent confirmatory tests.

## Distinguish a learned transition from its composition

Let M_theta(s,u) be the argmax of the four state logits of the direct Aleph model on the compact input [REPAIR, SEP, s, u+32]. For a word w=(u_1,...,u_T), define the externalized execution

$$\hat s_0=s_0,\qquad \hat s_t=M_\theta(\hat s_{t-1},u_t).$$

The ordinary contextual executor instead predicts g_theta(s_0,u_1,...,u_t) at each instruction boundary from a single causal transformer computation. These are different algorithms, with different compute and memory costs.

**Finite-alphabet closure lemma.** If M_theta(s,u)=F_u(s) for every pair (s,u) in a finite state set S and instruction set U, then the externalized executor is exact for every finite word over U and every initial state in S. Induction on t: the initial states agree, and equality at t-1 plus correctness of the corresponding table entry gives equality at t. This also preserves all orbit partitions. Exhaustively checking a 4×9 transition table certifies this restricted claim. It does not certify correctness of a contextual transformer forward pass or generalization to new instructions, unbounded states or Lean.

If only a subset of rules is verified, every word whose exact reachable transitions belong to that subset is certified. For broader state spaces, one can use an independently verified intermediate representation and a verifier-controlled fallback on uncertified transitions. Report verifier calls, per-step model latency, abstentions and error propagation. For this tiny alphabet the model predictions can be cached; a symbolic transition table is an exact and substantially cheaper baseline. The value of the study is identifying an acquisition/assembly gap in the neural model.

## Fixed-dose experiment

Nine width-24 direct Aleph miniature fits from scratch; 136392 physical trainable parameters, 14 blocks, 24 visits, four heads, one KV head, 512-symbol vocabulary and tied projection. The sandbox uses a direct-only transcription of the pinned gym.py and study.dense architecture (included as direct_excerpt.py); default repo CLI imports the original gym.py. All three arms use the same architecture and matched data stream within each seed. Seeds 229/233/239; 384 optimizer updates, AdamW .001, weight decay .01, gradient clipping 1, one CPU thread per process. Six words per update expanded over all four initial states; train lengths 4/8, one of three regimes per two words. Input positions 82944 per fit, including headers and starts. Terminal-only CE vs .5 terminal + .5 verified state CE vs .5 terminal + .5 shuffled state CE (marginals preserved). No gold intermediate states in the actor input.

Held panel: public RNG 2026112505; 32 words per cell, lengths 12/24/48, permutation, partial, reset, and double partial (the latter absent from training). All four initial states per word. No held words included in training; same finite instruction alphabet. 178.019929 summed training CPU process seconds (evaluation excluded), torch 2.10.0+cpu. Source state04.py SHA256 692dfeff18c09beac684dc1b45770c28d6127eddf2a05677f3ede020605db734; direct excerpt 0b4d4aeced38d5330707b34b0092cbc28aa9fe092b473f6440f7f419216e09b3.

At the initial 384-update endpoint, full contextual inference solved zero complete permutation orbits across all three seeds and all three lengths in each arm. Verified state supervision raised the pooled trace-hit count over all lengths/regimes from 33876/129024 (terminal) to 51238/129024 (state), versus 34846/129024 (shuffled). On the 48-step permutation panel, state-arm accuracy at first eight positions was 2078/3072, later positions 3837/15360. The local rule was much stronger than the contextual long-horizon prediction.

Using the final tied projection (untrained as an intermediate probe) at core block 10 after the first and second visits gave state-arm trace hits 48341 and 50424 /129024, versus 51238 after the final blocks. Terminal-only trace hits 34380, 33926, 33876. These are descriptive readout measurements, not causal evidence that the second pass implements exact transition composition.

## Post-hoc predicted-state feedback diagnostic

Query the trained network on all 36 legal one-step pairs and cache its predicted states (no gold targets used in rollout). Correct table entries by seed: terminal 15/13/9; state 34/35/32; shuffled 14/16/11. Each state model correctly learned all 12 permutation entries. Feed only the predicted state to the next rule.

| 48-step family | Terminal direct /96 | Terminal feedback /96 | State direct /96 | State feedback /96 | Shuffled direct /96 | Shuffled feedback /96 |
|---|---:|---:|---:|---:|---:|---:|
| permutation |0|0|0|96|0|0|
| partial |0|0|0|20|0|0|
| reset |22|23|17|88|20|23|
| double partial |5|9|7|38|6|6|

Complete orbit counts; pooled 3 seeds. State feedback solved 96,20,88,24 complete traces respectively; full-context direct solved zero complete traces in these 48-step cells. One-step evaluation assesses the known instruction alphabet, not held transition rules. Feedback was designed after observing the first endpoint and is an exploratory diagnostic on the same public held panel.

## Post-hoc one-step calibration

For every original endpoint, restart AdamW at .001 and apply 64 updates using the entire 36-entry oracle table as training data, 9216 additional token positions and 2304 supervised labels per fit. The exact table is a training-set closure check, not an independent test. Report fixed milestones, no early stopping.

| Original curriculum | Initial correct /108 | +8 updates | +16 | +32 | +64 |
|---|---:|---:|---:|---:|---:|
| Terminal |37|60|79|103|108|
| Verified states |101|108|108|108|108|
| Shuffled |41|82|94|108|108|

All nine reached 36/36 individually at 64 updates. At that point all nine achieved perfect feedback execution on every evaluated length and regime, including all traces, by construction of exact local table closure. Full-context permutation orbit count at 48 remained zero in all nine. The new training took 26.639542 CPU process seconds; total STATE04 training 204.659471 seconds. The evidence supports faster local rule acquisition from verified states, and an assembly failure in this small contextual transformer. Since all arms solve the local table after a short explicit calibration, the state curriculum has not demonstrated a distinct terminal competence ceiling. No extrapolation to a 1B model or paid placement.

## Next native experiment

Freeze a new panel of composed multi-register programs and graph edits with distinct held families. Compare native direct answer-only, verified intermediate state supervision, and explicit local transition curriculum (matched optimizer/label budgets) with two inference modes: contextual and verified/predicted-state feedback. Include no-op edits and dynamic graph context; state representations must retain sufficient information to make the next transition Markovian. Lean proof-state feedback needs the full verified goal/context or an independently certified sufficient abstraction; never replace a proof state with a lossy four-class analogue and call that sound. Track error recovery, verifier calls, token/forward cost, complete orbit/trace correctness, and native text/Lean regression. Include the exact symbolic executor as a control; for the finite toy task it is perfect without learning.

Use native_continue.py only after the historical direct parents and original corpus have been located and verified. The actual 14M continuation remains unrun. Neither paid slot is authorized.

## Replay

In this directory, run each seed and arm with state04.py --seed SEED --arm ARM --out out --standalone (omit --standalone for original repository gym). Then state04_diagnostics.py --checkpoint out/SEED_ARM.pt --out diagnostic.json --standalone, and state04_calibrate.py --checkpoint out/SEED_ARM.pt --out cal --standalone. See STATE04_RECEIPTS.csv for per-seed local and orbit counts. Full local combined JSON SHA256 57fbde5b02608b579a9053746c522410a089d120bb2b0e15233c92779a448ced; serialized weights and timing hashes may differ by environment.
