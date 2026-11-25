# ALEPH-STATE-01 | causal state propagation journal

2026-11-25. Status: exploratory CPU proxy; no accelerator credits. Retain direct recurrence A; hold both paid decisions. Parent evidence: [CONFIRM14](../CONFIRM14_RESULTS.md), pinned report 78fd4508784afad1cbf9645aca7c0c53172c728e. Native 14.18M checkpoints failed fresh register/graph transfer and solved zero complete repair orbits. Native local-transition competence was insufficient.

## 1. Mathematical task

Let S be a finite state space of size q, U the instruction alphabet, and F_u:S→S the exact transition oracle. For word w=(u_1,...,u_T), write

$$s_t^{(a)}=F_{u_t}(s_{t-1}^{(a)}),\quad s_0^{(a)}=a,\quad \Phi_{j:t}=F_{u_t}\circ\cdots\circ F_{u_j}.$$

An intervention at j substitutes a new verified state a while holding the remaining instructions w fixed. The complete orbit is O(w)={(a,Phi(w,a)):a in S}. For a uniformly sampled initial state and a predictor seeing the suffix alone, the optimal answer accuracy conditional on w is

$$B(w)=\frac{1}{q}\max_{b\in S}|\Phi_w^{-1}(b)|.$$

Proof: every constant prediction chooses one fiber; randomization is a convex mixture of such choices. If Phi_w is bijective, B=1/q and deterministic suffix-only complete-orbit accuracy is exactly zero. If Phi_w is constant, B=1 and an initial-state-blind exact executor can solve the orbit. Therefore separate survival and erasure panels; pooling them conceals state blindness.

For a pair (a,b) define the propagation signature

$$I_t(a,b)=\mathbf 1[s_t^{(a)}\ne s_t^{(b)}].$$

Under a common deterministic suffix I_t is nonincreasing: equal states have equal successors. For a bijective transition it is preserved. A constant reset erases it. In a graph revision, extend S to include graph and cursor; compare a pair whose full states differ, and separately score the observable projection, since equal cursors alone need not imply equal futures.

For affine instructions on Z/qZ, s_t=m_t s_{t-1}+c_t mod q,

$$\Delta_t=s_t^{(b)}-s_t^{(a)}=m_t\Delta_{t-1}\pmod q,\quad \Delta_T=\Delta_0\prod_{t=1}^T m_t\pmod q.$$

Addition has m=1, reset m=0. Composite q also admits partial erasure by nonunits. Example: on Z/4Z, doubling has fibers of size 2, hence B=1/2. Include this partial-erasure regime in a later panel. This separates injective, partially collapsing, and fully resetting operations.

## 2. Trainable objective

Given public instructions and initial state, predict a distribution p_theta,t over verified state classes at each boundary. Use only the public input in the causal model. Define

$$L=L_{end}+\lambda\frac{1}{T-1}\sum_{t<T}\mathrm{CE}(\delta_{s_t},p_{\theta,t})+\mu L_{orbit}.$$

An optional orbit term for affine examples is

$$L_{orbit}=\operatorname{mean}_{a,b,t}\operatorname{JS}(p_t^{(b)},P_{\Delta_t}p_t^{(a)}),$$

where P_d cyclically shifts probabilities by d. With reset Delta=0, require distributional agreement. Uniform distributions satisfy the orbit term, so retain supervised state/answer anchors. This is a proposed objective; the proxy below uses lambda=1, mu=0. Do not interpret a probe or orbit regularizer as a certificate of causal use.

An exact operator view: for a deterministic F let K_F be its column-stochastic pushforward matrix. If learned column-stochastic operators Q_t satisfy sup_p TV(Q_t p,K_t p)<=epsilon_t, the terminal discrepancy is bounded by sum epsilon_t. Proof: insert and subtract K_t times the learned input; stochastic maps are nonexpansive in TV. With contraction coefficients eta_t of K_t the sharper recurrence is e_t<=epsilon_t+eta_t e_{t-1}. A reset has eta=0, deleting inherited error in the ideal transition. This explains why one-step transition quality and reset behavior deserve separate measurements. It does not automatically bound a GRU/transformer with an unverified hidden-state abstraction.

## 3. CPU probe, executed

Use a 2,596-parameter GRUCell executor (24 hidden units; learned one-hot initial-state embedding; seven one-hot instructions; shared four-way decoder). Alphabet is +1,+2,+3 mod 4, or reset to one of four constants. Each batch expands 16 random instruction words into all four initial states; half have one interior reset. Alternate lengths 4 and 8; no hints or previous gold states are ever supplied to the model. Compare terminal CE, terminal plus correct intermediate CE, terminal plus marginal-preserving shuffled intermediate CE. Train matched seeds 137/163/191, matched data RNG, parameter count, AdamW lr=.005, weight decay=.01, one CPU thread. Each held length 12/24/48, each reset regime, uses 128 words (512 answers) per seed; public deterministic seed 2026112501. Train lengths and eval lengths are disjoint, but instruction alphabet is shared. Exact program evaluator supplies all labels.

240 updates per run = 92,160 instruction positions. After seeing the exploratory result, repeated the comparison from scratch at 960 updates = 368,640 positions. Both doses use the same diagnostic evaluation panel; do not treat the second dose as a sealed confirmatory test. Eighteen runs took 20.989 s summed training process time, excluding evaluation and setup, on torch 2.10.0+cpu. Paid compute: zero. The positions are operations, not byte tokens.

| Updates | Length/regime | Terminal answer % | State answer % | Shuffled answer % | Terminal orbits | State orbits | Shuffled orbits |
|---:|---|---:|---:|---:|---:|---:|---:|
|240|12/no reset|25.07|70.25|24.80|0|92|0|
|240|24/no reset|25.00|54.82|25.00|0|13|0|
|240|48/no reset|25.00|51.37|25.00|0|1|0|
|240|48/reset|25.78|64.71|25.52|99|244|98|
|960|12/no reset|98.89|100.00|81.51|367|384|227|
|960|24/no reset|90.82|100.00|75.78|263|384|170|
|960|48/no reset|76.56|99.93|72.92|161|383|150|
|960|12/reset|63.41|100.00|87.37|204|384|324|
|960|24/reset|52.21|100.00|79.36|153|384|282|
|960|48/reset|45.57|100.00|73.63|133|384|253|

Each orbit count denominator is 384 (128 per seed), each answer count denominator 1536. At length 48 no reset, the 960-dose correct-state arm solved 382/384 entire traces across all four starts; terminal-only solved 0/384 and shuffled 128/384. At length 48/reset correct-state solved 384/384 entire traces. Shuffled supervision did learn in some seeds; its seed variability matters. See [per-seed receipts](CPU_RECEIPTS.csv) and [source](state_proxy.py).

Replay from this folder:
    python state_proxy.py > state_proxy_results.json
    python -c 'import state_proxy as s,json; s.STEPS=960; print(json.dumps(s.run(),indent=2))' > state_proxy_960_results.json

This is exploratory evidence that intermediate labels can accelerate acquisition of an algorithm and improve extrapolation in a sequentially biased model. Exact arithmetic can solve the task perfectly. This does not show transformer transfer, byte-language benefit, graph revision competence, an advantage at 1B, or architectural optimality. No confidence claim across program families is supported by three seeds and one alphabet.

## 4. Registered-next candidate (not registered or run yet)

Keep Aleph A, native tokenizer and shared core fixed. Compare terminal, verified intermediate, and shuffled auxiliary loss. Introduce explicit teacher hints with a separate arm only, and withdraw them in training; all promotion evaluations must be hint-free. For latent state supervision attach the same readout in both learned arms at causal instruction boundaries, with the correct label never present in input or visible in attention. Count matched data positions, optimizer updates, auxiliary supervised labels and elapsed compute. Explicitly account for any added readout parameters. Compare 50k, 200k, 1M positions if local budget permits, and freeze the training dose before sealed evaluation. Use fresh training seeds and hold out whole program templates, graph lineages and instruction compositions. Include long additive suffixes, noncommuting permutations, reset after intervention, partial collapse, multiple register dependencies, graph edits whose effect appears multiple hops later, and no-op edits. The easy modulo-additive proxy is a diagnostic, not the final gym.

Measure terminal accuracy, complete survival-orbit accuracy, erasure/no-op correctness, full trace accuracy, and the frozen suffix cue. Report each family and seed, bootstrap by whole orbits/lineages and resample seeds for sensitivity. Test native 14M pass-specific readouts and carefully matched activation interventions with same suffix; ask whether the decoded correction causally propagates rather than simply being readable. Include sham patches, unchanged-state cases, and donor state from a disjoint program prefix. A useful auxiliary readout is evidence only; verify the downstream effect. Require > strongest registered heuristic on each main fresh family; proposed floors of 75% register and graph, 50% complete survival orbits, subject to registration and power. Evaluate byte text non-regression before any model-scale decision.

For DataGoblin export a verified state-boundary record: program_id, family_id, lineage_id, split, instruction prefix and suffix, initial state, verified intermediate state labels stored separately from input, counterfactual group ID, intervention location, terminal answer, verifier/version hash, reset/erasure signature. Lean proof states may be canonicalized only under a sound representation; use actual kernel-checked tactic transitions, preserve dependency context and universe/local binder information. Keep equivalent traces and sibling counterfactuals within one split. Do not add held diagnostic samples to pretraining.

Budget decision remains unchanged: A reference; B unpromoted; no paid 10B slot awarded by this proxy.
