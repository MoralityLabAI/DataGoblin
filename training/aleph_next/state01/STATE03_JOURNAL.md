# ALEPH-STATE-03 | cross-prefix causal interchange

2026-11-25. Zero paid compute. A remains the reference. This follow-up is diagnostic, not a preregistered confirmatory study.

## Causal abstraction

Let the neural executor have hidden transition h'=G_u(h), decoder D(h), and exact state transition F_u(s). For a suffix w define G_w and F_w by composition. A candidate state abstraction alpha is sufficient on a set of reachable hidden states if alpha(G_w(h))=F_w(alpha(h)) for every allowed w. The one-step commuting condition implies the multistep condition by induction. This is a behavioral criterion; a finite panel cannot establish it universally.

For independently generated recipient and donor prefixes reaching exact states a and b at boundary t, retain recipient suffix w and substitute donor h_b. Define target y_b=F_w(b) and original y_a=F_w(a). The surviving panel has y_a != y_b; the erased panel has y_a=y_b. Always include both. For full interchange the ideal output is y_b. Same-state donor replacement is a nuisance-invariance check; a wrong-state donor is judged against its own exact output, which can coincide with y_b when w is noninjective.

For a linear decoder W with q class rows, let C=I-11^T/q, V=row(CW), and P the orthogonal projector onto V. The centered-logit contribution of the complement vanishes: CW(I-P)=0. A readout patch is h_a+P(h_b-h_a); its centered logits at the intervention boundary equal the donor's. Subsequent correctness is an empirical test of dynamics; it does not follow from logit equality. Complement patch is h_a+(I-P)(h_b-h_a). Random control projects into an independently seeded subspace of equal rank, then rescales each perturbation to the readout-patch norm. The decoder subspace can be off-manifold when combined with a recipient complement.

A stronger causal abstraction would require interchange on disjoint prefixes across state families, preserve correct outputs under equivalent-state donors, and obey partition coarsening from STATE-02. A full GRU hidden replacement changes the entire recurrent state and provides an upper comparator; transformer residual patching at a single position is not equivalent because other prefix tokens and suffix representations carry information.

## Executed proxy

Replayed the nine STATE-02 fits using the same public training seeds, initialization and dose (960 updates, 414720 instruction positions per fit, GRU 2740 parameters). New public diagnostic seed 2026112503. For each cut (4 or 12) generate 128 independent pairs of equal-length distinct invertible prefixes. Expand all four initial states; each recipient has three changed target states and a uniformly sampled wrong donor among the two remaining states. Match donor initial state to its oracle boundary state; use an independent prefix, same suffix for all patches. Tail length 24, four suffix regimes (permutation, partial, reset, double partial). All donor/recipient boundary states are reachable. For every cell report all-case counts and the subset where the original output was correct; conditional denominators vary by arm. No examples from the new panel are added to training.

Nine fits took 17.0069 s summed training CPU process time, torch 2.10.0+cpu; intervention evaluation time excluded. Decoder centered rank 3 for the fits. Source [state03.py](state03.py), local SHA256 49c70a92f3e908c57a7e6eddddc94641f45531da601f94dd7ffcbbf85ac06d00. Replay: python state03.py > state03_results.json. Full local JSON SHA256 4386f4a36806fac294bb80efa48ca0250f18f6a2d6d13d09225738c77ee3be34 (timings vary).

### All-case surviving, cut 12 (common denominator 9672 per arm)

|Arm|Original correct|Full patch target %|Same donor target %|Wrong donor target %|Readout target %|Random target %|Complement target %|
|---|---:|---:|---:|---:|---:|---:|---:|
|Terminal|9663|99.886|0.052|26.189|99.225|14.816|0.527|
|Verified state|9650|99.783|0.072|26.241|99.690|15.436|0.114|
|Shuffled|9530|98.284|0.517|25.641|88.947|10.205|9.057|

### Conditional counts (unpatched correct), aggregated across 3 seeds and 4 suffix regimes

Counts are target hits for each patch; same-state and wrong-state donors are scored against the changed target for this table, not their own labels. The last column is exact wrong-state oracle agreement with the changed target (collision ceiling).

|Cut|Regime|Arm|n|Full|Same|Wrong|Readout|Random|Complement|Wrong oracle|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|4|surviving|terminal|9784|9776|0|2566|9717|1526|23|2569|
|4|surviving|state|9782|9776|0|2565|9766|1571|4|2569|
|4|surviving|shuffled|9666|9538|2|2448|8419|672|1073|2504|
|4|erased|terminal|8636|8632|8636|6050|8611|8262|8620|6048|
|4|erased|state|8635|8634|8635|6051|8633|8250|8632|6048|
|4|erased|shuffled|8301|8004|8290|5522|8050|8151|8229|5772|
|12|surviving|terminal|9663|9653|4|2533|9588|1431|50|2532|
|12|surviving|state|9650|9631|3|2535|9621|1485|4|2530|
|12|surviving|shuffled|9530|9372|20|2420|8487|948|845|2470|
|12|erased|terminal|8754|8749|8753|6219|8719|8383|8717|6222|
|12|erased|state|8758|8758|8758|6228|8753|8366|8754|6228|
|12|erased|shuffled|8413|8089|8376|5701|8136|8179|8322|5949|

The answer-only GRU learned a causally effective readout subspace already. Verified state supervision markedly improved trace fidelity in STATE-02, but only improved readout-interchange target accuracy by about 0.47 pp on the common surviving cut-12 panel; full interchange slightly favored terminal-only. Shuffled state supervision degraded this mechanism. These are descriptive results; each seed and repeated word are dependent. No claim of a causal benefit on native Aleph or of statistical significance. This is why a complete state trace must accompany terminal accuracy and an interchange test.

## Native implementation

[native_interchange.py](native_interchange.py) adds direct-only hooks at a specified shared physical block (1..10) and visit (1 or 2). Donor and recipient have aligned prefix lengths; patch either the boundary token or all prefix positions. Other recipient representations remain untouched; failure does not falsify a distributed state mechanism. Same-state donor, wrong-state donor and identity patches included; score both survival and erasure and all vs base-correct. Run the CLI with --checkpoint and --sha256 to use an explicit A parent. The default four words per regime is a small diagnostic; a new sealed family panel is required for promotion.

Executed random-init 14,179,584-parameter CPU smoke on local direct-only transcription of native Block/Aleph (not the repository checkout). At physical block 10 both pass-specific identity patches left every logit unchanged, and an altered token left earlier logits unchanged (max 0.0), with later max changes 1.1055629253 (visit 1) and 1.2126305103 (visit 2). This checks hook placement and causality, not learned semantic interchange. Source SHA256 9759de2b1e69551443c0ddbbeb52c0de3e6dc162a309bd4e93746065563aa77c. Original trained checkpoints unavailable.

[native_continue.py](native_continue.py) supplies a proposed mixed continuation runner: immutable SHA256-checked direct parent, original five native lanes reweighted by 0.8, same public state packet for matched arms (0.2 lane mass, terminal-only uses all of lane on final loss; auxiliary arms split it equally), matched optimizer restart, bounded one-thread CPU, explicit dose of 128/512/2048 updates (one-update smoke option), separate output folders, receipt and endpoint SHA256. This code has not been executed against the repository's historical corpus/checkpoints; run one-step integration first. Historical weights and pack are not available in sandbox. The public diagnostic is exploratory; do not use it as promotion data. The runner does not implement checkpoint resume within an invocation, or native text/Lean regression evaluation; use original independent evaluators and a new panel before any decision. No paid training authorized.

Example from repo checkout with exact local parent paths/hash:
    python training/aleph_next/state01/native_continue.py --checkpoint A.pt --sha256 ACTUAL_SHA256 --pack runs/placement/mixed --seed 137 --arm state --steps 1 --out ../state03-smoke-137
    python training/aleph_next/state01/native_interchange.py --checkpoint A.pt --sha256 ACTUAL_SHA256 --physical 10 --visit 1 --words 4

Freeze matched dose, panel and native text/Lean/regression and orbit gates before a true continuation trial. Keep A unchanged until independent competence is established.
