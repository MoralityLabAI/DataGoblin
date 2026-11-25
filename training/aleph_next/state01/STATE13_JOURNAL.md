# ALEPH-STATE-13 | Cost-aware reasoning routing and congruence decomposition

2026-11-25. Free CPU diagnostic following STATE-12. Direct Aleph A remains the reference; no paid model training or native checkpoint changes. No Lean executable is available in the sandbox. All accepted paths pass the restricted rewrite replay and independent denotational audit; none has a Lean kernel receipt.

## Design and costs

Seed 2026112519. Reconstruct and exclude exact ordered STATE-11 and STATE-12 equations and their reversals. Generate 96 positive training equations (shallow walks up to three steps or short local walks under two shared context layers; 24 per domain/template), with 32 negative mutations. Hold out 96 positive equations (16 per domain and template: longer walks up to seven steps, a local edit under seven shared context layers, and two changed subexpressions joined by addition or append), and 36 negative mutations. Randomly orient positives. No exact equation/reverse pair crosses splits; shared primitives, denotations, subterms and equivalent problems can recur. Synthetic public split, not an independently sealed Lean theorem cohort. All routes share a semantic gate in the restricted Nat/List grammar; unequal forms cause abstention. Negative examples are constructed by adding one or appending an independent list variable.

Four primitive attempts: two-frontier search (64 expansions), congruence decomposition with a shared total subproblem search budget of 64 expansions, full explicit normalization (256 steps per endpoint), and short normalization (16 steps per endpoint). The normalizer inherits the STATE-12 4096-node cap. The decomposition specialist recurses through equal outer constructors only when the corresponding children have independently equal semantic forms. Otherwise it searches the local subproblem, failing at the root when no decomposition is available. Lift every local certificate to the outer context by congruence. All paths checked against the rewrite schema and separate denotational evaluator at each step.

The four static routes are search then normalization; normalization then search; decomposition then search then normalization; and short normalization then search then normalization. Stop on the first certified success. Evaluate each primitive once per positive equation with search and distance caches cleared, and compose its measured cost into each counterfactual route. Thus route totals are offline accounting, not separate end-to-end timings. Charge one unit per recorded schema match attempt, structural/node visit, expansion, newly generated edge or normalization rewrite. Count all attempted work including failures; add 32 units per successful certificate step. Search matches count all nodes times rule count on every expanded expression, including cached matches. The proxy does not charge every operation (e.g. semantic normal-form arithmetic during decomposition, hashing and heap comparisons); actual attempt process time includes this work and verification, but cache/order effects remain. Gate time is separate and shared. Costs are not FLOPs, Lean kernel calls or billing estimates. Sensitivity weights 0,32,128,512 use fixed policies.

Three 996-parameter MLP routers (seeds 313,317,331; 400 full-batch AdamW steps at .003) regress log1p(route cost) relative to per-case cheapest route using 26 public structural features, then choose the lowest predicted cost. Failures of positive training routes carry 1e9 penalty. Features include domain, AST sizes/depth, aligned distance, shared structural fraction, root match and token counts. No family ID, held attempt outcome or verifier label enters routing. Train-only best static is decomposition first. Compare random route and retrospective oracle route; the latter uses held outcomes and is a headroom diagnostic, not a deployable policy. All attempts succeed in combination on this panel. Neither learned nor static policies may certify without a checked proof.

## Results (held, 96 positives + 36 negatives)

|Policy|Solved|False accepts|Work|Certificate steps|Cost (weight 32)|
|---|---:|---:|---:|---:|---:|
|Search first|96|0|256189|652|277053|
|Normalize first|96|0|257109|3260|361429|
|Decompose first|96|0|192171|597|211275|
|Short normalization first|96|0|262597|1138|299013|
|Random|96|0|261276|1245|301116|
|Learned 313|96|0|194878|621|214750|
|Learned 317|96|0|197290|621|217162|
|Learned 331|96|0|197292|621|217164|
|Oracle route|96|0|67590|719|90598|

Decompose first saves 23.74% of search-first weighted cost. Learned policies cost 1.64–2.79% more than the train-selected static route and never beat it at any of the four sensitivity weights. Oracle cost is 57.12% lower than decomposition first. It is retrospective and does not establish learnable routing. At weight 32, primitive attempts alone solve search 91, decomposition 73, normalization 95, short normalization 60 of 96. All portfolios recover all positives. The shared semantic gate abstains on all 36 negatives, so zero false accepts is a gate/checker result rather than learned recognition of invalid goals.

|Family (32 positives each)|Search first|Decompose first|Normalize first|Oracle|
|---|---:|---:|---:|---:|
|Long walk|209247|191073|165329|70661|
|Deep context|23588|8708|172495|8708|
|Two holes|44218|11494|23605|11229|

Decomposition first is particularly useful for shared context and independently changed children. Long walks account for most of the remaining weighted-cost headroom. All learned routers choose decomposition for every deep-context equation. They lose overall even though their predictions are trained on cost rather than classification. Need a stronger training distribution and cheap information about search/normalization difficulty before another learned router trial.

Total process time 2.22135 s including generation, attempts, fitting and certificate compression (prior to final receipt write). Model training .1613/.1668/.1723 s. Batched inference .00103/.00055/.00052 s across 132 held rows. Offline summed attempt CPU time .2938 s for search first, .2305 s for decomposition first, .2318/.2361/.2361 s for learned arms, excluding shared gate (.00418 s), feature extraction and routing overhead. These are descriptive, not controlled deployment latency benchmarks. Zero paid credits.

## Mathematical note

If f is a well-typed constructor and pi_j proves x_j=y_j for every differing child, congruence constructs f(x_1,...,x_k)=f(y_1,...,y_k). The sequence of lifted paths has length sum_j |pi_j| (the unchanged children need no steps). This construction can avoid a Cartesian product search across independent children. With branching factors b_j and depths d_j, naive simultaneous enumeration can have product_j b_j^d_j possibilities whereas separate searches may examine on the order of sum_j b_j^d_j in a simplified tree model. Actual savings depend on duplicate states, heuristics, rewrite applicability and certificate verification. Equal semantic forms merely suggest the decomposition; the emitted proof must still check.

Routing objective for a problem x and ordered solver portfolio p is

    C(p,x) = sum_{a attempted} W(a,x) + lambda * |certificate(p,x)| + M * 1[true and unresolved].

Use lexicographic coverage or an explicitly dominant failure penalty; a cheap abstention on a true goal is not a success. Compare deployment cost at matched verified coverage and include learning and verifier costs before promotion. A candidate for the next study is to learn the conditional cost after a short diagnostic probe of both global search and normalization, rather than guessing from a static AST encoding. Register a new long-composition panel, use an adaptive symbolic threshold as a control, and account for the cost of the probes themselves.

## Replay

From training/aleph_next/state01, with CPU PyTorch and NumPy:
    python state13.py --out /tmp/state13

The script imports state12.py, state11.py and state11_normal_forms.py. It emits a per-case JSON receipt, family summary CSV, and gzip-compressed held primitive proof paths. Source SHA256 464a24aadbeb095b41e23d4b4516e8be60aa03372672f6f107163008fdf7968c; committed Git blob 807f8844b5e9ff1328e8da66b6d1a16cbb9dfe04 matches the executed file byte-for-byte. Numerical receipt SHA256 9f6d14d51597cdf469c3e44afc11d7288b2c456e9bf3adc45b210cd6852e6f1b (timing dependent); certificate archive SHA256 776d216d40dcd87a336cec6551e927837466fc7ec0f18a82824649e0884db2a4. No generated proof enters the kernel-verified DataGoblin corpus.

Decision: use decomposition-first as the new static reference for this grammar. No learned router promotion. Native historical Aleph checkpoints and Lean kernel unavailable. Keep both 10B slots on hold.
