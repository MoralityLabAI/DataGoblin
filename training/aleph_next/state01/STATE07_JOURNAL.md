# ALEPH-STATE-07 | cost-aware witness allocation and open-world rejection

2026-11-25. CPU-only public diagnostic following STATE-06. A remains the reference; paid training remains frozen. The learned allocator is an MLP proxy for a possible LDT policy, not a native Aleph/LDT implementation.

## Design and custody

Reconstruct the 24,224 semantically deduplicated functions from STATE-06. Reconstruct its held operator IDs and exclude them from new train and evaluation pools. New split seed 2026112510: 256 seen-template held operators (64/class), 192 held-template operators (64/non-affine class), training pools 352/7008/4960/3248. Operator truth IDs disjoint from training tasks and previous evaluation. Candidate grammar is shared, including the held functions; teacher entropy uses the full grammar but observes outputs only of sampled train tasks. This is closed-world acquisition, not grammar discovery. Prior uniform by class and uniform within each class.

Generate 640 balanced train tasks, alternating random and greedy teacher trajectories, through at most eight witnesses. Teacher conditional predictive-entropy soft targets (temperature .25) at noninitial nonsingleton states. 3,225 examples; teacher scored 1,135,282,216 candidate-coordinate cells (including repeated initial scores). Three 71,104-parameter MLP fits, seeds 277/281/283, 500 AdamW updates, batch 128, LR .002. All three share teacher examples. Public seeds fixed in source; no sealed external adjudication or statistical promotion.

Compare random, full function entropy, class mutual information with function fallback after class unanimity, class-to-function switch after four queries or class unanimity, posterior sampling with replacement (32 candidates when version space >32; exact otherwise), learned ranker, and hybrid (learned when >32 candidates; exact entropy otherwise). All nonrandom policies share a cached initial function-entropy query at coordinate 0. Its one-time setup cost is excluded from per-policy acquisition counts; random starts randomly. All policies continue to fixed budgets 4/8/16, selecting random unqueried points after singleton or empty version space. Exact filtering always uses every surviving candidate, including for neural and sampled policies. No oracle output at an unqueried coordinate enters the policy. One query reveals one exact scalar output.

## Mathematics

For a finite version space V with posterior p(h|D), exact acquisition maximizes H(Y_x|D). Estimate predictive probabilities by m IID posterior draws, with each sampled candidate evaluated at every remaining coordinate. Conditional on D, Hoeffding and a union bound over coordinates and labels give

    Pr(max_{x,y}|p_hat(y|x,D)-p(y|x,D)| > eps)
        <= 2 |X| |Y| exp(-2 m eps^2).

Thus the sample cost for a fixed uniform probability error scales logarithmically with the number of candidate coordinates/labels, not with |V|. Entropy continuity converts distribution error to acquisition-score error; if all score errors are bounded by delta, the true score of the chosen query is within 2 delta of the exact greedy optimum. This does not establish a useful numerical uniform bound for m=32, nor global optimality of greedy query selection. The empirical near-match motivates this cheaper policy.

Cost decomposition: sum over query steps of |V_t| for exact filtering, plus |V_t| times the number of unqueried coordinates for exhaustive acquisition; sampling replaces the latter factor |V_t| with at most 32. Learned ranking incurs neural inference and training costs not captured by candidate-cell counts. Candidate-cell comparisons are a proxy, not FLOPs or dollars.

## Executed closed-world results

Pooled IID and template-OOD operators; 448 per policy/seed at each budget. Exact unique-function recovery:

| Policy | 4 | 8 | 16 | Filtering cells @8 | Acquisition cells @8 | Neural calls @8 |
|---|---:|---:|---:|---:|---:|---:|
| random |6|295|437|12501312|0|0|
| function |13|447|448|12425458|98805722|0|
| class |0|426|448|12666356|113640840|0|
| switch |0|444|448|12658596|113175742|0|
| sample32 |6|446|448|12425391|2931995|0|
| learned 277 |3|347|435|12434621|0|2445|
| learned 281 |3|363|439|12432310|0|2498|
| learned 283 |3|365|440|12429567|0|2477|
| hybrid 277 |3|445|448|12433451|341842|1349|
| hybrid 281 |3|447|448|12431011|332437|1352|
| hybrid 283 |3|448|448|12428243|327860|1343|

At budget 8 exact function entropy achieved 255/256 IID and 192/192 OOD; sample32 255/256 and 191/192. Sample32 cuts acquisition cells by 33.7x, total filtering+acquisition cells by 7.24x. Evaluation CPU (all three panels and all 16 queries; includes scoring) exact function .9256 s, sample32 .3832 s, hybrid .4630/.5375/.5031 s. The hybrid uses fewer counted symbolic cells but is slower than sample32 on this CPU. Pure neural ranking underperforms random at budget 16 in one seed, and underperforms exact acquisition at budget 8 in every seed. No learned deployment win demonstrated. Class objective improves early class identification (324/448 at 4 queries versus function 291/448) but has lower exact recovery. Results depend on the finite grammar and prior.

## Open-world stress test

Generate 128 distinct functions absent from the complete grammar (seed 2026112513). First 64 are a single-coordinate perturbation of a known function; other 64 combine known functions with a periodic perturbation. Reject all examples that coincide with any grammar function; this conditioning may bias perturbation locations. Scoring labels for unknown classes are -1, so class accuracy is irrelevant. Measure rejection and false singleton claims, not closed-world accuracy. No open-world training examples.

| Policy | wrong singleton @8 /128 | wrong singleton @16 /128 | sparse rejected @16 /64 | composite rejected @16 /64 |
|---|---:|---:|---:|---:|
|random|35|47|17|64|
|function|53|44|20|64|
|class|56|46|18|64|
|switch|55|47|17|64|
|sample32|56|47|17|64|
|learned 277|48|48|15|64|
|learned 281|46|41|20|64|
|learned 283|46|50|14|64|
|hybrid 277|63|47|17|64|
|hybrid 281|60|52|12|64|
|hybrid 283|59|54|10|64|

Even a wrong singleton can persist after further queries. For a known base function with a uniformly random single defective coordinate and no side information about its location, before hitting that coordinate the observed transcript is identical to the base function. Any deterministic or randomized adaptive policy making b distinct queries therefore detects the defect with probability b/N. This idealized result is not an exact model of the conditioned stress generator. The worst-case guarantee against a one-coordinate adversary requires all N points unless there is a valid structural promise or independent proof. Keep grammar-relative certificates distinct from unconditional verification.

## Decision and next gate

Use sample32 as the default cost-aware symbolic reference on this grammar, and retain the learned hybrid as a research arm only. Next STATE-08: counterexample-guided repair with expanding grammars, explicit repair description costs, independent certification, and verification budgets that charge expensive oracle calls separately from acquisition. Include sparse faults, false grammar coverage, and held repair families. Exact entropy and sample32 controls required. No paid training, no change to A; no claims about the historical 14M checkpoints.

## Reproduction and receipts

Run python state07.py with torch 2.10.0+cpu and NumPy 2.3.5. Source SHA256 8deea2493a51c216b1a1b6f170d220255cf5017b6a3e4b2abee17d3f6c8bc155. Executed source matched committed Git blob 09bdd35c45b53c08e31ae2765804b357bb476f2b. Full local JSON SHA256 c95fbc7eb276a2c59dc031fec2d3998f495f597e02659a9fb04a19b00cf78358. Includes per-operator outcomes at 4/8/16, all policy/panel counts, training metadata and CPU times. Measured teacher preparation 8.911596 s, neural training 2.508047 s, policy evaluation 7.696470 s; total 19.116113 CPU process seconds excluding setup, imports, serialization and other work. No accelerator spending. This diagnostic was designed and run in one session, not independently preregistered or sealed. Historical native Aleph checkpoints remain unavailable.
