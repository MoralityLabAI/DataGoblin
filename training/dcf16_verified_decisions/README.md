# DCF-16: Verified Boolean decision curriculum

2026-11-25. Research export for Data Goblin, following DCF-15 (HybridTRMLDT PR #29, https://github.com/MoralityLabAI/HybridTRMLDT/pull/29). This is a standalone curriculum adapter, not an import into the main training collator or a trained model.

The upstream dispatcher compiles reusable structural elimination plans. Its 864 original frame labels are independently replayed here in a log-space factor-elimination implementation with explicit assignment indexing and a different elimination tie order. All 864 posterior and decision labels agree (max posterior discrepancy 2.6645352591003757e-15). Original 288 n=8 frames also had brute-force checks upstream. Independent algorithm comparison is numerical FP64 evidence, not a formal interval certificate. All 144 new family holdout labels also agree with the upstream dispatcher (max discrepancy 3.6637359812630166e-15); 48 width-8 cases agree with brute force (max 1.0547118733938987e-15). Maximum dispatcher scope 6.

## Data contract and split

Each JSONL frame has schema dcf16-v1, a SHA-256 ID of the canonical input, split, source group, input and target. Input contains Boolean scopes, positive factor tables, queried variable and decision reward (+1 correct, -4 wrong, 0 abstain). Target contains oracle posterior, optimal action (0,1,null), expected utility and verifier status. Factor axis ordering is scope order. All 1008 input IDs unique. The export aborts if an output directory already exists. The first export detected duplicate inputs caused by shared holdout RNG streams in equivalent small graphs; output discarded, holdout family streams separated and export rerun.

|split|count|action0|action1|abstain|
|---|---:|---:|---:|---:|
|train|576|68|59|449|
|seed_holdout|288|27|37|224|
|family_holdout|144|25|21|98|

Train uses original development seeds. Seed holdout is the original test seed; nine exact labeled topology hashes overlap training. Family holdout uses ladders, clique chains and grids at widths8,16,32, with 18 family/width/seed groups and eight frames/group. No exact labeled topology hash overlap with training. This is not a graph-isomorphism proof or cryptographically sealed prospective benchmark. Keep both holdout files out of training; labels in the export are visible to the researcher.

## Utility-based evaluation

For reference posterior p, utility of action0=1-5p, action1=5p-4, abstain=0. Optimal utility is max of these. Score expected utility, regret, exact action accuracy, coverage and correct decisions on nonabstention opportunities.

|split|oracle utility|always abstain accuracy|unary-only accuracy|unary-only utility|unary-only regret|
|---|---:|---:|---:|---:|---:|
|train|.071715|.779514|.805556|.002776|.068939|
|seed_holdout|.085124|.777778|.836806|.024550|.060574|
|family_holdout|.143034|.680556|.680556|-.008910|.151944|

Always abstain has zero utility and zero opportunity accuracy despite high ordinary accuracy. Unary-only ignores relational evidence; its family holdout utility is negative. These are descriptive controls, not learned routing experiments. No GPU or paid compute. No runtime/working memory claim.

## Reproduce

Run DCF-15 benchmark.py from HybridTRMLDT research_gym/benchmarks/dcf15_structural_dispatcher on the DCF-15 branch (source hashes in upstream manifest) to regenerate frames.jsonl. Then run:

    OPENBLAS_NUM_THREADS=1 python export.py --source /path/to/frames.jsonl --out data
    python score.py data
    PYTHONPATH=/path/to/dcf15_structural_dispatcher python family_audit.py data
    sha256sum -c SHA256SUMS

Manifest covers actual local generated data and scripts. Raw JSONL splits, audit and baseline receipts are provided in the conversation archive; the branch stores the exporter, scorer, manifest and this report. Training dataset registration, EvalMath integration, hidden split infrastructure, plan-level routing traces, learned model experiments, and larger independent graph families remain outstanding.
