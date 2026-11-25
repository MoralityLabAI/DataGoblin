# DLMST / MIC-01 research curriculum intake

Source: Khanna, Putterman, Song, ECCC TR26-339 (2026-11-23), Section 1.4. Pinned source: https://github.com/Louiealbp/DLMST/tree/0072335716b145032b418760883daed2a89809db.

The [reference manifest](dlmst_mic01.json) inventories 18 manuscripts using original Git blob identifiers, source links and the chronology reported by the authors. These are public bibliographic references; stage assignments from filenames are hypotheses pending review. All 18 are one discovery family. No raw manuscript material is copied. The manifest is an intake reference, not a steward-released data collection envelope or corpus handoff.

**Disposition: QUARANTINE for training, reference metadata only; zero token allocation.** No repository license was visible, and proof validity, normalized text, exact tokenizer counts and deduplication remain unverified. Do not infer training permission from public visibility. DataGoblin's existing data etiquette and steward review gates apply before any use in a training recipe. The public DLMST trajectory is excluded from the held-out evaluation.

For the prospective training experiment, treat a discovery step as typed source context, old bound, new bound, computational model, assumptions, dependency anchors, and proof/verification status. When independently verified and eligible, examples can supervise proof-obligation selection, candidate generation, or asymptotic-bound revision. Compare ordered vs shuffled context under matched budgets with independent synthetic families. Public source variants must never be treated as independent discoveries.

The [EvalMath MIC-01 draft PR](https://github.com/MoralityLabAI/EvalMath/pull/83) contains a tested exact rational certificate proxy. Its public development preflight failed the resolution gate; it provides no evidence for model training uplift. Avoid increasing Aleph's 10B token budget or releasing these records on that basis.

## MIC-02 public finite diagnostic (2026-11-25)

[EvalMath PR #85](https://github.com/MoralityLabAI/EvalMath/pull/85), pinned report: https://github.com/MoralityLabAI/EvalMath/blob/0b7a781a0e383e8031f63584845412cf39bd1cbd/gyms/MIC02/README.md.

Four original synthetic certificate families (32 public cases each), 15,360 exhaustive update schedules, exact scoring. Ordered-vs-shuffled normalized objective improvement: recurrence +18.79 pp, counting +8.70 pp, amortized +0.39 pp, derandomization +0.00 pp. These are scheduling controls, not model or textual-history outcomes. No independent held-out families; family-level matched-sham calibration is unavailable. Disposition remains REDESIGN_OR_PILOT_ONLY.

Original synthetic traces could eventually support independently verified curriculum tasks, with typed state, candidate action, exact score, dependency structure and verification receipt. Preserve all public development roots outside confirmation. Prioritize recurrence and counting for nontrivial dependency-order tasks, redesign amortized plateau and derandomization ceiling cases, and include specialized exact-solver controls. Do not allocate training tokens or release the DLMST manuscripts. Keep both source-rights and independent-evaluation gates intact.

## MIC-03 learned scheduling diagnostic (2026-11-25)

[EvalMath draft PR #86](https://github.com/MoralityLabAI/EvalMath/pull/86), pinned report: https://github.com/MoralityLabAI/EvalMath/blob/c5869a551dfa26caa46d37406416277bce6d7009/gyms/MIC03/README.md.

96 training roots, six 1,793-parameter CPU fits and 192 public development rows. Mean normalized gap to exact one-pass schedule oracle: dependency rule 0.64972%, greedy 1.20702%, recurrent ranker 1.23294%, parameter-matched stateless ranker 1.24337%. Dense covering favors both learned rankers; recurrence-specific effect is only +0.01043 pp overall. Cyclic-permitted recurrence has 19.14378 pp residual gap between best one-pass schedule and least feasible certificate, indicating an action-budget bottleneck in this proxy. The executed repository preflight returned REDESIGN_OR_PILOT_ONLY. Prioritize verifiable multi-visit certificate improvement tasks with explicit residuals and stopping conditions; retain structural baselines and equal-budget exact ceilings. No training promotion or token allocation; the public roots remain development-only. No change to DLMST manuscript rights or eligibility.

## MIC-04 revisitable certificates and atomic repair (2026-11-25)

[EvalMath draft PR #87](https://github.com/MoralityLabAI/EvalMath/pull/87), pinned report: https://github.com/MoralityLabAI/EvalMath/blob/eab9c34beb07061cd027706f92d34ec9e460696e/gyms/MIC04/README.md.

Six CPU fits, 48 training roots, 96 public development roots across four generator lineages, 480 budget rows. Exact oracle gap to least feasible integer certificate falls from 14.59 pp at four visits to 3.81 pp at sixteen. Residual greedy remains ahead of learned routing. The 16-visit oracle gap contains 1.72 pp of irreducible coordinate-action rounding traps, found on 38/96 roots. Solving and atomically replacing strongly connected components yields the least certificate on all 96 roots; internal work is separately accounted for and is not matched to scalar-visit costs. This motivates independently checked training examples with grouped obligations, explicit intermediate verifier scope, atomic commit receipts, and separate action-class and search-budget regret. Recurrent-vs-stateless benefit at K=16 is -0.09537 pp. Planner returns REDESIGN_OR_PILOT_ONLY. Do not promote or allocate training tokens, and keep public development roots out of confirmation. DLMST reference rights and eligibility remain unchanged.
