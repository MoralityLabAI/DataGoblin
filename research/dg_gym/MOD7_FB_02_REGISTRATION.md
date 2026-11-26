# MOD7-FB-02: optimizer-aware marginal-gain feedback

Public development only. No paid compute. Same 2058-parameter simplex model
as FB-01. Seeds 11,23,37. Matched Adam .05 initialization/streams, batch 96.
Common 150 length-one task updates, 150 length-two full-dose CF updates
(weight .5). Then six 50-step windows at length four. Candidate actions
are the three internal full-dose patch sites; doses collapsed because of FB-01
score alias. Train data excludes exact full roots from the 64-root length-four
causal audit cohort (seed 63004). Monitor is 64 length-24 roots (seed 63024);
evaluation 512 roots each at 6,32,128,512 (seed 64000+length). Monitor and
evaluation are never gradient targets. Entire instruction family shared;
no transfer claim.

At each length-four window start, adaptive, static-shadow, shuffled-shadow
perform the same causal specificity audit (identity, genuine, wrong donor)
for all three sites, plus one optimizer-aware lookahead for task-only
(zero-weight shadow CF at site 0) and each of three CF actions. Clone
parameters and complete Adam state; use the same next actual batch for all
four branches, with no random draws inside branches. Discard clones. Score
each by unsteered mean exact target probability at length 24 minus task clone.
Adaptive selects highest gain (tie <=1e-9, smallest site); static cycles;
shuffled assigns the best score to a permuted site (seed 71000+seed+window).
Commit the chosen action for 50 steps, including the preview batch once.
No abstention in this test (fixed action family); keep negative scores.

Campaign budget: one credit per example-block forward visit and two credits
per corresponding differentiable visit (backward work proxy); count clone
updates and monitor/audit forwards. All shadow arms have equal 600 committed
updates and equal credit. Static-reinvest skips all lookahead and audits,
runs the same common prefix and 300 cyclic CF updates, then uses remaining
credits for full length-four cyclic CF updates. Task-reinvest shares the
common prefix, uses ordinary task loss alone in the length-four phase and
spends the same cap on complete updates. Stop before exceeding the cap.
Do not charge optimizer copies as zero actual cost: record clone/audit
wall and CPU time separately. Visit-credit comparison is a deterministic
proxy, not matched FLOPs, RAM, energy or elapsed wall time. Common final
evaluation budget is separate and identical for all arms.

Primary screen: adaptive-minus-static-shadow, adaptive-minus-shuffled-shadow,
adaptive-minus-static-reinvest length-512 mean target mass each >=.01 on
average, positive on all paired seeds; adaptive all 294 row argmaxes correct
and >=461/512 length-six soft accuracy. Report all arms, costs, selections,
scores and checkpoint hashes. Report task-reinvest as additional budget
control. Causal audit is a diagnostic, not the gain ranking itself. No
promotion/corpus admission, no sealed results. Process-level peak RSS only,
not per-fit peaks; GPU unavailable. Three seeds are descriptive.