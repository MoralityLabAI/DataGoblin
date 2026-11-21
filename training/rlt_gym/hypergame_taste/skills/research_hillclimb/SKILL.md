# Skill: Research Taste Hill-Climbing (RLT repair controller)

## Trigger

Use when a DataGoblin/Storyworld research agent must choose an experiment, question a previous plan, or repair a checked failure while preserving resource and evidence custody. Works for K-depth nested-belief worlds, experiment-design forks, and eventually source-locked PaperBench / nanoGPT / Storyworld episodes.

## Objective

Maximize verified task utility and useful scientific information per unit of experiment compute and verifier work. Prefer recoverable, falsifiable research steps. Record negative results and hypotheses that remain viable.

## Required inputs

- Frozen root identifier and predecision public task view.
- Current model/harness identity, skill version, visit budget and remaining experiment budget.
- Legal action menu and access-controlled tools.
- Previous public hypotheses, predicted outcomes and observed experiment results.
- Independent checker receipts from earlier visits, each with exact input/artifact hash and authority.
- Optional declared K-depth belief map. The observer owns only its visible p/p2 or higher-order belief fields.

Reject the request as insufficient_counterfactual_data for goal inference if the advertised legal actions lack grounded subjective and objective successor tables. Preserve unavailable or censored check results as unknown.

## Action schema

Return one JSON object containing:

    {
      "action": "pilot | replicate | submit_0 | submit_1 | abstain",
      "hypothesis": "A falsifiable claim about the observed game",
      "probability_h1": "rational or calibrated decimal",
      "predicted_observation": "Brief prediction for an experiment; null for terminal actions",
      "cost": 0,
      "evidence_ids": ["predecision evidence IDs"],
      "revision_of": "prior proposal ID or null",
      "reason": "The expected information or capability gain under the declared budget"
    }

For broader benchmarks, the action enum is supplied by the taskset. The policy may propose a new experiment only through the declared legal extension interface. A proposed experiment is a candidate; it must be checked by the environment before its usefulness becomes a verified claim.

For the strategic research worlds, use `pilot`, `replicate`, `intervene:<mode>`, `publish:<mode>` or `abstain` according to the visible legal menu. The local model collector requests only the action field. An intervention tests the initially omitted mode, while a publication commits to one mode. Record the remaining credits and turns before proposing. Update the forecast after a probe exposes its observation. During controller training, use only a prior verifier acceptance or rejection as correction feedback. The grader retains exact action values, hidden truth and unused successor signals.

## Executable cycle

1. **Observe.** Read the actor-specific predecision state and visible experiment history. Compute or estimate a belief state and record what evidence supports it. Keep objective outcomes, after-turn reveals, future publications and sealed checker labels outside the model prompt.
2. **Propose.** Pick a candidate research action. Estimate its probability of useful results, false-accept risk and resource cost, and identify a potential discriminating observation.
3. **Check.** Send the action to the independent taskset verifier. The verifier reports a status and artifact/receipt hash. Real experiment observations become visible only after the corresponding action and at the declared time.
4. **Repair.** On checked failure or high validated regret, re-evaluate the public hypothesis, propose a different action, and retain the failed candidate and source receipt. A verifier output that reports only a score or failure code should never be treated as an oracle explanation of the right answer.
5. **Stop.** Commit on an accepted verified terminal action, abstain when no action has positive expected value, or mark exhausted_budget when the bound is reached. Record all consumed visits and checks.

Run a first-pass policy, a heuristic repair policy, a check-every-visit ordered policy, and a deferral policy on matched roots. Track prospective checks separately from offline audit replays.

## Rollout and training record

A full transition has:

    {
      "root": "...",
      "split": "train",
      "K": 2,
      "model": "...",
      "source": "native | codex | opus | local",
      "context": {"visible_task": {}, "visible_history": []},
      "pre_verdict_proposal": {"action": "..."},
      "independent_check": {"status": "...", "receipt_sha256": "..."},
      "revision": {"action": "..."},
      "resources": {"visits": 1, "verifier_calls": 1, "compute_units": 0},
      "evidence_class": "verified_teacher | synthetic_exact_oracle"
    }

DataGoblin's first exact-reference collector writes synthetic_exact_oracle fork rows and teacher requests. For LLM data, preserve the real source session and token receipts, import replies separately, verify every candidate on the frozen task, and retain failed and censored attempts. The synthetic source is a training aid and engineering control rather than evidence of trained RLT performance.

Select at most four training presentations per root **across all data views**. Keep original source prompt, corrected answer, pre-verdict state, and policy preference separate. Preference comparisons require identical public state at the decision point. Avoid mixing a checker-observed answer into a supposedly pre-check observation.

## Model-mesh deployment

- **TRM:** query/situation router and coarse branch selector.
- **LDT:** index/filter/compact candidate observations, with explicit observation-loss and collision controls.
- **RLT:** recurrent policy correction over proposal, verifier feedback and revised evidence.
- **Taskset oracle:** verify decisions, simulate allowed experiments and score true utility.

Compare RLT against direct-feedback and matched damping at fixed fit and runtime schedules. Random controls should receive equal check and visit budgets. Exact expected-regret labels are available on the public synthetic task family; external research taste benchmarks will require blinded outcomes and expert agreement.

## Next curriculum

- **K=0,1,2:** learn simple belief adjustment, calibrated abstention and value-of-information thresholds.
- **K=4,8:** learn stable nested perceptions and correction when a plausible endorsement is misleading.
- **HG-02/04 transfer:** preserve decision-sufficient information under high-dimensional irrelevant features and noisy audits.
- **Research fork mining:** rank proposed minimum viable experiments before outcomes; ground the label in independent post-experiment evidence.
- **Active research taste:** propose new hypotheses, select discriminating tests, compare matched compute and replay verifier receipts.
- **Full Storyworld:** multiagent subjective games with conflicting legal actions, changing world constraints, partial observer visibility, source-locked transitions and preregistered held-out worlds.

## Promotion and failure gates

Report per-K and per-task-family paired solve rates, true utility, expected regret, prospective verifier calls, recurrent visits, tests consumed, false-publication rate and abstention. Include confidence intervals over **root clusters**. Reject promotions that gain only through unseen outcome leakage, seed inference, extra checker calls hidden in training, task-mix drift, or invalid receipt replay.

Promote the learned controller only after it improves held-out research choices over the cost-matched heuristic and frozen ranker at the same budget. Preserve unsuccessful hypothesis searches in the training archive. Keep sealed_poison as a custody canary.
