# DG-Gym v0.5 / MOD7-FB-01: mechanistic feedback during training

Status: public development. No paid compute, sealed evaluation, training/corpus
admission or general transformer transfer claim.

## Feedback calculus

Extend I to (M,U,F,Gamma,A): measurement operators, permissible interventions,
feedback computation, control policy and information authority. An action binds
block/visit/state slice, direction or donor rule, dose, loss weight, gradient
mode and a bounded training window. All feedback audits bind the current
checkpoint, feedback cohort and optimizer step. A stale receipt or a final
evaluation receipt cannot drive selection.

For matched roots, let b,d,w be counterfactual target probabilities after
identity, genuine donor and wrong-donor interventions. Specificity is
E[d-max(b,w)], evaluated by taking the maximum within each root before
averaging. The registered ideal response S* is supplied by the task adapter.
A deficit 1-S/S* is a diagnostic; it is not an estimate of the benefit of
training on the action.

For training windows, a stronger selection estimand is
G(u)=J(U(theta,opt,u))-J(U(theta,opt,0)), where J measures unsteered retention
on a separate monitor cohort and U clones the checkpoint, optimizer, RNG and
training batch schedule for a short paired update. Charge all cloned updates.
An SGD first-order proxy for maximizing J is
-eta*lambda*<grad J, grad L_CF(u)> relative to a task-only update. This
approximation is conditional on local differentiability/small steps; Adam
requires optimizer-aware updates or actual clones. The clone/gain estimator
is a proposed next adapter, not a result of this experiment.

The generic select_window module supports exploratory deficit ranking and
cost-normalized positive lower estimates of marginal gain. It rejects stale
or mismatched checkpoint/cohort/step bindings, non-feedback scope, duplicates,
missing cost estimates, and infeasible resource bounds. A deterministic tie
tolerance resolves score aliases. A gain policy abstains when the gain bound
is missing or nonpositive; a competence gate can close all interventions.
Audit costs must already be deducted. Reserve remaining cumulative wall,
CPU/GPU time, visits and verifier calls plus RAM/VRAM peak capacity for the
complete chosen window. The module is a decision primitive, not a profiler,
receipt authenticator, optimizer or OS cap enforcer. The production trainer
still needs adapters for probes, checkpoints, monitors and supervisors.

## MOD7-FB-01

Registration: f965876cce7b6f1fe8afee75018efdc00f1d5663.
Exact replayed pilot blob eca7044bc902665cf859da417b9ab31580e980d7;
SHA-256 5b580eef574fac8aec4aeda32489beca3f70e64c3b19cdb2113c77d975bac304.
PyTorch 2.10.0+cpu, one thread. Fifteen 2058-parameter fits, three paired
seeds per arm, 600 updates, 57,600 accepted program draws. Length 1/2/4
curriculum at 150/150/300 updates. Feedback roots excluded from same-length
training draws by exact root encoding, with 112/124/129 rejected draws
per seed and identical streams across arms. The final public roots are
lengths 6/32/128/512. The exact residue supplies oracle-assisted donor
selection and counterfactual targets; the architecture uses known state
coordinates.

Every 50 updates during multistep training, measure all site/dose pairs on
96 fixed feedback roots for the active length. Use true donor, identity
and wrong donor. The adaptive arm selects the largest deficit; static
cycles across candidates; shuffled feedback randomly permutes the action
association. All arms run the audits and the ordinary plus steered forward
on multistep batches. Task/concept have zero-weight counterfactual shadow
loss. Concept uses intermediate NLL; steering arms use 0.5 times exact
counterfactual mixture cross entropy. Donor gradients are enabled.

| Arm | Mean target mass at 6/32/128/512 | Worst row TV | Median train s | Median audit s |
|---|---|---:|---:|---:|
|Task|.982581/.910959/.695539/.291088|.00516358|.874910|.024434|
|Concept|.985928/.927454/.744568/.351097|.00415160|.958882|.023586|
|Static|.987891/.937261/.775240/.396900|.00369242|.856831|.022913|
|Adaptive|.987713/.936366/.772390/.392355|.00360439|.856915|.023171|
|Shuffled feedback|.987906/.937351/.775493/.397301|.00363774|.848358|.022676|

All 15 fits meet 294/294 row argmax and 512/512 soft accuracy at all depths.
Length-32 full-dose donor patches produce 1536/1536 correct per arm;
mean specificity .895901/.915182/.926651/.925605/.926746, respectively.
Primary paired adaptive-minus-static length-512 target mass:
-.0037707243, -.0005001645, -.0093649330. Adaptive-minus-shuffled:
-.0038016850, -.0016433680, -.0093952180.
Both fail the preregistered +.01 mean and positive-each-seed screen.
Static counterfactual training improves descriptive unsteered target mass
over task and concept training. Adaptive ranking adds no measured benefit
in this fixture. Pooled outcomes share roots; no independent pooled CI.

Every fit consumes 302400 train forward example-visits (158400 ordinary,
144000 steered) and 32832 feedback-audit example-visits. This is +10.86%
over training visits before backward and evaluation accounting. CPU wall
timings are descriptive sandbox measurements, not matched wall-time claims.
Per-fit peak RAM and GPU throughput unavailable. All 15 replay checkpoint
bytes and non-timing fit results match the first run. The replay source
only binds the score lambda's loop-local variable explicitly for lint.

## Exact dose-alias result

Fix a site, receiver/donors and a linear suffix acting on a probability state.
Write target mass under identity as b; let D and W be the full-patch genuine
and wrong-donor mass increments relative to b. For alpha>0 the convex edit
gives d(alpha)=b+alpha*D and w(alpha)=b+alpha*W. Hence

    d(alpha)-max(b,w(alpha)) = alpha*(D-max(0,W)).

Averaging yields S(alpha)/alpha=S(1), so the normalized deficit cannot rank
doses in exact arithmetic for this architecture. All 315 paired dose-score
differences in the first-run receipts were <=1.78814e-7 in magnitude.
The original controller used exact float ordering, and some choices were
rounding-sensitive. This diagnoses an alias in the action score; it does
not establish why the measured training differences occurred. Nonlinear
suffixes or different dose-dependent policies can break the identity.

Next: identify equivalent actions before ranking; use a separate paired
marginal-learning-gain estimator; compare against static full-dose,
static cycle, shuffled feedback and task-only arms under matched total
campaign budgets. No gain-based controller has been experimentally validated
yet. General SAE/residual steering needs calibration, norm-matched and
semantic shams, collateral checks, gradient-path declaration, and frozen
probe/feature identities. Final evaluation is unsteered and never feeds back.
