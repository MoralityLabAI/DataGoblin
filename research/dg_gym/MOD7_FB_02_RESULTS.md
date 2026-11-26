# MOD7-FB-02 / DG-Gym v0.6: optimizer-aware steering feedback

Public development only; no paid compute or promotion. Registration
36c8149b740b3786ea118a2733400c64e077a98a preceded execution. Source
mod7_feedback_gain.py Git blob 109d92fb1fe46970d231360b55fd8a7b3f9a8b45,
SHA-256 98443dae180323b2cdfb8362a9663509a3ebf583cfa26e7e274c3e169e34a7d3.
Dependency mod7_feedback.py Git blob eca7044bc902665cf859da417b9ab31580e980d7,
SHA-256 5b580eef574fac8aec4aeda32489beca3f70e64c3b19cdb2113c77d975bac304.
Both local blob hashes match the pinned repository sources. PyTorch
2.10.0+cpu, one thread, 2058 parameters, seeds 11,23,37.

## Design

All arms share the first 300 updates: 150 length-one ordinary updates and
150 length-two full-dose counterfactual updates. Each steering step uses
terminal NLL + .5 times exact counterfactual NLL. Candidate actions in
the length-four phase are full-dose patch sites 0,1,2. Dose aliases were
removed. For each of six feedback rounds, shadow/adaptive arms audit
causal specificity on 64 length-four feedback-only roots, then clone the
current parameters and complete Adam state four times. Each clone performs
one update on the same preview batch, with ordinary or site-specific
counterfactual training. The clone monitor is mean exact terminal mass
on 64 length-24 roots. The original weights and optimizer tensor state are
checked unchanged; all clone weights are discarded. The preview batch
becomes the first committed update. The selected site is held for 50 steps.
Static and shuffled shadow controls incur the same lookahead work.
Reinvestment controls skip both audits and lookahead. Static reinvestment
uses the remaining credits on cyclic CF training; task reinvestment uses
ordinary training after the shared prefix. The latter is a task continuation,
not a task-only-from-initialization control.

Feedback-only length-four full roots are excluded from training draws,
with one rejection for seed 37 and none for 11/23. Monitor and final
evaluation have different lengths from training and different generator
seeds. All arms share the same elementary instruction family.
The monitor is used for adaptation and is development evidence only.

## Observed outcomes

All 15 fits learned 294/294 elementary transition argmaxes and obtained
512/512 correct soft argmax outcomes at every length 6,32,128,512.
Mean exact target probability across three fits at each length:

|Arm|6|32|128|512|Maximum row TV|
|---|---:|---:|---:|---:|---:|
|Adaptive|.98820672|.93883861|.78048103|.40527133|.00385624|
|Static shadow|.98820808|.93884324|.78052447|.40534581|.00374726|
|Shuffled shadow|.98821064|.93885475|.78053421|.40535515|.00371102|
|Static reinvest|.98973136|.94654129|.80553229|.44900452|.00318287|
|Task reinvest|.99326455|.96462012|.86710131|.57967980|.00203615|

Length-512 target mass by paired seed:

|Seed|Adaptive|Static shadow|Shuffled shadow|Static reinvest|Task reinvest|
|---|---:|---:|---:|---:|---:|
|11|.407648186|.407803524|.407818208|.451481163|.582016551|
|23|.403021130|.403048505|.403102590|.446673834|.577593463|
|37|.405144666|.405185415|.405144666|.448858569|.579429376|

Mean adaptive differences in probability units:
-.0000744872 vs static shadow, -.0000838273 vs shuffled shadow,
-.0437331940 vs static reinvest and -.1744084691 vs task reinvest.
The registered positive-all-seeds and +.01 mean screen FAILS.
All adaptive sites have positive one-step gains relative to the task clone.
Across 18 adaptive windows, the mean best-minus-worst site score spread
is 9.74935e-7, maximum 5.03027e-6. These are development point estimates
on a fixed monitor, not statistical confidence bounds. A one-step ordering
does not establish 50-step learning value. Static reinvest beats adaptive
in all seeds; task reinvest beats static reinvest in all seeds.

## Resource ledger

A credit counts one forward example-visit plus two per differentiable
forward example-visit as a backward proxy. Full patched rollouts are charged,
even when overwritten prefix intermediates contribute zero gradient.
This deliberately counts proxy work, not measured hardware FLOPs.

|Arm|Committed updates|Committed draws|Train F|Clone F|Audit F|Monitor F|Backward proxy|Credits|Unused|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Each shadow/adaptive|600|57600|302400|18432|10752|36864|320832|1010112|0|
|Static reinvest|644|61824|336192|0|0|0|336192|1008576|1536|
|Task reinvest|989|94944|336576|0|0|0|336576|1009728|384|

24 discarded clone updates per shadow/adaptive fit use 2304 root draws
on already committed preview batches. Audit/clone/monitor consume 102912
credits (10.19% of campaign cap). All unused credits are below the next
permitted complete update. Median campaign wall seconds: adaptive 1.3272,
static-shadow 1.4039, shuffled-shadow 1.3058, static-reinvest 1.3060,
task-reinvest 1.2988. Adaptive median audit .02133 s and clone+monitor
.08425 s. Timings include data generation and actual audit/clone costs
within the training loop; exclude model/optimizer initialization, final
evaluation and saving. Timings are sandbox descriptive, not an equal-wall
comparison. Whole process high-water RSS 359892 KiB, including PyTorch
runtime and all fits; per-fit RAM/VRAM and GPU throughput unavailable.
Common final evaluation uses 512*(6+32+128+512) forward example-visits
per fit, outside the selection budget.

Replay matched all 15 checkpoint payloads byte-for-byte, all non-timing
fit receipts, and the independent arithmetic budget recomputation. The
companion archive contains exact sources, both run receipts and weights.

## Meta-feedback: economic admission rule

Let X=(theta,optimizer,RNG,cursor), B a remaining resource budget,
and J an unsteered objective on an allowed development cohort. Define a
baseline continuation policy b and a feedback policy f which can spend
budget on audits, previews and selected training actions. The appropriate
campaign estimand is

    V_f(X,B)-V_b(X,B)
    = E[J(Train_f(X,B)) - J(Train_b(X,B))].

Both policies receive the same cap. The baseline reinvests any unused
audit allowance into productive updates. A positive one-step clone gain
G(u)=J(U_u(X))-J(U_0(X)) is not a sufficient condition for positive
campaign value. The value of information of the audit must be compared
against its opportunity cost.

If an audit costs c and an externally justified bound on decision gain
over a baseline action is U, and the lower bound on achievable
improvement from spending c on ordinary continuation is L, then U<=L
is sufficient to reject the audit under the additive comparison
assumption (common post-action value, no unaccounted interaction).
Without that assumption evaluate full matched-budget continuations.
Define a separate practical gate on robust incremental campaign gain
against a reinvestment control. Include monitor and clone expenses;
measure actual elapsed time and resource peaks before hardware claims.

Action score dispersion is a diagnostic for this gate, not a certified
upper bound on long-window gain. To avoid paying repeatedly for nearly
equivalent actions, next test multi-window prediction against held-out
development monitor gains, use a registered tolerance, and allow a
no-audit action. Consider tasks with localized causal faults and
heterogeneous action effects, e.g. BPD component repair or Lean proof
obligation recovery. The present explicit-state homogeneous MOD7
fixture cannot establish feedback benefit for those systems.
