# RV22-Z: native Distill audit and isolated decision-value probe

Parent RV22-Y 7290997e passed both hosted workflows. This experiment
returns to adaptive-evidence research-value learning. No private actor,
checkpoint training, external inference, GPU lease or paid model API.

## Frozen protocol

The runner checks the clean Git HEAD of the public upstream
MoralityLabAI/Compactification-Theory checkout at
7fa19c90372b13d71676e6fef13819d2a18b9be3. Run the actual native
Distill.research_curriculum.build with counts (48,16,32) and independently
audit every exported forced-action continuation value, policy replay,
one-step value, regret, baseline and split digest with RV22-Y. Keep native
referee rows evaluator-only. Publish only aggregate audit and probe
receipts. Native source contract and code pins are separate from data
admission. No reference-only rows become training examples.

Generate a **separate synthetic probe** via the pinned generator's
_data adaptive_evidence helper: 128 train roots (2026770000),
32 calibration (2026780000), 64 fresh standard/frontier eval (2026790000),
64 foundation-only eval (2026800000). Also evaluate the 32 native public
eval worlds; their referee records are not used as features or fit
targets. Require exact canonical exclusion from the entire native
reference export and among fresh panels. Native eval must belong to
the reference set. Hash each panel's sorted canonical-key digests.
Root-level exclusion does not establish general isomorphism separation.

Independently compute synthetic fit targets Q*(s,a)-Q1(s,a), divided
by reward scale, using the integrated exact engine for root and
one-test-descendant states from fresh train roots only. Verify Q1
against the public analytic calculation. Duplicate descendant
observations are permitted within each training root. Sixteen
public features, no optimal-continuation features. Fit one-hidden-layer
FF29 and tied tanh recurrence width 12/depth 3 using the same
40-epoch Adam schedule, seeds 29/31/37, 493 vs 492 active head MACs
per candidate. These are head-only counts; not end-to-end compute
matching or native TRM module training. Calibrate stopping margin
(0,.025,.05,.1) on 32 fresh calibration roots; freeze fits and
thresholds before evaluating any held-out policy. Compare stop,
one-step, receding two-step and exact DP. Roll out every candidate
policy from public state only; replay the full tree and deduct costs
along every path. Exact oracle used by offline evaluator only.
Paired root bootstrap 10000 resamples over fixed three-seed means.
No unseen-gym or recurrence-specific promotion. No model weights
exported or production training admission.

## Execution

Local regression: 2 passed, including exact XOR residual,
oracle-isolated actor execution, split exclusion, heldout binding,
and full smoke training. Local CPython environment cannot resolve
public GitHub hosts, so native execution is assigned to dedicated
read-only hosted CI, using the exact pinned public checkout.
Do not claim native results until the aggregate receipts are observed.

Measured locally tested source Git blob 14b4c7ee20189e028e19d9db52bf5b16ae29807d.
Test blob 1c75e255299a58cddd52de78201fe8f741015e10.
Local regression sources' SHA256:
301be946697d3b304e9c551fd1e645a24af8cc8cb113f3e6fdf40093d03854ae (runner)
e3ceca7bbb766de6c6de97f26b42c620cff1a77496aebffc7cf65612e6fb40f9 (test).

Reproduce on a CPU operator host with reviewed clean pinned checkout:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m datagoblin.research_v2_2.native_probe --distill-checkout "$DISTILL_CHECKOUT" --out /tmp/rv22z

Pending native actor, Jinn/Beast and Baby Goblin checkpoint gates
remain unchanged. This is a synthetic development probe only.
