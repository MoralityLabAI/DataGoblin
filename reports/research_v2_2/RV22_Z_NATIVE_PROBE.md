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


## Observed hosted result (complete)

Native-public job 140863715350, workflow run 46107595084, executed
commit c04a278cffae8ef8e6af8736e4e669b9ac37ce38 against the pinned upstream checkout.
Both hosted probe regression tests passed. Actual native export audit:
48/16/32 distinct exact roots; 408/136/272 actions; 96 roots and 816
action records total. Manifest ID 4b4de3a735e8a0c3f2d5aab5393ce4cd50b6bc0bbcf3a9ca6f706deb7c6d0695.
All native values, policies, regrets, baselines and split checks
passed. The reference-only custody is unchanged.

The separate probe fitted 6286 action examples from 128 synthetic
roots, calibrated on 32. FF thresholds .10/.10/.05; recurrent
thresholds .025/0/.025 (seeds 29/31/37). Public-only full policy replay
on 160 held-out roots. Mean realized net utility:

| Arm | Native eval (32) | Fresh eval (64) | Foundation (64) |
|---|---:|---:|---:|
| stop | 0.237039 | 0.370873 | 0.569700 |
| one_step | 4.818429 | 5.326028 | 4.594303 |
| two_step | 6.371971 | 6.824629 | 5.039959 |
| ff | 6.104807 | 6.586967 | 4.561260 |
| trm | 6.089762 | 6.541821 | 4.529500 |
| exact | 6.595800 | 7.054236 | 5.041161 |

Paired root bootstrap (95% percentile interval over fixed-seed means):

| Paired recurrent delta | Native | Fresh | Foundation |
|---|---|---|---|
| trm_vs_one_step | 1.2713 [0.5553, 2.0904] | 1.2158 [0.7602, 1.7236] | -0.0648 [-0.3581, 0.2439] |
| trm_vs_two_step | -0.2822 [-0.5645, -0.0065] | -0.2828 [-0.4731, -0.1029] | -0.5105 [-0.8003, -0.2680] |
| trm_vs_ff | -0.0150 [-0.1479, 0.1068] | -0.0451 [-0.1537, 0.0566] | -0.0318 [-0.1107, 0.0430] |

The learned residual improves on one-step search in the standard/
frontier cohorts. Two-step search remains superior, and the tied
recurrent head has no established advantage over the nearly head-MAC-
matched FF29. Foundation is an out-of-training difficulty cohort;
neither head improves on one-step mean utility there. This is evidence
against unguarded difficulty transfer, not a general capability result.
Do not claim full compute matching: feature construction, policy size,
and search overhead are not matched or timed. All three evaluation
panels are spent. No actor endpoint, native TRM training, language model
checkpoint change, or production training admission occurred.

Aggregate receipt: RV22_Z_NATIVE_PROBE_RECEIPT.json. Native export
and referee files remain separate; only aggregate artifacts uploaded.
Hosted run: https://github.com/MoralityLabAI/DataGoblin/actions/runs/46107595084
