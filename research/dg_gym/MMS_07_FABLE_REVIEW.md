# MMS-07 Fable review: mechanistic memory and optimizer-space feedback

Date: 2026-11-27. Reviewer: Fable (mathematical referee and algorithm designer).
Status: **REVIEW AND DESIGN ONLY. No gym training was run. No new learning-benefit evidence.**

Sources read at pinned revisions:

| Repository | Revision | Material |
| --- | --- | --- |
| DataGoblin | `c15012053f4db5bba7943a3fefdf818a9dfe9388` | MMS-06 implementation and results (PR #85) |
| DataGoblin | `ccd3ffa9de775218762e39ec541ceb8bb32d2014` | independent audit (PR #88); all code citations below are at this revision |
| LoopedTransformers | `3dad5bd9f7b076fffdad9c717dbc98b04982cba7` | original MMS (PR #397) |
| LoopedTransformers | `7eec3fab6ca7dabf62c7fc44c743522c5728dfa1` | canonicalization repair (PR #401) |
| BPD, Compactification-Theory, EvalMath | default-branch heads on 2026-11-26 | `OPTIMAL_BASIS_AND_MASKS.md`, `BPD_TRANSFER_ADDENDUM_20261118.md`, `causal_gauge_compactification/THEORY.md`, `nor50`/`nor51` memoization results, `evalmath-mechinterp-power` |

Companion files: `MMS_07_REGISTRATION_DRAFT.md` (preregistration), `mms07_math_checks.py` (29 executable checks),
`receipts/mms07_math_checks_receipt.json` (execution receipt, status PASS, script SHA-256 `355662ca…59eb89fa`).

Sections are numbered by problem: §1.4 is Problem 1, subsection 4; Problem 7 is the companion registration.

Evidence tags used throughout:

- **[P]** proved in this document.
- **[X: name]** verified by the named executed check in `mms07_math_checks.py`. Checks that exercise repository code
  import the pinned modules unmodified; the receipt records their SHA-256 (`factor_maps.py` `7ad66bb7…`,
  `mms_utility.py` `973739e4…`, both equal to the digests in `MMS_06_RESULTS.md`; `fast_gradient_bank.py` `65d1f304…`,
  the audit revision).
- **[R]** read from a committed receipt. **[D]** deduced from code or receipts without execution.
- **[S]** synthetic illustration on random networks or toy problems; not a gym measurement.
- **[L]** classical result recalled from the literature, not reopened for this review.
- **[C]** conjecture.

All negative MMS-01 to MMS-06 results are preserved as recorded. Nothing below reinterprets them as positive.

---

## Findings in brief

1. **Local cotangent alignment is the wrong admission test, and the right one is cheap.** The first-order loss effect of
   injecting $\delta$ at a site is $-\eta\langle q,\delta\rangle$, where $q = D_\theta z\,[P G]$ is the pushforward of
   the preconditioned gradient to the site, not the cotangent $c$ (Proposition 1.2). $c$ and $q$ agree in direction only
   when the site kernel is isotropic and the off-site term vanishes; Theorem 1.4 gives the tight cone condition. For an
   injection applied locally to the site weight, $q$ costs one layer matmul and the AdamW update difference is exact and
   elementwise (§1.7).
2. **At first order, memory cannot beat a memoryless rule.** The best injection of a given norm is $\delta\parallel q$,
   which uses only the optimizer state (Theorem 1.8). Under AdamW a perfectly aligned, full-coverage replay at a cut site
   is exactly a no-op by scale invariance (Proposition 1.7). Whatever replay does to learning therefore comes from
   partial coverage, misalignment and interference, which is sample and path reweighting plus noise. MMS-07 needs two
   memoryless controls (dose-only, pushforward-only) or it cannot attribute an effect to memory.
3. **An unbiased use of the same RAM exists.** A control-variate table over cached `(input, cotangent)` pairs gives an
   estimator that is unbiased for any staleness, compression or eviction pattern, with variance ratio $1-r^2$ measurable
   online before any intervention (Theorem 1.9). Whether it improves learning is a conjecture.
4. **The gradient bank hash aliases on structured keys.** With the documented RMP default shape (hidden 128, sequence 64,
   visits 8, middle visit) and power-of-two MiB tiers, the pinned hash maps 20,000 contiguous example IDs onto 2,892
   slots at every tier from 512 MiB to 4 GiB, and 2,048 IDs onto 698 slots (D1). A capacity sweep at those tiers would
   show no scaling contrast for an arithmetic reason. The 4 MiB LBF-20D configuration happens to be injective.
5. **The utility scheduler compares learning progress with zero.** Ordinary training has default gain 0 and is not
   explored while regret is falling; in a null world where every action is identical the pinned scheduler selects a
   memory action in 100% of windows (D2). Paired clones from a common state make ordinary's incremental value zero by
   construction and remove the learning-curve confound (§4).
6. **Factor geometry.** Activity must depend on the rank-one magnitude $m_j=\lVert u_j\rVert\lVert v_j\rVert$ relative
   to $\lVert\hat W\rVert_F$; identity must be a tolerance matching on the rank-one cosine, not a rounded hash; a
   cancelling pair wins top-k selection while contributing nothing to $W$; and for SVD maps $P=\Pi$ exactly, so the
   outer-product-versus-projector contrast is vacuous unless the map is non-orthogonal (§2).
7. **Staleness.** Cached cotangent error is bounded by a constant times parameter drift, with relative sensitivity that
   grows with the remaining recurrent depth $K-k$ (Theorem 3.1). The refresh period that maximizes value per compute
   follows a square-root law in the map decorrelation time and the refresh cost (Theorem 3.3).
8. **Economics.** A mechanistic action is worth its overhead only if it raises per-update progress by more than
   $\kappa_a-1$ times the ordinary per-update progress, where $\kappa_a$ is the step-cost ratio (Proposition 4.2). With
   the MMS-06 step-cost ratios at 1.5 s (1.38 to 1.43 for the dense and SVD arms, up to 2.06 for the refreshed sparse
   map), replay must speed per-update learning by about 40% to 106% merely to tie.
9. **Exact and approximate memory separate cleanly.** Exact caches cannot change outputs, so any retention policy,
   including one driven by an unsound neural predictor, preserves verifier soundness (Theorem 6.1). Approximate memory
   changes the trajectory and needs causal evaluation.
10. **MMS-06 evidence remains development evidence.** The three ordinary points in the receipts are non-decreasing in
    update count (1.0035 at 169.7 updates, 1.0119 at 400, 1.1256 at 964), and every memory arm at 1.5 s ran fewer updates
    than ordinary (at most 123.7 against 169.7). The receipts therefore do not show that 1.5 s is a pre-overfitting
    regime; they are compatible with regret rising in update count across the whole observed range (D9).

Disposition: no promotion. MMS-07 as registered in the companion draft tests replay against an early-stopping-protected
ordinary comparator with memoryless controls, world-level replication and a registered equivalence margin.

---

## Verified defects and counterexamples

Items already reported by the independent audit are marked *(audit)* and are listed only where this review adds a
reproduction or a consequence.

| ID | Defect | Evidence | Consequence |
| --- | --- | --- | --- |
| D1 | `FastGradientBank._slots` (and LT `GradientBank._indices`) reduce `key * 6364136223846793005` wrapped to int64 modulo the capacity. On arithmetic key sets the occupied-slot count departs strongly from a random hash. LBF-20D keys $3i+1$, $N=12{,}288$: 2,192 distinct slots at 1.5 MiB (random model 7,377), 203 at capacity 53,237; 4.9% of capacities in 1,000 to 60,000 give under half the random-model occupancy. RMP default keys $8i+4$ with 16,400-byte rows: capacity is exactly $(\text{MiB}/16)\cdot 1023$ at power-of-two tiers; 20,000 IDs occupy 1,023 slots at 128 MiB and 2,892 at 512, 2,048 and 4,096 MiB; 2,048 IDs occupy 698 at every tier. Capacity minus one gives full occupancy in each case. | [X: p5_hit_rate_laws] | RAM-tier sweeps confound capacity with aliasing. The LT recommendation to sweep 128/512/2048/4096 MiB on RMP would report "no scaling contrast" for arithmetic reasons. The MMS-06 4 MiB tier is injective, so MMS-06 hit counts are unaffected. |
| D2 | `UtilityDirectedScheduler.select` exploits a memory action whenever its EMA gain exceeds `max(0, ordinary EMA)`. Ordinary starts at 0 and is not explored. Feedback is the raw window regret change. In a null world with identical actions and a falling regret curve, 300 simulated runs of 10 windows select `block@0.5` in every window and never select ordinary. | [X: p4_scheduler_null_simulation] | The scheduler measures learning progress, not treatment effect. Consistent with the fixed-exposure receipt, where the scheduler arm's lookup CPU is 95.8% of the always-on `svd_refreshed` arm [R, X: p7_receipt_deductions]. |
| D3 | Utility is `gain / cost`. For equal negative gains the costlier action ranks higher; the exploration branch picks the 4 MiB, 10 s action over the 0.5 MiB, 1 s action. | [X: p4_ratio_utility_negative_numerator] | Ratio utilities are order-preserving only for positive numerators. |
| D4 | `canonicalize_factors` sets `active = (‖u‖ > eps) & (‖v‖ > eps)`. Factors (1, 1) are active; (1e13, 1e-13) in either order are inactive. Factors (1e200, 1e-200) of the all-ones matrix give `np.linalg.norm = inf` and a silently dropped component. *(audit, reproduced and extended)* | [X: p2_activity_gauge_defect, p2_extreme_scale_imbalance] | Activity is gauge dependent. |
| D5 | `_component_id` hashes `np.round(x, 4).tobytes()`. (i) A $\pm 10^{-6}$ entry gives $+0.0$ or $-0.0$ and different IDs. (ii) The sign pivot `argmax(abs(u))` flips at near-ties and changes the ID. (iii) Distinct `v` within $2\times10^{-5}$ alias. (iv) A $2\times10^{-8}$ change across a rounding boundary changes the ID. LT hashes `round(x*1e4).to(int64)` bytes, so the two repositories assign different IDs to the same component. *(aliasing: audit; the rest new)* | [X: p2_component_id_instability]; LT representation [D] | Rounded hashes are neither injective nor continuous. `basis_digest` changes at essentially every refresh. |
| D6 | A cancelling pair $(u,v),(u,-v)$ with large $\lVert v\rVert$ contributes zero to $W$ yet is active, scores highest, and fills the top-2 selection. The injected direction is one $W$ does not write to. | [X: p2_selection_scale_and_cancelling_pair] | Selection can be captured by gauge artifacts in non-orthogonal maps. |
| D7 | Tie-breaking rounds scores to 6 absolute decimals. Rescaling the layer input by $10^{-8}$ changes the selected components. | same check | Selection is not invariant to positive rescaling of activations. |
| D8 | $P=U_S^\top U_S$ and $\Pi$ give opposite admission signs on explicit inputs (two components at 60°); the canonical SVD map has $U U^\top = I$ to $2\times10^{-8}$, so $P=\Pi$ there. | [X: p2_outer_sum_vs_projector] | The `svd_*` arms cannot distinguish the two operators. |
| D9 | Ordinary final-development regret is non-decreasing over the three update counts in the receipts; all memory arms at 1.5 s ran fewer updates than ordinary. | [R, X: p7_receipt_deductions] | The 1.5 s comparison is not shown to be free of the update-count confound. |
| D10 | In `mms06_lbf20d_campaign.py` no `state_signature` is passed and `record_coordinate_delta` is never called; each trajectory ID holds one action. Repair retrieval, coordinate history and eligibility decay were inert. | [D] | The `utility_scheduler` arm tested a three-action bandit, not the memory features of `mms_utility.py`. |
| D11 | The SNR sketch receives `(c_hit.mean(0) @ fmap.u.T)[:8]`: the first eight components in hash order, under maps refreshed every 64 steps with a constant `basis_digest="block_svd"`. | [D] | The online covariance mixes coordinates from different bases. Component-coordinate statistics need transport or physical coordinates (§3.3). |
| D12 | The shuffled-feedback control randomizes the sign of the delta, which changes the reward distribution. *(audit)* | [D] | Use label permutation or a yoked schedule. |
| D13 | `refresh` timings of 9.375 and 4.6875 ms are 6 and 3 ticks of 15.625 ms over 10 calls. | [R, D] | Resolution of the refresh figure is ±1.56 ms. |
| D14 | `fit_sparse_factors` penalizes `v.abs().mean() + u.abs().mean()`, which is not invariant under reciprocal scaling. | [D] | The fit selects a gauge; minor once canonicalization is invariant. |

Counterexamples constructed or reproduced:

| ID | Claim refuted | Construction | Evidence |
| --- | --- | --- | --- |
| CE1 | Positive local alignment implies parameter-space descent | registered scalar example; $L$: 1.105 → 1.1041 ordinary, 1.1061 steered; $q=Kc+\rho=1-1.1$ | [X: p1_scalar_counterexample] |
| CE2 | The failure needs tied parameters | untied cut site, $z=A\theta$, $A=\mathrm{diag}(1,10)$; local cosine 0.845, $\langle g,d\rangle=-0.8$ | [X: p1_anisotropy_counterexample] |
| CE3 | A local (truncated) certificate transfers to hook-style full propagation under contraction | $h_1=(1+w)x$, $z_1=wh_1$, $w=-0.6$: local derivative $+0.4$, total $-0.2$, with $\lvert\partial z_1/\partial z_0\rvert=0.6<1$ | [X: p1_tied_recurrence_local_vs_full] |
| CE4 | Increasing a gradient coordinate increases the Adam step along it | nine steps of gradient 1, then 3 at $t=10$: sensitivity $-0.048$; $u(3.0)=0.973$, $u(3.6)=0.943$ | [X: p1_adam_sensitivity] |
| CE5 | Alignment with a truncated training gradient implies alignment with the true gradient | planar configuration at $\alpha+\beta=\pi/2$ | [X: p1_truncated_gradient_angle_budget] |
| CE6 | Per-sample collinear amplification is descent on the batch objective | synthetic: 8.9% of samples have $\langle q_b,\rho c_b\rangle<0$ | [S, X: p1_local_alignment_failure_rate_synthetic] |
| CE7 | Replay stability is delay independent beyond $\eta\lambda(1+\rho)<2$ | $a=1.75,\rho=0.2$: unstable at delays 2, 4, 6; stable at 1, 3, 5; plain GD radius 0.75 | [X: p3_delay_independent_stability] |
| CE8 | $P$ and $\Pi$ admit the same samples | 60° pair, both false accept and false reject | [X: p2_outer_sum_vs_projector] |

---

## Problem 1: optimizer-space gradient steering

### 1.1 Setting

Parameters $\theta\in\mathbb R^p$; batch loss $L(\theta)$; intervention site $z=z(\theta)\in\mathbb R^n$ (all samples and
positions of one module output at one visit, stacked). Write

$$c=\frac{\partial L}{\partial z},\qquad A=D_\theta z\in\mathbb R^{n\times p}\ \text{(total Jacobian, all upstream paths)},\qquad
r=\nabla_\theta L\big|_{z\ \text{held fixed}},\qquad g=A^\top c+r .$$

In a tied recurrent network with visits $k=0,\dots,K-1$ of one module, $z_k=W x_k+b$, let $J_k$ be the immediate
Jacobian of $z_k$ in $\theta$ (inputs fixed) and $T_{m\leftarrow j}=\partial z_m/\partial z_j$. Then
$A_m=\sum_{j\le m}T_{m\leftarrow j}J_j$ plus the contribution of non-site upstream parameters, and $r$ collects skip
paths and the direct contributions of later visits.

**Lemma 1.1 (hook semantics) [P, X: p1_adjoint_identity_tied_net].** A tensor hook that returns $c+\delta$ at the site
produces the parameter gradient $g+d$ with $d=A^\top\delta$. Equivalently, the hook computes the exact gradient of the
surrogate $L(\theta)+\langle\bar\delta,z(\theta)\rangle$ with $\bar\delta$ held constant.

*Proof.* Reverse mode is linear in the cotangent at the site and everything upstream of the site receives the modified
cotangent; nothing downstream does. $\square$ The check confirms $g_{\text{hook}}-g=A^\top\delta$ to $10^{-12}$ on a
3-visit tied residual tanh network (the `recur3` topology) and the adjoint identity
$\langle v,A^\top\delta\rangle=\langle D_\theta z[v],\delta\rangle$ to relative error $6\times10^{-15}$.

This is the semantics of `FastMechanisticHook` and of LT `MechanisticMemory`: the site is the middle visit of a tied
block, so $d$ includes the visit-0 contribution of the tied weight and the entry layer [D].

### 1.2 The first-order identity

Let the optimizer step be $\theta^+=\theta-\eta P(\cdot)$ with $P\succ0$ fixed, and let $G$ be the gradient of the
objective of interest (the batch, the training set, or the population).

**Proposition 1.2 [P, X: p1_first_order_prediction_and_dominance].**

$$L(\theta-\eta P(g+d))-L(\theta-\eta Pg)=-\eta\langle q,\delta\rangle+O(\eta^2),\qquad q:=A\,P\,G=D_\theta z\,[PG].$$

*Proof.* $\langle G,Pd\rangle=\langle G,PA^\top\delta\rangle=\langle APG,\delta\rangle$. $\square$

$q$ is a Jacobian-vector product: minus the first-order displacement of the site activation under the ordinary update,
per unit learning rate. The steered step beats the ordinary one to first order if and only if $\delta$ points along the
direction in which the optimizer is already moving the site activation. The check reproduces the predicted change to
0.5% at $\eta=10^{-4}$ for collinear, $q$-aligned, anti-aligned and $q$-orthogonal injections.

**Proposition 1.3 (second-order sufficient condition) [P, X: p1_smoothness_sufficient_condition].** If $L$ is
$L_s$-smooth then

$$L(\theta-\eta P(g+d))-L(\theta-\eta Pg)\le-\eta\langle G,Pd\rangle+\eta^2L_s\lVert Pd\rVert\Big(\lVert Pg\rVert+\tfrac12\lVert Pd\rVert\Big),$$

so $\langle q,\delta\rangle>\eta L_s\lVert Pd\rVert(\lVert Pg\rVert+\frac12\lVert Pd\rVert)$ suffices for improvement.

*Proof.* Apply the descent lemma on the segment from $\theta_1=\theta-\eta Pg$ to $\theta_1-\eta Pd$ and bound
$\lVert\nabla L(\theta_1)-G\rVert\le L_s\eta\lVert Pg\rVert$. $\square$

This is weaker than the condition in `MMS_06_INDEPENDENT_REVIEW.md`,
$\langle g,d\rangle>\frac{L\eta}{2}(\lVert g+d\rVert^2+\lVert g\rVert^2)$: on 40,000 random quadratics the present
condition held in 32,983 cases and the audit's in 25,147, the audit's never held where this one failed, and this one
was never violated.

### 1.3 Decomposition and failure modes

With $G=g$,

$$q=K c+\rho,\qquad K:=APA^\top\ \text{(site kernel in the optimizer metric)},\qquad \rho:=APr\ \text{(off-site pushforward)}.$$

$\langle q,\delta\rangle=\delta^\top Kc+\delta^\top\rho$. Local alignment $\langle\delta,c\rangle>0$ controls neither
term. Five distinct failure modes follow.

- **F1, interference.** $\rho$ dominates. In CE1, $K=1$, $c=1$, $\rho=-1.1$. For tied weights
  $\rho=\sum_{k>m}K_{mk}c_k+\dots$, the later-visit cotangents pulled back through the cross-visit kernel: the visits
  disagree on how the shared weight should move.
- **F2, anisotropy.** $K\not\propto I$ on $\mathrm{span}\{c,\delta\}$ (CE2). No tying is needed.
- **F3, batch heterogeneity.** $K$ is not block diagonal across samples. Even $\delta_b=\beta_bc_b$ gives
  $\sum_b\beta_b\langle g_b^{\text{site}},Pg\rangle$, which is negative when up-weighted samples oppose the batch (CE6).
  Per-sample collinear amplification is descent on a reweighted objective.
- **F4, optimizer nonlinearity.** Adam's response to an added gradient component can be zero or negative (§1.6, CE4).
- **F5, truncation.** A certificate against a truncated gradient consumes angular budget (§1.5, CE5).

### 1.4 The weakest spectral condition

**Theorem 1.4 (cone condition, tight) [P, X: p1_cone_bound_tightness].** Let $S=\mathrm{span}\{c,\delta\}$, let
$0<\mu\le M$ be the extreme eigenvalues of the compression of $K$ to $S$, $\kappa=M/\mu$, and $\varphi=\angle(c,\delta)$.
Then

$$\langle q,\delta\rangle\ \ge\ \lVert\delta\rVert\lVert c\rVert\Big[\tfrac{M+\mu}{2}\cos\varphi-\tfrac{M-\mu}{2}\Big]-\lVert\delta\rVert\,\lVert\rho\rVert .$$

Hence $\langle q,\delta\rangle>0$ whenever

$$\cos\varphi>\frac{\kappa-1}{\kappa+1}+\frac{2\lVert\rho\rVert}{(M+\mu)\lVert c\rVert}.$$

The kernel part is tight: for every $\varphi$ with $\cos\varphi\le(\kappa-1)/(\kappa+1)$ there are $K,c,\delta$ with
that spectrum and angle and $\delta^\top Kc\le0$.

*Proof.* On $S$ write $K_S=\frac{M+\mu}{2}I+E$ with $\lVert E\rVert\le\frac{M-\mu}{2}$; then
$\delta^\top Kc\ge\frac{M+\mu}{2}\langle\delta,c\rangle-\frac{M-\mu}{2}\lVert\delta\rVert\lVert c\rVert$. For
tightness take $K_S=\mathrm{diag}(M,\mu)$, $c$ at angle $(\pi-\varphi)/2$ from the first axis and $\delta$ its
reflection across the second axis; then $E c=-\frac{M-\mu}{2}\lVert c\rVert\hat\delta$ and equality holds. $\square$

The threshold is the sine of the largest turning angle of $K$ (first antieigenvalue $2\sqrt{\mu M}/(\mu+M)$,
Kantorovich/Gustafson [L]). Consequences:

- No condition phrased through $\varphi$ and the spectrum of $K$ on $S$ alone can be weaker.
- The MMS rule $\cos\varphi>0$ is sufficient only if $\kappa=1$ on $S$ and $\delta^\top\rho\ge0$.
- In CE1 the bound reads $1>0+2(1.1)/2$, which fails; with a second cotangent of $-0.9$ it holds and the step improves.
- 20,000 random instances gave minimum normalized slack $2.4\times10^{-8}$ and the tight instance a residual of
  $10^{-16}$.

Site kernels of trained networks are far from isotropic, so in the worst case only near-collinear injections are safe,
and collinear injections are sample reweighting (F3).

### 1.5 Tied recurrence, injection truncation and truncated training gradients

Let $d^{(\tau)}=\sum_{j=m-\tau}^{m}J_j^\top T_{m\leftarrow j}^\top\delta$ be the injection propagated through $\tau$
earlier visits; $\tau=0$ is the local injection $d^{(0)}_W=\delta^\top X_m$, $d^{(0)}_b=\sum_b\delta_b$.

**Theorem 1.5 [P, X: p1_adjoint_identity_tied_net].**
$\lVert d^{(\infty)}-d^{(\tau)}\rVert\le\lVert\delta\rVert\sum_{j<m-\tau}\lVert T_{m\leftarrow j}\rVert\lVert J_j\rVert$.
If $\lVert T_{m\leftarrow j}\rVert\le\Lambda\lambda^{m-j-1}$ with $\lambda<1$ and $\lVert J_j\rVert\le a$, the right side
is at most $\lVert\delta\rVert a\Lambda\lambda^{\tau}/(1-\lambda)$, and

$$\langle Pg,d^{(\infty)}\rangle\ \ge\ \langle q^{(\tau)},\delta\rangle-\lVert Pg\rVert\lVert\delta\rVert\frac{a\Lambda\lambda^{\tau}}{1-\lambda}.$$

*Proof.* $d^{(\infty)}-d^{(\tau)}=\sum_{j<m-\tau}J_j^\top T_{m\leftarrow j}^\top\delta$; take norms, sum the geometric
series, and apply Cauchy–Schwarz to $\langle Pg,d^{(\infty)}-d^{(\tau)}\rangle$. $\square$

For the residual tanh block, $T_{m\leftarrow j}=W D_m\prod_{i=j+1}^{m-1}(I+W)D_i$ with $D_i=\mathrm{diag}(1-h_i^2)$, so
$\Lambda=\lVert W\rVert$ and $\lambda=\lVert I+W\rVert$, which is generally not below 1. The check confirms the bound
with $m=2$ (largest observed gap 23% of the bound).

Contraction alone does not transfer a local certificate (CE3): the margin $\langle q^{(\tau)},\delta\rangle$ must exceed
the truncation term. The practical consequence is to **inject locally** ($\tau=0$). The certificate is then exact, costs
one matmul (§1.7), and has no dependence on products of Jacobians. Neither form is principled with respect to $L$; the
hook form is the exact gradient of an auxiliary linear loss and the local form is a truncation of it. The certificate is
what carries the guarantee.

**Theorem 1.6 (truncated training gradients) [P, X: p1_truncated_gradient_angle_budget].** Let $g^{\text{tr}}$ be the
training gradient under truncation (TRM final-recursion differentiation, one-step gradients, detached carries) with
$\angle(g^{\text{tr}},G)\le\beta$. If $\angle(d,g^{\text{tr}})\le\alpha$ and $\alpha+\beta<\pi/2$ then
$\langle d,G\rangle\ge\lVert d\rVert\lVert G\rVert\cos(\alpha+\beta)>0$. The bound is tight in the plane.

*Proof.* Angles between directions obey the triangle inequality on the sphere:
$\angle(d,G)\le\angle(d,g^{\text{tr}})+\angle(g^{\text{tr}},G)$. $\square$

If $\lVert g^{\text{tr}}-G\rVert\le\varepsilon\lVert G\rVert$ then $\beta\le\arcsin\varepsilon$, so the certificate
threshold on a truncated-gradient model is $\cos\alpha>\varepsilon$ rather than $\cos\alpha>0$. Under contraction with
truncation after $\tau$ steps, $\varepsilon\lesssim\lambda^{\tau+1}/(1-\lambda)$.

Multiple sites add at first order, $\langle Pg,d\rangle=\sum_s\langle q_s,\delta_s\rangle$, and interact at second order
through $K_{ss'}=A_sPA_{s'}^\top$. Visits of one tied module have strongly correlated Jacobians, so doses at several
visits add coherently; the joint budget is $\sum_{s,s'}\delta_s^\top K_{ss'}\delta_{s'}$, not a per-site norm.

### 1.6 AdamW

With $m=\beta_1m^-+(1-\beta_1)g$, $v=\beta_2v^-+(1-\beta_2)g^2$, $\hat m=m/(1-\beta_1^t)$, $\hat v=v/(1-\beta_2^t)$ and
$u=\hat m/(\sqrt{\hat v}+\epsilon)$:

**Lemma 1.7a (sensitivity) [P, X: p1_adam_sensitivity].** Coordinatewise,

$$\frac{\partial u}{\partial g}=\frac{a_t}{\sqrt{\hat v}+\epsilon}-\frac{b_t\,\hat m\,g}{\sqrt{\hat v}\,(\sqrt{\hat v}+\epsilon)^2},\qquad
a_t=\frac{1-\beta_1}{1-\beta_1^t},\quad b_t=\frac{1-\beta_2}{1-\beta_2^t}.$$

For $\epsilon\to0$ the sensitivity is negative exactly when $\hat m g/\hat v>a_t/b_t$. At $t=1$ it is zero: the first
Adam step is sign descent and ignores the magnitude of an added component. The ratio $b_t/a_t$ is 1 at $t=1$, 0.654 at
10, 0.204 at 50, 0.105 at 100, 0.030 at 400, 0.016 at 1000 and 0.010 asymptotically. All MMS-06 LBF-20D runs (mean update counts at
most 964) lie inside the $\beta_2$ warm-up window $t\lesssim1/(1-\beta_2)=1000$ [R, D]. The closed form matches
finite differences to $10^{-7}$.

**Proposition 1.7 (Adam null) [P, X: p1_adam_scale_null].** For $\epsilon\to0$ and decoupled weight decay, Adam's
update is homogeneous of degree zero in the gradient history. If at every step the steered gradient equals
$(1+\rho)g$, the trajectory is identical to ordinary AdamW.

*Proof.* $m$ and $\sqrt v$ both scale by $1+\rho$, so $u$ is unchanged at every step, and decoupled decay does not
see the gradient. $\square$

This is the case of a perfectly aligned replay ($\delta=\rho c$) at a cut site ($r=0$) that covers every sample. The
check gives a trajectory difference of $10^{-13}$ when all samples are scaled by 1.2 and $2.6\times10^{-3}$ when 60%
are. Under AdamW the "dose" of an ideal replay is therefore invisible, and every effect of replay is carried by its
departures from ideal: partial hit coverage (hit samples are up-weighted relative to misses), misalignment (a component
orthogonal to $c$), and non-cut paths.

**Exact one-step certificate.** Because $d$ changes $m$ and $v$, the relevant object is
$\Delta u=u(g+d;\text{state})-u(g;\text{state})$, computed exactly by evaluating the update formula twice, $O(p)$
elementwise on the affected parameters and no passes. Weight decay cancels. The fraction $1-\beta_1$ of $d$ acts now
and the rest over about $1/(1-\beta_1)$ later steps, against gradients $g_{t+k}$ not yet seen. Under slow drift those
are best predicted by $\hat m_{t-1}$, which is also independent of the current minibatch. The reference direction for
the certificate is therefore the previous update direction
$\tilde u=\hat m_{t-1}/(\sqrt{\hat v_{t-1}}+\epsilon)$, and the predicted marginal improvement is

$$\widehat{\Delta J}_t=\eta_t\langle g_{\text{ref}},\Delta u_t\rangle-\frac{\eta_t^2}{2}\Big[\lVert u+\Delta u\rVert_{\hat H}^2-\lVert u\rVert_{\hat H}^2\Big],$$

with $g_{\text{ref}}\in\{\hat m_{t-1},\ \text{a probe gradient from the feedback split}\}$ and $\hat H$ a diagonal
empirical-Fisher or K-FAC proxy. Using the current minibatch $g$ as $g_{\text{ref}}$ is optimistic, since $d$ is built
from the same batch.

Gradient clipping (RMP uses `clip_grad_norm_(1.0)` [D]) changes the clip factor when $d$ is added; by Proposition 1.7
this is immaterial under Adam except through $\epsilon$ and the $v$ history.

### 1.7 Certificates and their costs

Notation: batch $B$, positions $T$, site output width $n$, input width $n_{in}$, components $C$, selected $k$, forward
cost $F$ (backward about $2F$), prefix fraction $f_m=(m+1)/K$.

| Estimator | Certifies | Extra cost per step | Notes |
| --- | --- | --- | --- |
| Local alignment $\langle\delta_b,c_b\rangle>0$ (current) | nothing in parameter space | $O(BTn)$ | Theorem 1.4 |
| **Local pushforward** $q_b=\tilde U_Wx_b+\tilde u_b$ | $\langle\tilde u,d^{(0)}\rangle=\sum_b\langle\delta_b,q_b\rangle$ exactly, per sample | one matmul, $O(BTn\,n_{in})$ | uses optimizer state already in RAM; same order as the hook's existing `x @ V` |
| Explicit $d^{(0)}_W=\delta^\top X_m$ and exact Adam $\Delta u$ | exact one-step update change on site parameters | $O(BTn\,n_{in}+n\,n_{in})$ | cost of one layer weight-gradient |
| Prefix JVP of $z_m$ along $\tilde u$ | exact $q$ for hook-style propagation | $\approx f_m(1\text{–}2)F$, amortizable every $r_J$ steps | forward mode during the forward pass; available before backward |
| Second cotangent through the prefix | $g$ and $d$ separately, any metric | $\approx 2f_mF$ | batched cotangent |
| Probe gradient on the feedback split | alignment with held-out gradient | one small forward/backward every $r_p$ steps | independent of the training batch |
| Rank-one sketches $(a_l\!\cdot\!c)(b_l\!\cdot\!x)$, $l\le r$ | approximate per-sample gradient inner products | $O(BTr(n+n_{in}))$ | storable in the bank at $r$ numbers per row |
| K-FAC/Gauss-Newton quadratic for the dose | optimal scale $\rho^*$ on a quadratic model | $O(n\,n_{in}(n+n_{in}))$ per evaluation | heuristic curvature |
| BPD component sketch $\Gamma=\hat UG_W\hat V$ | gradient energy inside the map, $\chi=\lVert\hat UG_W\rVert_F^2/\lVert G_W\rVert_F^2$ | $O(Cn\,n_{in})$ | $\gamma_j=\hat u_j^\top G_W\hat v_j$ is the gradient in the component amplitude; $m_j\gamma_j=\partial L/\partial\log m_j$ |

Two remarks on the BPD decomposition. First, the current score, mean $\lvert x\!\cdot\!\tilde v_j\rvert$ times mean
$\lvert h\!\cdot\!\hat u_j\rvert$, equals $\lvert\partial\ell_{\text{hist}}/\partial\log m_j\rvert$ when $T=1$ and is a
different quantity when $T>1$, because a product of means of absolute values is not the absolute value of a mean of
products. Second, an injection restricted to $\mathrm{span}(U_S)$ can recover at most the fraction
$\lVert\Pi_{U_S}q\rVert/\lVert q\rVert$ of the first-order optimum. Map reconstruction cosine measures the energy of
$W$ in the map; $\chi$ and $\lVert\Pi_{U_S}q\rVert/\lVert q\rVert$ measure the energy of the gradient in it. Only the
latter bounds usefulness.

On random-initialized tied networks the batch cosine between $c$ and $q$ has median 0.47 (5% to 95%: 0.39 to 0.55), and
a collinear injection attains 61% of the $q$-aligned first-order gain in the single instance checked [S]. Batch-level
sign reversal was rare unless $\cos\varphi\le0.1$ (13.75% of trials for hook propagation, 5.75% local), because
orthogonal components average out across the batch; per-sample reversal was common (41.6% at $\cos\varphi=0.1$, 15% at
0.6, 8.9% at 1.0). The practical failure is inefficiency and added variance more often than outright ascent.

### 1.8 First-order dominance

**Theorem 1.8 [P, X: p1_first_order_prediction_and_dominance].** Fix $\theta$, $P$ and the true gradient $G$. Over
$\lVert\delta\rVert\le R$ the first-order gain $\eta\langle q^*,\delta\rangle$, $q^*=APG$, is maximized by
$\delta^*=Rq^*/\lVert q^*\rVert$; with per-sample budgets $\lVert\delta_b\rVert\le R_b$ the maximizer is
$\delta_b^*=R_bq_b^*/\lVert q_b^*\rVert$. Any other rule attains the fraction $\cos\angle(\delta,q^*)$. A memory-based
rule exceeds the memoryless plug-in $\hat\delta\parallel\hat q=AP\hat g$ only if
$\cos\angle(\delta_{\text{mem}},q^*)>\cos\angle(\hat q,q^*)$.

*Proof.* Cauchy–Schwarz on a linear objective that separates over samples. $\square$

Corollaries.

- In full-batch training $\hat g=G$ and no memory rule beats $\delta\parallel q$ at first order.
- A cached cotangent for example $i$ is a function of the same example at an earlier $\theta$. It carries information
  about $G$ beyond the current minibatch only through other examples, which same-key replay never uses: the injected
  direction for sample $b$ is built from sample $b$'s own history and pushed through sample $b$'s own Jacobian.
- Any first-order value of memory is variance reduction: a better estimate of $q^*$ than the optimizer's own momentum.
- Components of $\delta$ orthogonal to $q$ have zero first-order effect and nonnegative second-order cost.

Beyond first order, history could carry curvature (secant pairs $(\Delta\theta,\Delta c)$, as in quasi-Newton methods).
That is a different algorithm from replay and is not analysed here [C].

### 1.9 An unbiased alternative: the control-variate table

Let $g_i$ be per-example gradients of some parameter block, $t_i$ arbitrary table entries (stale, compressed, zero for
missing or evicted keys) that do not depend on the current batch draw, $\bar t=\frac1N\sum_it_i$, and a batch
$\mathcal B$ of size $B$ drawn uniformly with replacement.

**Theorem 1.9 [P, X: p1_control_variate_table].** For any $\alpha$ independent of the draw,

$$\hat g_\alpha=\frac1B\sum_{b\in\mathcal B}(g_b-\alpha t_b)+\alpha\bar t$$

satisfies $\mathbb E\hat g_\alpha=\bar g$, and
$\mathrm{Var}\,\hat g_\alpha=\frac1B[\sigma_g^2-2\alpha\sigma_{gt}+\alpha^2\sigma_t^2]$ in trace form, minimized at
$\alpha^*=\sigma_{gt}/\sigma_t^2$ with value $\frac{\sigma_g^2}{B}(1-r^2)$, $r^2=\sigma_{gt}^2/(\sigma_g^2\sigma_t^2)$.

*Proof.* Table entries are fixed before the draw, so $\mathbb E[\frac1B\sum_bt_b]=\bar t$ and the correction has mean
zero; the variance is that of a mean of $B$ independent draws of $g_i-\alpha t_i$. $\square$

This is the SAGA estimator with a scaled table (Defazio, Bach and Lacoste-Julien, 2014 [L]). The Monte Carlo check with
30% missing entries gives bias within one standard error and variance ratio 0.5753 against $1-r^2=0.5756$.

Site instantiation. The direct gradient of a linear site weight at visit $m$ is $\sum_tc_{i,t}x_{i,t}^\top$. Store
$(x_i,c_i)$, or rank-$r$ component coordinates $(V^\top x_i,U_Sc_i)$, per key; maintain $\Sigma=\sum_it_i$ on write
(subtract the overwritten slot, add the new one); use $\hat g_W=g_W-\alpha[\frac1B\sum_{b\in\text{hits}}t_b-\Sigma/N]$.

Properties that bear on Problems 2, 3 and 5:

- Unbiasedness holds for every staleness, hash collision, eviction pattern, basis drift and compression error, provided
  $\Sigma$ is consistent with the slots and reads precede writes. These degrade $r^2$, not correctness.
- $r^2$ as a function of entry age is observable for free on every hit, because the batch supplies both the old and the
  new value. If $r^2$ at the typical hit age is small, the method has a built-in STOP: $\alpha^*\to0$ recovers ordinary
  training.
- The factor map becomes a codec for the table rather than a selector of directions; gauge questions do not affect
  correctness because $\Sigma$ lives in physical parameter coordinates.
- Non-uniform sampling (RMP frontier curriculum) needs per-stratum sums weighted by inclusion probabilities.
- Per-step cost is $O(BTn\,n_{in})$; memory is $N\cdot T(n+n_{in})$ BF16 values, or $2NTr$ with rank-$r$ coordinates.

Learning benefit is a conjecture. Variance-reduced optimizers have a poor record in deep networks (Defazio and Bottou,
2019 [L]) because iterates move faster than tables stay correlated. The regime here (small fixed pools, no augmentation,
one linear site) is the favourable one, and the $r^2$ diagnostic settles the question before any outcome is opened.

---

## Problem 2: BPD factor geometry and gauge invariance

### 2.1 Gauge structure

$W=(VU)^\top=\sum_ju_jv_j^\top$, with $u_j\in\mathbb R^{d_{out}}$ the rows of $U$, $v_j\in\mathbb R^{d_{in}}$ the columns
of $V$, and rank-one components $R_j=u_jv_j^\top$ of magnitude $m_j=\lVert R_j\rVert_F=\lVert u_j\rVert\lVert v_j\rVert$.

**Lemma 2.1 [P].** If $V,V'$ have full column rank $C$, $U,U'$ have full row rank $C$ and $VU=V'U'$, there is a unique
$R\in GL(C)$ with $V'=VR$, $U'=R^{-1}U$. If the factors are rank deficient (duplicates, cancelling pairs), the fibre
$\{(V,U):VU=M\}$ is strictly larger than one $GL(C)$ orbit.

*Proof.* $U,U'$ are surjective, so $\mathrm{col}(V')=\mathrm{col}(V'U')=\mathrm{col}(VU)=\mathrm{col}(V)$ and
$V'=VR$ with $R$ invertible; then $V(RU'-U)=0$ and injectivity of $V$ give $U'=R^{-1}U$. $\square$

**Lemma 2.2 [P, X: p2_gl_mixing].** The multiset $\{R_j\}$ is invariant exactly under the monomial subgroup
(nonzero diagonal scalings, signs, permutations). A generic $R$ preserves $W$ and the full-span projector
(difference $5\times10^{-8}$ in the check) and changes the component set (no shared IDs), every proper sub-selection
(top-2 projector difference 0.44) and the outer-product sum $U^\top U$ (difference 0.68).

By the BPD representation-invariant mask criterion (`OPTIMAL_BASIS_AND_MASKS.md`), a mask $D$ is invariant under a
relative gauge $R$ if and only if $DR=RD$; the check confirms a block mask is invariant under a block gauge and not
under a generic one. Top-k selection is a diagonal mask in component coordinates, so it is meaningful only relative to
a map whose component set is itself identified, that is, up to monomial gauge. Under repeated singular values the SVD
component set is defined only up to rotation within the eigenspace, and only the block is identifiable.

**Proposition 2.3 (invariants) [P].** Functions of $\{R_j\}$ that are monomial-invariant: $m_j$; the projectors
$\hat u_j\hat u_j^\top$ and $\hat v_j\hat v_j^\top$; the rank-one cosine

$$\cos_R(j,l)=\frac{\langle R_j,R_l\rangle_F}{m_jm_l}=(\hat u_j\!\cdot\!\hat u_l)(\hat v_j\!\cdot\!\hat v_l),$$

which is unchanged by a joint sign flip of $(u,v)$ and equals $-1$ for $R_l=-R_j$; and
$\lVert\hat W\rVert_F^2=\sum_{j,l}m_jm_l\cos_R(j,l)$. All follow from
$\langle uv^\top,u'v'^\top\rangle_F=(u\!\cdot\!u')(v\!\cdot\!v')$.

**Proposition 2.4 (no continuous sign rule) [L].** The double cover $S^{n-1}\to\mathbb{RP}^{n-1}$ has no continuous
section for $n\ge2$. Every sign canonicalization is discontinuous somewhere. The `argmax(abs(u))` pivot is discontinuous at
magnitude ties (D5). Function never needs the sign: $P$, $\Pi$ and the scores are sign invariant. Identity should not
use it either.

### 2.2 Canonical form and pruning rule

Reference implementation: `canonicalize_invariant` in `mms07_math_checks.py`.

1. **Scale-safe norms.** $\lVert x\rVert=s\lVert x/s\rVert$ with $s=\max\lvert x\rvert$; carry $\log m_j=\log\lVert u_j\rVert+\log\lVert v_j\rVert$.
   Store unit factors $(\hat u_j,\hat v_j)$ and $\log m_j$; form $m_j\hat v_j$ only when a product is needed.
2. **Merge collinear components.** If $\lvert\hat u_j\!\cdot\!\hat u_l\rvert\ge1-\varepsilon_{\text{merge}}$, replace the
   pair by $\hat u_j(m_j\hat v_j\pm m_l\hat v_l)^\top$; likewise for collinear $v$. This removes duplicates exactly and
   sends cancelling pairs to magnitude zero.
3. **Classify by relative magnitude.** With $\omega_j=m_j/\lVert\hat W\rVert_F$: `exact_zero` (a factor is exactly
   zero), `cancelled` (zero after merging), `unrepresentable` ($\omega_j$ below float32 unit roundoff, so the component
   cannot change a float32 output), `pruned` ($\omega_j<\varepsilon_{\text{prune}}$, a recorded policy choice), `active`.
4. **Order by magnitude,** descending; near-ties form a tie block. Do not order by hash.
5. **Relative tie tolerance** in selection: scores within $\tau\cdot\max_j s_j$ are tied.

**Proposition 2.5 (certified pruning) [P].** Removing a set $D$ changes $\hat W$ by $\sum_{j\in D}R_j$, with
$\lVert\cdot\rVert_F^2=\sum_{j,l\in D}m_jm_l\cos_R(j,l)\le(\sum_{j\in D}m_j)^2$, and exactly $m_j$ for one component.
Pruning the smallest components while $\sum_Dm_j\le\varepsilon\lVert\hat W\rVert_F$ certifies relative reconstruction
error at most $\varepsilon$.

The rule is invariant under reciprocal scaling, signs and permutations, and equivariant under $W\to sW$. The checks give
identical output for $(1,1)$, $(10^{13},10^{-13})$ and $(10^{200},10^{-200})$, and the cancelling pair is classified
`cancelled` [X: p2_activity_gauge_defect, p2_extreme_scale_imbalance, p2_selection_scale_and_cancelling_pair].

A scalar diagnostic for gauge artifacts is the inflation index
$\iota=(\sum_jm_j^2)^{1/2}/\lVert\hat W\rVert_F$: 1 for mutually orthogonal components, large when components cancel.

### 2.3 Three kinds of identity

| Tier | Object | Guarantee | Use |
| --- | --- | --- | --- |
| Artifact identity | SHA-256 of exact bytes | byte equality; no physical claim | custody, replay, manifests |
| Approximate physical identity | matching certificate: permutation $\pi$, per-pair $\cos_R\ge1-\varepsilon$, competitor gap $\max_{l\ne\pi(j)}\cos_R(j,l)\le1-\gamma$, magnitude ratios | equality up to monomial gauge within tolerance, with uniqueness margin | deciding whether two maps describe the same components |
| Validated transport | operator between checkpoints plus a functional validation record: source and target digests, matching certificate, data split digest, validation metric against a **target-estimated comparator at the same budget**, validity horizon | measured, time-limited | reusing component-indexed statistics after refresh |

The rounded hash is neither tier: it is not injective (aliasing) and not continuous (boundary, pivot, $\pm0$). The
check shows rank-one-cosine matching returns $1-2\times10^{-12}$ and $1-10^{-14}$ in the two situations where hashes
differ [X: p2_component_id_instability]. The requirement of a target-estimated comparator comes from the BPD transfer
addendum, where source-assisted transfer failed every registered directional gate against a projector estimated at the
target with the same budget [R].

### 2.4 Outer-product sum versus orthogonal projector

Let $U_S$ have unit rows, Gram matrix $G=U_SU_S^\top$ with eigenpairs $(\lambda_i,w_i)$ and orthonormal directions
$e_i=U_S^\top w_i/\sqrt{\lambda_i}$.

**Proposition 2.6 [P, X: p2_outer_sum_vs_projector].**

1. $\Pi=\sum_ie_ie_i^\top$ and $P=\sum_i\lambda_ie_ie_i^\top$. $P=\Pi$ if and only if the selected rows are orthonormal.
2. $\lVert Ph\rVert\le\lambda_{\max}(G)\lVert h\rVert\le k\lVert h\rVert$, while $\lVert\Pi h\rVert\le\lVert h\rVert$.
3. $\min_h\cos\angle(Ph,\Pi h)=2\sqrt{\lambda_{\min}\lambda_{\max}}/(\lambda_{\min}+\lambda_{\max})$ over the positive
   spectrum (sampled minimum 0.8660254 against the bound 0.8660254 for the 60° pair).
4. $h^\top Pc$ and $h^\top\Pi c$ can have opposite signs in both directions.
5. $P/\lambda$ is the direction of the infinite-ridge limit of $\Pi_\lambda=U_S^\top(G+\lambda I)^{-1}U_S$; $\Pi$ is the
   zero-ridge limit. $\Pi_\lambda$ weights direction $i$ by $\lambda_i/(\lambda_i+\lambda)$.

*Proof.* Items 1, 2 and 5 are the singular value decomposition of $U_S$. Item 3 is
$\cos=\sum\lambda_ia_i^2/\sqrt{\sum\lambda_i^2a_i^2\sum a_i^2}$ with $a_i=e_i\!\cdot\!h$, minimized by the Kantorovich
inequality. Item 4 is the bilinear forms $\sum\lambda_ia_ib_i$ and $\sum a_ib_i$ having different signs for unequal
$\lambda_i$; an explicit instance is in the check. $\square$

Does overlap cause harmful amplification? The MMS hook renormalizes $\delta$ to $\rho\lVert c\rVert$, so the norm
amplification in item 2 is cancelled. What survives is directional bias toward the consensus direction of overlapping
components (18.4° for one duplicate, tending to 45° as duplicates accumulate) and admission errors (item 4). On
synthetic correlated dictionaries with $k=4$ of $d=64$ the two operators disagreed on admission for 3.3% of random
pairs with no shared direction and 28% with a strongly shared one [S].

**Proposition 2.7 (specificity at lower rank) [P, X: p2_outer_sum_vs_projector].** Let $\tilde u_j=\sum_l(G^{-1})_{jl}\hat u_l$
be the dual basis. The oblique projector $O_j=\hat u_j\tilde u_j^\top$ satisfies $O_j\hat u_l=\delta_{jl}\hat u_l$ and
$\Pi=\sum_{j\in S}O_j$. The matched filter $\hat u_j\hat u_j^\top$ reads $\alpha_j+\sum_{l\ne j}G_{jl}\alpha_l$ where
$\alpha=G^{-1}U_Sh$ are the least-squares coefficients. The dual read is cross-talk free at the price of a variance
inflation factor $(G^{-1})_{jj}\le1/\lambda_{\min}(G)$.

*Proof.* $O_j\hat u_l=\hat u_j(G^{-1}G)_{jl}$, and $U_S^\top G^{-1}U_S=\sum_j\hat u_j\tilde u_j^\top$. The read
$\tilde u_j\!\cdot\!h=\alpha_j$ has variance $(G^{-1})_{jj}\sigma^2$ under isotropic noise of variance $\sigma^2$. $\square$

So a Gram-aware intervention can transmit the share of $h$ attributable to $k'<k$ chosen components without leakage, by
computing the dual basis in a context set $S$ and applying $\sum_{j\in S'}O_j$. For the 60° pair the cross-talk is 0.5
and the inflation 4/3. Selection should likewise be Gram-aware (greedy maximization of $\lVert\Pi_Sh\rVert$, as in
orthogonal matching pursuit), since top-k by individual score spends rank on near-duplicates.

### 2.5 Measurable stability criteria

| Criterion | Quantity | Reading |
| --- | --- | --- |
| Spectral separation | $\lVert\Delta W\rVert_{op}/\mathrm{gap}_k$ | Wedin: $\lVert\sin\Theta\rVert\le\lVert\Delta W\rVert/\mathrm{gap}$ for the rank-$k$ singular subspace [L]; a component is individually transportable only when its own gap dominates the drift |
| Component separability | $\min_{l\ne j}(1-\lvert\cos_R(j,l)\rvert)$ | small means merge or treat as a block |
| Gram conditioning | $\kappa(G_S)$, $\max_j(G^{-1})_{jj}$ | bounds the $P$/$\Pi$ angle and the dual-basis noise; use $\Pi_\lambda$ with $\lambda$ at the rank tolerance |
| Projector stability | $\lVert\Delta U_S\rVert_{op}/\sigma_{\min}(U_S)$ | bounds $\lVert\Pi'-\Pi\rVert$ at fixed rank [L] |
| Selection margin | $(s_{(k)}-s_{(k+1)})/s_{(1)}$ | small means the selected set is not determined |
| Inflation | $\iota$ | large means cancelling structure |
| Refresh match | minimum matched $\cos_R$, competitor gap | the receipt that two successive maps are the same components |
| Gradient coverage | $\chi$, $\lVert\Pi_{U_S}q\rVert/\lVert q\rVert$ | upper bound on first-order usefulness of the map |

### 2.6 Adversarial regression cases

| Case | Input | Required behaviour |
| --- | --- | --- |
| A1 | $(1,1)$ versus $(10^{\pm13},10^{\mp13})$ | identical activity, magnitude and projector |
| A2 | $(10^{200},10^{-200})$ factors of the all-ones matrix | finite output, active, $\log m=\log2$ |
| A3 | $u$ entry $\pm10^{-6}$ | same physical identity (match cosine above $1-10^{-10}$) |
| A4 | $u=(0.6,-0.6000001,0.1)$ versus $(0.6000001,-0.6,0.1)$ | same physical identity; same $P$ and $\Pi$ |
| A5 | exact duplicate component | merged; rank and selection unchanged relative to the merged map |
| A6 | cancelling pair with magnitude $10^3$ | classified `cancelled`; never selected |
| A7 | inputs scaled by $10^{-8}$ | identical selection |
| A8 | two components at 60°, $h=e_1+e_2$, $c=E(1,-0.5)$ and its negative | $\Pi$-admission follows the sign of $h^\top\Pi c$ |
| A9 | repeated singular value, rotated eigenbasis | block-level projector identical; per-component identity reported as undetermined |
| A10 | generic $GL(C)$ mixing | full-span $\Pi$ invariant; component-level claims refused without a matching certificate |
| A11 | near-collinear selected rows, $\lambda_{\min}(G)=10^{-10}$ | truncated or ridge inverse; receipt records rank, tolerance and $\kappa$ |
| A12 | map with $\chi<0.05$ and reconstruction cosine above 0.99 | flagged: high reconstruction, negligible gradient coverage |

---

## Problem 3: mechanistic memory as a dynamical system

### 3.1 State and dynamics

$M_t=(\mathcal B_t,\Phi_t,\mathcal O_t)$: bank entries (key, payload in physical activation coordinates or coded
coordinates with a basis version, write step, lineage); maps (basis version, factors, source step, digests); outcomes
(action, context, paired difference, propensity). The coupled system is

$$\theta_{t+1}=\theta_t-\eta_t\,U_{\text{opt}}\big(g_t+A_t^\top\delta_t(M_t,\text{batch}_t);\ \text{state}_t\big),\qquad
M_{t+1}=\mathrm{Update}(M_t,\theta_t,g_t,h_t,\text{evidence}_t),$$

with the controller choosing among STOP, ORDINARY, OBSERVE, MECH$(a)$, AUDIT$(a)$ and REFRESH.

Two structural facts. First, a bank keyed by example ID returns information that is on average one epoch old: a key
recurs after about $N/B$ steps. Second, the feedback from $M$ into $\theta$ is a delayed-gradient loop.

**Proposition 3.2 (delay-independent stability, linear model) [P, X: p3_delay_independent_stability].** For
$\theta_{t+1}=(1-a)\theta_t-\rho a\,\theta_{t-\tau}$ (gradient descent on a quadratic mode with $a=\eta\lambda$ and an
unnormalized replay of age $\tau$), all characteristic roots lie inside the unit disk for every $\tau$ whenever
$\lvert1-a\rvert+\rho a<1$, that is, $\rho<1$ and $a<2/(1+\rho)$.

*Proof.* A root with $\lvert z\rvert\ge1$ of $z^{\tau+1}-(1-a)z^\tau+\rho a=0$ would satisfy
$\lvert z\rvert\le\lvert1-a\rvert+\rho a/\lvert z\rvert^\tau<1$. $\square$

With $\rho=0.2$ the delay-independent stable range shrinks from $a<2$ to $a<1.667$; beyond it stability depends on the
delay (CE7). Norm-bounded, gated replay satisfies the same majorant. Under Adam the effective $a$ is normalized and the
margin is not binding at the registered learning rates.

### 3.2 Staleness of cached cotangents

Model: $h_{k+1}=F(h_k;\theta)$, $k<K$, loss $\ell(h_K)$, cotangent $p_k=\partial\ell/\partial h_k$. Assume on a region
containing both trajectories: $\lVert\partial_hF\rVert\le\lambda$, $\lVert\partial_\theta F\rVert\le a$, $\partial_hF$
Lipschitz with constants $\beta_h$ in $h$ and $\beta_\theta$ in $\theta$, $\lVert\nabla\ell\rVert\le\gamma_1$,
$\nabla\ell$ $\gamma_2$-Lipschitz. Let $S_k=\sum_{i<k}\lambda^i$.

**Theorem 3.1 [P].** For the same input and parameters $\theta,\theta'$,

$$\lVert p_k'-p_k\rVert\le L_k\lVert\theta'-\theta\rVert,\qquad
L_k=\gamma_1\lambda^{K-k-1}\sum_{j=k}^{K-1}\big(\beta_haS_j+\beta_\theta\big)+\gamma_2a\lambda^{K-k}S_K .$$

*Proof.* $\lVert h_k'-h_k\rVert\le aS_k\lVert\Delta\theta\rVert$ by induction. Hence
$\lVert J_j'-J_j\rVert\le(\beta_haS_j+\beta_\theta)\lVert\Delta\theta\rVert$. Telescoping
$p_k=J_k^\top\cdots J_{K-1}^\top\nabla\ell(h_K)$ replaces one factor at a time, each term bounded by
$\gamma_1\lambda^{K-k-1}\lVert J_j'-J_j\rVert$, plus $\lambda^{K-k}\gamma_2\lVert h_K'-h_K\rVert$. $\square$

Reading, with $n=K-k$ the remaining depth:

- $\lambda<1$: $L_k\le\gamma_1n\lambda^{n-1}b_\infty+\gamma_2a\lambda^n/(1-\lambda)$,
  $b_\infty=\beta_ha/(1-\lambda)+\beta_\theta$. The absolute error is uniformly bounded in depth. Because
  $\lVert p_k\rVert$ itself scales like $\lambda^n$, the **relative** staleness grows about linearly in the remaining
  depth.
- $\lambda=1$: $L_k$ grows quadratically in depth. $\lambda>1$: exponentially.
- The two terms separate **readout staleness** (the downstream map changed) from **state staleness** (the activation at
  which the cotangent was taken moved). Both are driven by the same $\Delta\theta$ in a tied network.

On random tied tanh networks with $K=8$ the relative cotangent change per unit parameter perturbation decreases
monotonically from the first visit to the last at every initialization scale tried (3.52 at visit 0 to 1.09 at visit 7
for scale 0.9) [S, X: p3_staleness_depth_scaling].

Parameter drift under AdamW: $\lVert\theta_t-\theta_s\rVert_\infty\le c_A\sum_{i=s}^{t-1}\eta_i$ plus the decay term,
where $c_A$ depends only on $(\beta_1,\beta_2)$; the worst case is
$(1-\beta_1)/(\sqrt{1-\beta_2}\,(1-\beta_1/\sqrt{\beta_2}))\approx32$ for the defaults and values near 1 are typical.
(The bound follows from $\lvert m_t\rvert\le(1-\beta_1)\sum_i\beta_1^{t-i}\lvert g_i\rvert$ and
$v_t\ge(1-\beta_2)\beta_2^{t-i}g_i^2$; bias correction only lowers the ratio.) An a priori age budget is therefore $\tau_{\max}(k)\approx\varepsilon\lVert p_k\rVert/(L_kc_A\eta)$,
shorter for sites far from the loss. In practice the budget is measured: $r(\text{age})$, the correlation between cached
and current cotangent on hits, is free. Two regimes are predicted: ballistic drift early ($1-r\propto\text{age}^2$) and
diffusive drift late ($1-r\propto\text{age}$) [C].

**The intervention-error chain.** With a stale map of age $\tau_{\text{map}}$ and a cached cotangent of age
$\tau_{\text{bank}}$,

$$\frac{\lVert\hat\delta-\delta^*\rVert}{\lVert\delta^*\rVert}\ \lesssim\
\underbrace{\frac{\lVert W_t-W_{t-\tau_{\text{map}}}\rVert_{op}}{\mathrm{gap}_S}}_{\text{map error (Wedin)}}
+\underbrace{\frac{L_k\lVert\theta_t-\theta_{t-\tau_{\text{bank}}}\rVert}{\lVert p_k\rVert}}_{\text{cotangent staleness}},\qquad
\lVert\hat d-d^*\rVert\le\lVert A_m\rVert\,\lVert\hat\delta-\delta^*\rVert,$$

and by Proposition 1.2 the first-order loss effect of the error is at most $\eta\lVert Pg\rVert\lVert\hat d-d^*\rVert$.
For hook propagation $\lVert A_m\rVert\lesssim\max_j\lVert x_j\rVert\,\Lambda/(1-\lambda)$; for local injection it is
$\lVert X_m\rVert$. Map drift enters divided by a spectral gap, parameter drift multiplied by remaining depth, and the
result is multiplied by the upstream gain.

### 3.3 Physical coordinates, transport and invalidation

1. **A cotangent stored in physical activation coordinates needs no transport when the basis changes.** The neuron
   basis is fixed; its validity is governed by Theorem 3.1, not by the component map. This is the design already chosen
   in `MechanisticMemory` ("store detached physical cotangents") and it is correct.
2. **Coded entries are decoded with the basis they were encoded in.** Keep basis versions alive by reference count
   ($C\cdot n$ numbers each). Decoding $\hat c=U_s^\top\alpha$ is a valid approximation of the old physical cotangent
   regardless of the current basis.
3. **Transport is required only for statistics indexed by component:** coefficient histories, per-component gains, the
   SNR sketch of D11. The transport is $T_{s\to t}=U_t(U_s)^\dagger$ on the overlap, with information loss bounded by
   $\lVert\sin\Theta(\mathrm{span}_s,\mathrm{span}_t)\rVert$, admitted only with a tier-3 record (§2.3).
4. **Invalidate** on: lineage mismatch; site shape or hook position change; hash or index scheme mismatch; key no longer
   denoting the same example; and statistically, when the lower confidence bound of $r(\text{age})$ in an age bin falls
   below $r_{\min}$ (set `max_age` to the largest surviving bin). Do not invalidate physical entries because the basis
   digest changed.

The current scheduler rejects repairs whose `basis_digest` differs unless a boolean bypass is set, and the digest changes
at every refresh (D5); once D10 is repaired this would discard all history every 64 steps.

### 3.4 Refresh criterion

Let a fresh map give incremental benefit $b_0$ per step, decaying with map age $a$ as $b(a)=b_0(1-a/\tau_m)$; let
$\bar g_0$ be the baseline progress per step, $c$ the step cost and $C_r$ the refresh cost.

**Theorem 3.3 [P, X: p3_refresh_square_root_law].** The refresh period maximizing value per compute,
$\varrho(R)=[R\bar g_0+\int_0^Rb]/(Rc+C_r)$, is

$$R^*=\frac{-C_r+\sqrt{C_r^2+2cC_r\tau_m(1+\bar g_0/b_0)}}{c}\ \approx\ \sqrt{2\,\tau_m\frac{C_r}{c}\Big(1+\frac{\bar g_0}{b_0}\Big)},$$

valid when $R^*<\tau_m$. In general $R^*$ satisfies $\bar g_0+b(R^*)=c\,\varrho(R^*)$: refresh when the instantaneous
value rate falls to the long-run average rate.

*Proof.* Set $\varrho'(R)=0$; the numerator reduces to $cR^2+2C_rR-2C_r\tau_m(1+\bar g_0/b_0)=0$. $\square$ This is the
Young/Daly checkpoint-interval form and the marginal value theorem [L]. The closed form agrees with a grid optimum to
0.2% in seven configurations.

Consequences:

- $R^*$ grows as $\sqrt{C_r}$, so a sparse refit costing about 50 steps warrants a period roughly ten times that of an
  SVD costing half a step.
- When replay is marginal ($b_0\ll\bar g_0$), $R^*\to\infty$: do not refresh. When $b_0\le0$ refresh has no value.
- If $R^*\ge\tau_m$ the map decays before it pays for itself and the action should not be used.

Illustration with MMS-06 costs [R] and assumed $\tau_m$: SVD ($C_r/c\approx0.45$) gives $R^*\approx9$, 21, 42 for
$\tau_m=100$, 500, 2000; the sparse refit ($C_r/c\approx50$) gives 179 and 400 for $\tau_m=500$ and 2000. The fixed
period of 64 used in MMS-06 is far too short for the sparse map at any plausible $\tau_m$, in line with the recorded
1.46 s of refresh cost in a 6.47 s run [R].

Adaptive rule: at each refresh measure the quality lost since the last one, $\Delta=1-q$, from the matching certificate
or the direction cosine of $\delta$ on a probe batch; set $\hat\tau_m=R_{\text{last}}/\Delta$; schedule the next refresh
at $R^*(\hat\tau_m)$, clipped; trigger early if the free proxy
$\lVert W_t-W_{\text{map}}\rVert_F/\mathrm{gap}_S$ crosses a threshold.

---

## Problem 4: causal attribution and feedback economics

### 4.1 Three quantities

- **Direct feedback** $Y_w=V(\theta_{t+W})-V(\theta_t)$ under the chosen action. Descriptive.
- **Delayed credit:** the effect of the action in window $w$ on outcomes in windows $w+k$ through $\theta$, the
  optimizer moments ($\beta_2=0.999$ remembers about 1000 steps) and the bank.
- **Causal improvement** from a state $s$: $\Delta_a(s;W)=\mathbb E[V(\theta^a_{t+W})-V(\theta^{\text{ord}}_{t+W})\mid s]$
  at equal steps, and $\Delta^C_a$ at equal compute.

$Y_a=\Delta_a+Y_{\text{ord}}(s)$. The scheduler compares $Y_a$ from early states with $Y_{\text{ord}}$ from late states,
or with zero (D2). On a toy problem the sequential raw-progress estimate is 10.198 for a null treatment and 10.201 for
a real one: it does not see the treatment at all [S, X: p4_estimator_comparison].

### 4.2 Estimators

| Estimator | Identifies | Cost | Variance | Limits |
| --- | --- | --- | --- | --- |
| Sequential raw progress (current) | nothing causal | 0 | n/a | confounded by learning curve, checkpoint age, curriculum, selection |
| Randomized windows, unstratified | marginal effect under the logged state distribution | 0 beyond logging propensities | curve-dominated (toy SE 0.45 against an effect of 0.06) | useless at feasible sample sizes |
| Randomized windows, stratified by window index or regression-adjusted for trend | same | 0 | toy SE 0.0035 | needs many runs sharing a time axis; includes carryover; cannot be used inside one run |
| **Paired clones with common random numbers** | $\Delta_a(s;W)$ from the same state | one extra window and two evaluations per audit | toy SE 0.0027; exactly 0 under the null | measures a $W$-step effect; branch bookkeeping |
| One-step lookahead on a probe batch | one-step causal change in held-out loss | two forward passes | low | blind to delayed effects |
| Two-phase surrogate | $\mathbb E\Delta_a$ | cheap surrogate every window, clone audit with probability $\pi$ | $\mathrm{Var}(D-\beta s)/\lvert\mathcal A\rvert+\beta^2\mathrm{Var}(s)/n$ | unbiased for any surrogate quality because audits are sampled at random |
| Off-policy (IPW, doubly robust) | value of an alternative controller from logs | 0 | high | requires logged propensities bounded away from 0; the current scheduler is deterministic, so positivity fails |

Paired clone audit: snapshot $(\theta,\text{optimizer state},\text{RNG},\text{bank journal mark})$; run branch $a$ for
$W$ steps and branch ordinary for $W$ steps (per-update estimand) or $\lceil\kappa_aW\rceil$ steps (equal-compute
estimand) on the same minibatch sequence; evaluate both on the feedback split; record $D=V(\theta^a)-V(\theta^{\text{ord}})$
with the propensity of the audit. The branch that continues is chosen by the registered policy.

Two-phase estimator: $\hat\Delta=\overline{\beta s}_{\text{all}}+\overline{(D-\beta s)}_{\mathcal A}$ with $\beta$ fitted
on earlier audits; $s$ is the accumulated certificate $\sum_t\eta_t\langle g_{\text{ref}},\Delta u_t\rangle$ of §1.6.

With paired differences the incremental value of ordinary is zero by definition. That legitimizes the scheduler's
default for ordinary, provided the feedback it is given is $D$ rather than $Y$.

### 4.3 Opportunity cost

**Proposition 4.2 (break-even) [P].** Let ordinary progress per step be $\bar g_0$ at cost $c_0$, and let action $a$
yield $\bar g_0+\Delta_a/W$ per step at cost $\kappa_ac_0$, where $\kappa_a$ includes amortized refresh, audits,
verifier, lookahead and selection. Under a fixed remaining budget and locally linear value in steps:

- if $\bar g_0>0$, $a$ is preferable to ordinary iff $\Delta_a/W>(\kappa_a-1)\bar g_0$;
- if $\bar g_0\le0$, ordinary is dominated by STOP, and $a$ is preferable to STOP iff $\bar g_0+\Delta_a/W>0$.

*Proof.* Compare value rates $(\bar g_0+\Delta_a/W)/(\kappa_ac_0)$ and $\bar g_0/c_0$. $\square$

Mechanistic information is cheapest to justify at a plateau, where ordinary compute no longer converts to progress, and
most expensive during fast early learning, which is exactly when the scheduler credits it. With measured $\kappa$ of
1.38 to 2.06 at 1.5 s and 1.42 to 1.96 at fixed exposure [R], replay must raise per-update progress by 38% to 106%.

Audits are worth their cost by a value-of-information test. With belief $\Delta_a\sim\mathcal N(\mu,\sigma^2)$, audit
noise $\sigma_e$, threshold $\vartheta=(\kappa_a-1)W\bar g_0$ and $N_{\text{rem}}$ remaining windows, the knowledge
gradient is $N_{\text{rem}}\tilde\sigma f(-\lvert\mu-\vartheta\rvert/\tilde\sigma)$,
$\tilde\sigma=\sigma^2/\sqrt{\sigma^2+\sigma_e^2}$, $f(z)=z\Phi(z)+\phi(z)$ (Frazier, Powell and Dayanik [L]). Audit iff
this exceeds the audit cost converted at the ordinary value rate. Exploration stops by itself when no audit can change
the decision.

Sequential stopping uses anytime-valid bounds. For $\sigma$-sub-Gaussian paired differences the two-sided normal-mixture
boundary $\lvert\sum_{i\le n}(D_i-\mu)\rvert<\sigma\sqrt{(n+\varrho)\log((n+\varrho)/(\varrho\alpha^2))}$ holds for all
$n$ simultaneously with probability at least $1-\alpha$ (Robbins; Howard et al. [L]). Monte Carlo at $\alpha=0.05$,
$\varrho=20$, horizon 2000 gave a crossing rate of 0.032; the radius of the mean is $1.66\sigma$ at $n=8$ and
$0.39\sigma$ at $n=64$ [X: p4_confidence_sequence_coverage]. Adopt $a$ when the lower bound exceeds $\vartheta$;
declare $a$ futile when the upper bound falls below it.

### 4.4 Controller specification

Actions: `STOP`, `ORDINARY` (no bank writes, no audit, zero overhead), `OBSERVE` (bank writes only), `MECH(a)`,
`AUDIT(a)`, `REFRESH(map)`.

1. **Feasibility.** An action is feasible only if an upper bound on its full-window cost fits the remaining budget. An
   unmeasured cost is infeasible until a calibration micro-run, charged to the budget, measures it. If no ordinary
   window plus the mandatory final evaluation fits, the action is `STOP`.
2. **Stop.** If the monitor shows no ordinary progress by the registered patience rule and no action has an upper
   confidence bound above its threshold, `STOP`.
3. **Exploit.** Choose the action with the largest posterior mean net rate among ordinary and actions whose paired
   estimate exceeds $\vartheta_a$.
4. **Audit.** With probability set by the knowledge gradient, audit the most decision-relevant action. Log the
   propensity.
5. **Accounting.** Charge audits, refreshes, lookahead and selection to $\kappa_a$. Never divide a negative gain by a
   cost; compare net rates.
6. **Controls.** Label-permuted feedback (same marginal reward distribution) and a yoked schedule replayed from another
   seed (same action marginals, no dependence on this run's feedback).

---

## Problem 5: memory and compute scaling

### 5.1 Hit-rate laws

Keys are sampled uniformly with replacement from $N$, $B$ per step, reads before writes.

**Proposition 5.1 (transient, collision-free) [P, X: p5_hit_rate_laws].**
$\mathbb E[\text{hits at step }s]=B\,(1-(1-1/N)^{Bs})$.

*Proof.* A queried key hits iff it appeared in one of the $s$ earlier batches. $\square$

Retrodiction on committed receipts: for LBF-20D ($N=12{,}288$, $B=128$, 240 steps) the expectation is 19,382 with
Monte Carlo SD 27. The receipts record 19,376 (LT MMS-01 to 05) and 19,402 (bank profile). The receipts are consistent
with a collision-free bank, as the audit's injectivity observation implies.

**Proposition 5.2 (steady state, fixed random hash) [P, X: p5_hit_rate_laws].** With capacity $\mathrm{Cap}$ and
$\lambda=N/\mathrm{Cap}$, a key shares its slot with $J\sim\mathrm{Poisson}(\lambda)$ others and holds it with
probability $1/(1+J)$, so

$$H_\infty=\frac{1-e^{-\lambda}}{\lambda}\ \le\ \min(1,\mathrm{Cap}/N).$$

With a maximum age $A$, $H=\mathbb E_J[(1-e^{-(J+1)AB/N})/(J+1)]$. Simulation with a mixing hash agrees within 0.01 at
$\mathrm{Cap}/N\in\{0.25,0.5,1,2,4\}$. Capacity needed for a target hit rate: $0.63N$ for 0.5, $2.15N$ for 0.8,
$4.66N$ for 0.9, $9.66N$ for 0.95. A direct-indexed table (slot equals key) gives $\min(1,\mathrm{Cap}/N)$: full
coverage at $\mathrm{Cap}=N$. Keys here are dense example indices known in advance, so direct indexing is available
whenever $N\cdot\text{row}$ fits, and a mixing hash (splitmix64 finalizer) otherwise. The pinned multiplicative hash is
neither (D1): on RMP keys the mixing hash occupies 19,282 slots at 4 GiB where the pinned hash occupies 2,892.

**Scaling law.** To hold hit rate $H$:

$$M(H)=\phi(H)\cdot N_{\text{keys}}\cdot(2Td+16)\ \text{bytes},\qquad\phi=\min(1,H)\ \text{direct},\quad\phi=1/\lambda(H)\ \text{hashed},$$

where $N_{\text{keys}}$ is examples times clock or supervision-stage multiplicity times instrumented site-visits. RAM
grows linearly in $N$, $T$ and $d$. Rank-$r$ coded rows replace $d$ by $r$ ($2r$ for the control-variate table).

| Workload | Row | Rows at 4 GiB | RAM for full direct coverage |
| --- | --- | --- | --- |
| LBF-20D, $T{=}1$, $d{=}64$ | 144 B | 29,826,161 | 1.7 MiB for 12,288 keys |
| RMP pointer_chase, $T{=}64$, $d{=}128$ | 16,400 B | 261,888 | 313 MiB per 20,000 keys |
| Mathlib TRM-XS, $T{=}384$, $d{=}176$ | 135,184 B | 31,771 | 1.03 GiB for a 2,048-row pool at 4 stages; 97 GiB for 193k rows at 4 stages |

Hit rate is not value. A hit of age $g$ carries correlation $r(g)$ with the current cotangent (a variance reduction of
$r^2(g)$ for the control-variate table), and the typical age is one epoch, $N_{\text{keys}}/B$ steps.
**Usefulness requires the cotangent decorrelation age $\tau_c$ to exceed the epoch length.** Enlarging the corpus at
fixed batch size lengthens the epoch and ages every hit, whatever the RAM. This is the binding scaling constraint, ahead
of RAM capacity and ahead of bandwidth.

### 5.2 Cost model

$$C_{\text{step}}=c_0(B,T,d,K)+\tau_{\text{fix}}+\beta_{\text{bank}}BTd+\beta_{\text{proj}}B_{\text{hit}}T\,\frac{d_{in}C+dC+2dk}{C_0}+\beta_{\text{xfer}}BTd+\frac{C_{\text{refresh}}}{R}+\frac{C_{\text{verify}}}{W}+\frac{C_{\text{sched}}}{W}.$$

- $\tau_{\text{fix}}$: per-call hook, dispatch and conversion overhead. On the MMS-06 host the whole extra step cost at
  LBF-20D shape was 3.48 ms (dense), 4.66 ms (SVD frozen) and 5.31 ms (SVD refreshed) at fixed exposure [R].
- $\beta_{\text{bank}}\approx3.7$ ns and $\beta_{\text{proj}}\approx5.4$ ns per element at $C_0=(C{=}16,k{=}4)$, fitted
  on four shapes in the sandbox that produced the receipt [X: p5_sandbox_microbenchmark; informational, not the MMS-06
  host]. At LBF-20D shape ($BTd=8192$) the data-proportional terms total about 0.075 ms, roughly 2% of the measured
  extra cost. **The MMS-06 overhead is almost entirely fixed cost.**
- $c_0\propto K\,B\,T\,d^2$ for an MLP-dominated block. The overhead ratio is
  $[\tau_{\text{fix}}+\beta BTd]/c_0$: the fixed part falls as $1/(KBTd^2)$ and the data part as $1/(Kd)$ at fixed $C$.
  If $C$ grows with $d$ the projection ratio tends to a constant.
- Refresh: full SVD is $O(d^3)$; from the quantized 4.7 to 9.4 ms at $d=64$ (D13), expect 40 to 75 ms at $d=128$ and
  100 to 200 ms at $d=176$. A rank-16 randomized SVD is $O(d^2r)$.
- Memory traffic per step is about $12\,BTd$ bytes (BF16 payload plus float32 temporaries, read and write): 6 MB at RMP
  shape, 13 MB at TRM-XS shape. Bandwidth is not binding below $BTd\sim10^8$. Parallel arms share bandwidth, not
  compute; at these sizes sharding by key across workers is lock-free and unconstrained.

### 5.3 Asynchronous CPU-resident maps for GPU training

Lookups depend only on batch keys, which are known before the forward pass.

1. **Prefetch.** While the GPU runs the forward pass, the CPU reads the bank rows for the batch keys and uploads them
   to pinned memory with a non-blocking copy ($2BTd$ bytes in BF16: 1.05 MB at RMP shape, 2.16 MB at TRM-XS).
2. **Decide on the GPU.** Projection, pushforward certificate and scaling run in the backward hook as GPU matmuls. No
   synchronous CPU round trip occurs inside backward.
3. **Deferred write.** Cotangents are copied to the host after the step and written $L$ steps later. A key recurs
   within $L$ steps with probability about $LB/N$, so the write latency is harmless.
4. **Maps.** A weight snapshot is downloaded every $R$ steps, factorized on the CPU, and the factors uploaded
   ($C\cdot d$ numbers). Map age is at most $R+L_{\text{map}}$.
5. **Identities.** Every artifact carries (run lineage UUID, step, basis version, hash or index scheme); full weight
   digests at checkpoints only. A mismatch invalidates (§3.3).
6. **Staleness bound.** Bank entry age is at least one epoch plus $L$; map age at most $R+L_{\text{map}}$; both enter
   the chain of §3.2.
7. **VRAM alternative.** If $N\cdot\text{row}$ fits in VRAM, keep the bank on the GPU: about 160k RMP rows fit in the
   2,500 MB RMP VRAM cap.

Transfers are a fraction of a millisecond at these sizes. The costs that matter are synchronization and RAM for
coverage.

### 5.4 Falsifiable predictions

| ID | Prediction | Falsified if |
| --- | --- | --- |
| F5.1 | Uniform sampling, direct or injective bank: cumulative hits follow Proposition 5.1 | observed hits outside ±3 Monte Carlo SD |
| F5.2 | Mixing-hash bank: steady-state hit rate $(1-e^{-\lambda})/\lambda$; stratified version under curriculum sampling | deviation above 0.03 |
| F5.3 | RMP pointer_chase, default shape, pinned hash, contiguous IDs: `occupied_rows` at 512, 2,048 and 4,096 MiB are equal, at most 698 for 2,048 IDs and 2,892 for 20,000 | occupancy grows with the tier |
| F5.4 | Extra CPU per update at RMP shape, one family batch, MMS-06 host: dense 5 to 8 ms, SVD replay 7 to 13 ms; the data-proportional share is at least 30% | extra cost within 15% of the LBF-20D figure (overhead still constant) or above three times the prediction |
| F5.5 | Overhead ratio below 5% requires $c_0$ of roughly 100 to 260 ms per update at RMP shape | ratio below 5% at smaller $c_0$ without code changes |
| F5.6 | Mathlib TRM-XS, 2,048-row pool at 4 stages: direct coverage needs 1.03 GiB per site-visit; hashed 90% needs 4.8 GiB; data-proportional extra about 10 ms at $B=16$ | hit rates above the law at smaller RAM |
| F5.7 | Cotangent correlation $r(\text{age})$ decays faster at earlier visits: $\tau_c(k)$ decreases with $K-k$ on RMP ($K=8$; observe visits 1, 4, 7) | no ordering or the reverse |
| F5.8 | No replay arm shows a specific benefit when the median hit age exceeds $\tau_c$ (defined by $r=0.5$) | a sham-controlled benefit with stale hits |

F5.4 to F5.6 rest on sandbox constants and carry that uncertainty; the functional form (fixed plus linear in $BTd$) is
the firmer claim.

---

## Problem 6: hybrid exact and mechanistic memory

### 6.1 Separation

A memory-augmented procedure $\mathcal A(x;M)$ is **output-invariant** if $\mathcal A(x;M)=\mathcal A(x;\varnothing)$
for every valid memory state $M$.

**Theorem 6.1 (non-interference) [P].** Let $f$ be deterministic and let the cache hold only pairs $(\kappa(x),f(x))$
inserted after exact computation or verification, with keys that are complete for $f$ (every input that affects the
answer is in the key, including solver version and domain state) and exact tag comparison on read. Then for every
admission, retention and eviction policy, including one driven by an arbitrary predictor, the procedure is
output-invariant. Policy affects cost only.

*Proof.* A read returns a stored value only on an exact key match, and a stored value equals $f$ at that key. Policy
decides which keys are present, which changes only whether $f$ is recomputed. $\square$

**Proposition 6.2 [P].** For exact memory the value functional is $-\mathrm{Cost}(M)$ with outputs fixed; adding a
valid entry never changes an output, and its worst case is the lookup overhead. For approximate guidance the output
depends on $M$, value is not monotone in $M$, and errors propagate through the dynamics of §3. The first is an online
caching problem with measurable item values; the second is a causal estimand (§4).

A middle class is **certified approximate** memory: guidance filtered by a guarantee in expectation (the
control-variate table) or by a per-step certificate (pushforward admission).

Evidence in hand agrees with the separation. LBF-20M exact caching reproduces all answers and reduces query wall time
by 16.8% to 23.2% at 12 KiB against the truly disabled baseline [R]; the end-to-end pipeline effect is not yet measured
(audit). Neural replay has not passed promotion. Compactification-Theory NOR-50 shows the same structure in another
domain: memoized exact execution ran at 0.51 times the direct cost for sources with at least two reuses and at 1.00
times for at most one, with all outputs matching [R]. Exact memory pays in proportion to reuse and never changes an
output.

### 6.2 A common calculus

Item $i$: class, size $s_i$, validity predicate, reuse rate $\lambda_i(t)$, value per use. One currency, seconds of
ordinary compute:

- exact: $\varrho_i=\lambda_i\,\Delta c_i$ (reuse rate times recompute seconds), certificate: content-addressed key,
  solver version, optional witness; valid forever;
- approximate: $\varrho_i=\lambda_i\,\mathbb E[\Delta_i\mid\text{age}]/v_0$, where $\Delta_i$ is the causal gain per use
  and $v_0$ the ordinary value per second; certificate: lineage, step, basis version, scheme, age within the measured
  $r(\text{age})$ horizon.

Retention under a byte budget keeps items in decreasing density $\varrho_i/s_i$ (GreedyDual-Size-Frequency for the
online case [L]); at the optimum the marginal value per byte is equal across classes. As $v_0\to0$ at a plateau the
compute-equivalent value of approximate items with $\Delta>0$ diverges, in agreement with Proposition 4.2.

**Proposition 6.3 (advice with estimated values) [P].** With uniform sizes, true densities $w_i$ and estimates
$\hat w_i$, keeping the top $k$ by $\hat w$ loses at most $2k\max_i\lvert\hat w_i-w_i\rvert$ against the top $k$ by $w$.

*Proof.* $\sum_{S^*}w-\sum_{\hat S}w=\sum_{S^*}(w-\hat w)+(\sum_{S^*}\hat w-\sum_{\hat S}\hat w)+\sum_{\hat S}(\hat w-w)$,
and the middle term is nonpositive. $\square$

So a mechanistic predictor of reuse and recompute cost can steer exact retention with bounded loss and, by Theorem 6.1,
no soundness risk. Combining predictions with a marker or LRU fallback retains a worst-case competitive guarantee
(Lykouris and Vassilvitskii, 2018 [L]). Candidate features: state signatures at the time the subproblem is generated,
persistence of the example's cotangent norm in the bank, component activations of the factor map.

Both memories saturate early on the current gyms: the LBF-20M working set is 11,980 bytes and the LBF-20D bank shows
identical hits at 4 and 16 MiB [R]. Retention decisions are not binding at this scale, so no hybrid can show value on
them.

### 6.3 Proposed gym: LBF-20H

- **Task.** A stream of roots composed from a library of $L$ component templates with Zipf reuse, so separator
  subproblems recur across roots; recompute costs are heavy-tailed; the working set is 4 to 16 times the RAM budget.
- **Common budget.** CPU-seconds and RAM bytes, both charged for every arm, including map fitting and predictor
  training.
- **Arms.** Ordinary search (no cache); exact cache with LRU; exact cache with GreedyDual-Size-Frequency; ordinary
  training of a neural branching heuristic with no cache; hybrid (exact cache with learned retention advice, labels for
  the heuristic drawn from cached exact results); hybrid with permuted advice (sham).
- **Outcome.** Verified solved roots, or exact query cost, on fresh final roots per unit budget. Every answer is
  verified exactly, so the outcome is sound whatever the neural parts do.
- **Why the hybrid might win.** Exact memory raises labels per CPU-second for training; the trained model lowers nodes
  per root and tells the cache what to keep. Each could fail: with uniform reuse and uniform costs LRU is near optimal
  and advice is useless; with no recurrence the cache is useless.
- **Falsification.** The hybrid's margin over GreedyDual-Size-Frequency must exceed the permuted-advice sham under the
  same test as in the registration; otherwise the neural component adds nothing.

---

## Proposed algorithms and complexity

| Algorithm | Purpose | Per-step time | Memory | Guarantee |
| --- | --- | --- | --- | --- |
| A1 Pushforward admission with local injection (OSA) | replace local alignment | $O(BTn\,n_{in})$ | previous update direction, $O(n\,n_{in})$ | exact first-order alignment with $\tilde u$ per sample (Prop. 1.2) |
| A2 Exact AdamW $\Delta u$ logging | marginal improvement estimate | $O(BTn\,n_{in}+n\,n_{in})$ | none | exact one-step update change |
| A3 Prefix JVP certificate | audit hook-style propagation | $\approx1.5f_mF/r_J$ amortized | dual tensors on the prefix | exact $q$ |
| A4 Control-variate table | unbiased variance reduction | $O(BTn\,n_{in})$ | $N\cdot T(n+n_{in})$ BF16 or $2NTr$ coded | unbiased; variance ratio $1-r^2$ (Thm 1.9) |
| A5 Invariant canonicalization | gauge-safe maps | $O(C(d_{in}+d_{out})+C^2(d_{in}+d_{out}))$ per refresh | $C(d_{in}+d_{out}+1)$ | monomial invariance, scale equivariance, certified pruning (Prop. 2.5) |
| A6 Gram-aware projection and selection | $\Pi$, $\Pi_\lambda$, dual basis, greedy selection | $O(BT(dk+k^2)+k^3)$ per hit batch | $k^2$ | best approximation in the span; cross-talk free reads (Props. 2.6, 2.7) |
| A7 Matching certificate | physical identity across refresh | $O(C^2(d_{in}+d_{out}))$ per refresh | $C^2$ | explicit tolerance and gap |
| A8 Direct-indexed or mixed-hash bank | predictable occupancy | unchanged | unchanged | $\min(1,\mathrm{Cap}/N)$ or $(1-e^{-\lambda})/\lambda$ (Prop. 5.2) |
| A9 Adaptive refresh | value-rate optimal period | $O(1)$ plus the match at refresh | none | optimal under the linear decay model (Thm 3.3) |
| A10 Paired-clone audit controller | causal feedback, STOP | $\pi(1+2c_{\text{eval}}/(Wc))$ relative overhead | one state snapshot and a bank journal of $WB$ rows | unbiased $\Delta_a(s;W)$; anytime-valid stopping |
| A11 Learned retention advice for exact caches | hybrid memory | predictor inference per insertion | predictor | output invariance (Thm 6.1); bounded loss (Prop. 6.3) |

---

## Prioritized implementation specification for Neon

Each item lists the change, the acceptance test and the bound it must meet. P0 items block MMS-07 execution.

**P0-1 Bank addressing** (`fast_gradient_bank.py`; LT `mms.py` `GradientBank`).
Add `index_scheme="direct"` (slot equals key minus base, refused if the key range exceeds capacity) and
`hash_scheme="splitmix64_v2"`. Keep both legacy schemes loadable. Add a preflight that takes the planned key set and
refuses to run if distinct slots fall below 0.9 of the random-model expectation; write the occupancy forecast into the
receipt. Tests: RMP keys $8i+4$ at 512 and 4,096 MiB (pinned scheme: 2,892 and 2,892 of 20,000; new schemes: 20,000 and
at least 19,000); LBF keys at capacity 53,237 (203 under the pinned scheme). Cost unchanged.

**P0-2 Factor maps** (`factor_maps.py`; LT `mms.py` `MechanisticMemory._refresh`).
Implement §2.2: scaled norms, log-magnitudes, merge, relative-magnitude classes, magnitude ordering, relative tie
tolerance. Replace hash identity in decisions by the matching certificate; keep the artifact SHA-256 for custody. Make
both repositories produce byte-identical canonical artifacts for the same input, with a cross-repository parity test.
Tests: A1 to A12. Complexity as A5 and A7.

**P0-3 Scheduler semantics** (`mms_utility.py`).
Accept paired differences $D$ as feedback and reject raw progress; add `STOP`; log propensities; feasibility by cost
upper bounds; net-rate comparison instead of `gain / cost`; statistics in physical coordinates or behind a transport
record; label-permutation and yoked controls. Tests: the null simulation must select ordinary or STOP in at least 90%
of windows; equal negative gains must not prefer the costlier action; an unmeasured action must be infeasible.

**P0-4 Runner** (new `mms07_campaign.py`).
World generator with per-world disjoint train, feedback, monitor and final cohorts; monitor access, checkpoint selection
and early stopping for every arm; calibrated update budgets with the CPU ledger as a check; outer wall and process
clocks around setup, timers and cleanup; incremental RAM reservation before allocation; fail closed on any breach.
Tests: ledger closure within 2%; identical monitor schedule across arms; replay determinism at fixed update budgets.

**P0-5 Telemetry in receipts.**
Per arm: $\cos\varphi$ histogram and the parallel/orthogonal split of $\lVert\delta\rVert^2$; pushforward certificate
pass rate and its agreement with local alignment; $r(\text{age})$ by age bin with hit-age quantiles; $\chi$ and
$\lVert\Pi_{U_S}q\rVert/\lVert q\rVert$; $\kappa(G_S)$, inflation $\iota$, match certificate at each refresh; exact
$\langle g_{\text{ref}},\Delta u\rangle$; occupancy. None of these is an outcome.

**P1-1 Pushforward admission and local injection** (A1, A2). Hook returns the unmodified cotangent; the correction
$\delta^\top X_m$ is added to the site weight gradient after backward, per sample only where
$\langle\delta_b,q_b\rangle>\tau\lVert\delta_b\rVert\lVert q_b\rVert$. Test: $\sum_b\langle\delta_b,q_b\rangle$ equals
$\langle\tilde u,d\rangle$ to $10^{-6}$ relative.

**P1-2 Gram-aware operators** (A6): `projector in {"outer","orth","ridge","dual"}` with Gram receipts.

**P1-3 Controls:** dose-only ($\delta=\rho c$, no bank); pushforward-only ($\delta_b=\rho\lVert c_b\rVert q_b/\lVert q_b\rVert$,
no bank); key-permuted sham (history of another hit sample, same admission rule and norm); schedule-only (all bank and
projection work performed, $\delta$ discarded).

**P1-4 Control-variate table** (A4) with running sum maintenance, per-stratum sums, $\alpha$ from previous-step
statistics, and the $r^2$ receipt.

**P2-1 Clone audits and stopping** (A10). **P2-2 Adaptive refresh** (A9). **P2-3 Prefix JVP audit** (A3) at low
frequency, to measure how far the local certificate is from the hook-propagation truth. **P2-4 Asynchronous GPU
pipeline** (§5.3), design only until a CPU arm passes.

**P3 LBF-20H** (§6.3), with A11.

---

## Executable checks

`python research/dg_gym/mms07_math_checks.py --out research/dg_gym/receipts/mms07_math_checks_receipt.json`
(NumPy and standard library; about 15 s; passes `ruff check` under the repository configuration). 29 of 29 pass.
The receipt records the script and module SHA-256, interpreter, NumPy version and platform.

| Check | Establishes |
| --- | --- |
| `p1_scalar_counterexample` | CE1 and its decomposition |
| `p1_anisotropy_counterexample` | CE2 |
| `p1_adjoint_identity_tied_net` | Lemma 1.1, local certificate, Theorem 1.5 bound |
| `p1_local_alignment_failure_rate_synthetic` | [S] alignment statistics, CE6 |
| `p1_first_order_prediction_and_dominance` | Proposition 1.2, Theorem 1.8 |
| `p1_cone_bound_tightness` | Theorem 1.4 |
| `p1_tied_recurrence_local_vs_full` | CE3 |
| `p1_adam_sensitivity` | Lemma 1.7a, CE4 |
| `p1_truncated_gradient_angle_budget` | Theorem 1.6, CE5 |
| `p1_smoothness_sufficient_condition` | Proposition 1.3 |
| `p1_adam_scale_null` | Proposition 1.7 |
| `p1_control_variate_table` | Theorem 1.9 |
| `p2_activity_gauge_defect`, `p2_extreme_scale_imbalance` | D4, reference canonical form |
| `p2_component_id_instability` | D5, matching certificate |
| `p2_selection_scale_and_cancelling_pair` | D6, D7 |
| `p2_gl_mixing` | Lemma 2.2, invariant-mask criterion |
| `p2_outer_sum_vs_projector` | D8, Propositions 2.6 and 2.7 |
| `p3_delay_independent_stability` | Proposition 3.2, CE7 |
| `p3_staleness_depth_scaling` | [S] depth ordering predicted by Theorem 3.1 |
| `p3_refresh_square_root_law` | Theorem 3.3 |
| `p4_scheduler_null_simulation` | D2 |
| `p4_ratio_utility_negative_numerator` | D3 |
| `p4_estimator_comparison` | [S] §4.2 |
| `p4_confidence_sequence_coverage` | §4.3 boundary |
| `p5_hit_rate_laws` | Propositions 5.1 and 5.2, D1 |
| `p5_capacity_numbers` | row and capacity arithmetic |
| `p5_sandbox_microbenchmark` | informational timing constants |
| `p7_receipt_deductions` | D9 and the scheduler lookup share, read from committed receipts |

---

## Falsification and promotion criteria

The full statement is in `MMS_07_REGISTRATION_DRAFT.md` §10. In summary:

**Not evidence of learning benefit, alone or together:** cache hit rate, reconstruction cosine, local alignment,
pushforward certificate pass rate, variance ratio $1-r^2$, probe accuracy, gradient coverage, positive raw window
progress.

**Promotion of a replay arm** requires all of: (i) a positive world-level paired difference against early-stopping
protected ordinary at the registered compute budget, by the registered exact sign-flip test under Holm; (ii) the same
against the key-permuted sham, the dose-only control and the pushforward-only control; (iii) an effect at or above the
registered margin $\delta_{\min}$; (iv) the same sign with one-sided $p<0.05$ on the second gym; (v) an EvalMath
preflight verdict of `READY` on the confirmation plan before any sealed cohort is opened.

**Refutation at the registered scale:** with at least 16 worlds, every primary replay arm has a one-sided 95% upper
bound on the mean paired difference below $\delta_{\min}$.

**Theory-level falsifiers:** F5.1 to F5.8; a consistent effect of the dose-only arm under AdamW (against
Proposition 1.7); agreement above 95% between local alignment and pushforward admission on real hits (pushforward
admission redundant); certificate sign uncorrelated with clone-audit sign (certificate useless as a controller input);
a sham-controlled replay benefit with $r<0.2$ at the median hit age (staleness theory wrong).

---

## Review notes: what is established, what is numerical, what is conjecture

**Established here (proved; verified by executed checks where marked).**
Lemma 1.1; Propositions 1.2, 1.3, 1.7; Theorems 1.4 (with tightness), 1.5, 1.6, 1.8, 1.9; Lemma 1.7a; Lemmas 2.1, 2.2;
Propositions 2.3, 2.5, 2.6, 2.7; Proposition 3.2; Theorems 3.1, 3.3; Proposition 4.2; Propositions 5.1, 5.2;
Theorem 6.1; Propositions 6.2, 6.3. Defects D1 to D8 by execution against pinned code. Counterexamples CE1 to CE5, CE7,
CE8.

**Read or deduced from committed material, not re-executed.**
D9 (receipt arithmetic), D10, D11, D12, D14 (code reading), D13 (tick arithmetic); the LT/DG component-ID representation
difference; hook propagation semantics in both repositories; all MMS-06 cost ratios; the LBF-20M ratios.

**Numerical evidence of limited scope.**
Synthetic alignment statistics on random tied networks (median $\cos(c,q)$ 0.47; per-sample reversal rates); synthetic
$P$/$\Pi$ admission disagreement; the depth ordering of cotangent sensitivity; the toy estimator comparison; sandbox
timing constants. These illustrate the theorems. They are not measurements on any gym and they say nothing about
learning benefit.

**Conjectures.**
(C1) The control-variate table improves verified learning per unit compute on small fixed pools. (C2) Pushforward
admission improves on local-alignment admission in outcome, not only in certificate. (C3) $1-r(\text{age})$ follows
$\text{age}^2$ early and $\text{age}$ late. (C4) The hybrid of §6.3 beats its components under a common budget. (C5)
The overhead predictions F5.4 to F5.6 transfer from sandbox constants to the MMS-06 host. (C6) History is more useful as
curvature information than as a replayed direction.

**Assumptions that could change the conclusions.**
Theorem 1.8 is first order and for a fixed metric; a second-order benefit of history is not excluded. Proposition 1.7
needs $\epsilon\to0$, a cut site and full coverage; none holds exactly in MMS-06, which is why the dose-only arm is a
registered control and not an assumption. Theorem 3.1 uses global Lipschitz constants that are loose in practice.
Proposition 4.2 assumes value locally linear in steps. Proposition 5.2 assumes uniform access and a hash that behaves
randomly, which D1 shows the pinned hash does not. D1 was computed for contiguous example IDs; other ID sets change the
counts but not the capacity factorization.

**Recalled literature (not reopened).** Kantorovich inequality and antieigenvalues (Gustafson); Wedin and Davis–Kahan
$\sin\Theta$ theorems; SAGA (Defazio, Bach, Lacoste-Julien, 2014); Defazio and Bottou, "On the ineffectiveness of
variance reduced optimization for deep learning" (2019); Young (1974) and Daly (2006) checkpoint intervals; Charnov's
marginal value theorem (1976); Robbins (1970) and Howard, Ramdas, McAuliffe, Sekhon (2021) on confidence sequences;
Frazier, Powell, Dayanik (2008) knowledge gradient; Cao and Irani (1997) GreedyDual-Size; Lykouris and Vassilvitskii
(2018) caching with predictions. Citations should be checked before external publication.
