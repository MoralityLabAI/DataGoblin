# Independent MMS-06 review and MMS-07 gates

Scope: code inspection at DataGoblin c15012053f4db5bba7943a3fefdf818a9dfe9388
and LoopedTransformers 3dad5bd9f7b076fffdad9c717dbc98b04982cba7 / canonicalization
7eec3fab6ca7dabf62c7fc44c743522c5728dfa1; public development only.
Historical receipts stay immutable. The following are review findings, not
new training measurements. No paid compute. PRs 84, 85, 401 were green
at their pinned heads when checked. All remain drafts.

## Implemented in this review branch

### Hash semantics

Old Torch GradientBank multiplies signed int64 keys by 6364136223846793005,
wraps at 64 bits, then takes a nonnegative remainder. FastGradientBank v0
instead takes an unsigned remainder. For a negative signed product, the
two slots differ by 2^64 modulo capacity. At capacity 29127 this offset
is 1024. On all 12288 possible keys 3*i+1 in the bank profile, 6143 slots
differed; both mappings happened to be injective on that finite set. Thus
the reported equal hits and zero collisions do not prove collision parity.

Concrete witnesses at the same capacity:
* (0,23269): signed slots (0,0), unsigned slots (0,1024).
* (16,23285): signed slots (27837,28861), unsigned slots (28861,28861).

Independent NumPy/Torch reproduction: 21518 signed-slot comparisons
against Python arbitrary-integer arithmetic over 14 capacities passed;
the Torch reference agrees for the 12288-key domain. This was a local
reproduction of the source rules, not a full repository replay.

The review patch defaults to the signed policy and versions the hash
scheme in checkpoints and receipts. Original unsigned v0 remains explicit
for historical replay; unversioned v0 states may only be loaded into the
unsigned policy. State loading validates array shape/dtype, tag slots,
age and counters before assignment. Temporary copy memory is additional.
The original MMS-06 commit remains the historical replay target.

### Ledger snapshot

The original total properties omitted elapsed work inside an open scope,
including scheduler budget decisions within a scheduler scope. The patch
includes the active leaf scope in snapshots without modifying closed buckets.
This is a sequential process-clock ledger, not a concurrent worker profiler.
New regression cases cover collision witnesses, signed reference arithmetic,
state version compatibility, malformed restore and nested live accounting.

## Remaining review issues

### Evidence and resource scope

1. Registration specifies a 6-second fixed-CPU arm; receipts use 4 and 1.5
seconds. The lower budget was chosen after observing the 4-second outcome
and changing the projection implementation. Treat both as development and
register a fresh budget curve plus fresh final roots before confirmation.
The 4-second comparison is a valid comparison of those run-to-budget
policies, conditional on accounting, but does not identify a causal
per-update learning advantage. Test ordinary checkpoint selection/early
stopping with identical monitor access and selection cost. Do not use
final_dev to choose a checkpoint, budget, architecture or intervention.

2. The 1.5-second summary records 122x CPU / 132.46x wall bank speedup;
the 4-second receipt records 130x / 60.48x. Preserve both profiles and
clock resolution (15.625ms CPU ticks). The faster bank remains an important
engineering result, but repeat on colliding, capacity-pressured workloads
and distinguish bank from whole-step throughput. At fixed 400 updates
ordinary averages 8.255ms/step, dense 11.732, frozen SVD 12.917,
refreshed SVD 13.568. Extra whole-step costs are 3.477, 4.662, 5.313ms,
not the isolated 0.4-0.5ms operation quoted in the report. If those
increments were constant, a 5% overhead target would need baseline
steps >=69.53,93.23,106.25ms, respectively. Width/sequence length/rank
change the extra costs, so these are illustrative thresholds only.
The current 1.5s receipt's maximum sampled RSS is 689410048 B, higher
than the 442777600 B maximum in the 4s receipt. Reconcile the report's
438-443 MB claim with the two campaigns.

3. Runner has two persistent banks (0.5 and 4 MiB); 4718448 B owned was
reported for scheduler arms. The runner defines total_ram_cap_bytes as
(bank_mib+1) MiB, not the registered 4 MiB bank + 256 KiB scheduler.
It passes total cap rather than remaining headroom when any bank is live.
Selection may allocate a previously absent bank while another is retained.
The scheduler's fits_total_cap method is not invoked by the runner.
Use an incremental allocation reservation plus total live-storage
accounting and enforce before allocation; include maps, scratch, queues,
checkpoints and context metadata. Report measured peak separately.
CPU stopping after a completed step can overshoot: call it a common
stopping rule with observed overshoot, not an exact hard elapsed cap.

4. Outer campaign accounting: data construction, model initialization,
retro ordinary training and retro factor fitting are outside per-arm
ledger. Retro arm is diagnostic only, not a prospective competitor.
Add an outer clock and reconcile every unassigned interval; keep shared
preparation and amortization explicit. Both summaries carry the same
runner hash; dependency hashes from the exact pre-vectorization run must
be retained, not inferred from the later report.

5. LBF-20M: the code independently checks only trial zero's last inner
repetition for each arm/root/workload, not every timed answer. Other runs
retain internal checks. Query timing excludes cache initialization/reset,
setup and final verification; global lift/project caches persist across
inner repetitions. Call this a cold query-memo workload with potentially
warm helper caches, not a wholly cold solver. Positive query-wall gains at
12 KiB remain 16.82%,20.43%,23.22%. Holding setup and verification at the
disabled trial-zero values gives illustrative pipeline savings of
4.19%,7.79%,13.49%, not measured end-to-end effects. Publish paired
per-root/per-trial pipeline differences with consistent verification and
cache reset scopes, plus independent output-digest checks for all repeats.
Report the 89 messages vs 1422/2844/5688; fixed tier results are conditional
on these selected 12 roots. Cap covers estimated query payload only.
No inference about arbitrary workloads or RSS reduction.

### Control and causal semantics

6. The scheduler still pools different checkpoints, basis versions,
split digests and horizons in metric_context_key, and traces are keyed
by trajectory_id without complete action/receipt binding. Missing traces
fall back to reported action; repeated signals can be credited twice;
context and trajectory counts can grow without a global cap. Retrieval's
validated_transport_ok is a boolean bypass, not a transport operator.
Use explicit run/trajectory/receipt IDs and monotone windows, consume
each outcome once, distinguish trusted direct observations from delayed
attribution; transport records need versioned mappings and validation.
Unknown CPU cost passes feasibility and ordinary can be returned even
when no action fits; distinguish STOP from ordinary and reserve complete
windows using upper costs, including monitoring.

7. The measured feedback value is the change in regret across the last
training window, not the incremental value versus a paired ordinary
continuation from the same checkpoint. Nonstationary learning curves
confound sequential comparisons. SNR/exploration do not turn the observed
change into a causal estimate. Add matched cloned or randomized windows,
a no-feedback/reinvestment comparator, and schedule-only shadow controls.
Shuffled feedback currently randomly changes the sign of the gain
magnitude, so it changes distribution; add a matched permutation control.
Feedback, cache policy and checkpoint selection must obey data custody.

8. Dense in the new runner admits by dot(h,current)>0, while old dense
uses the projected BPD direction for admission, then applies h. Treat
the new dense arm as a different policy; profile implementation parity
separately from comparing those policies.

### Mathematical notes for Fable

9. Gauge invariance has a concrete boundary counterexample. The current
active predicate requires ||U_j|| and ||V_j|| individually > eps. The
rank-one scalar factors (1,1) and (1e13,1e-13) reconstruct the same
unit matrix, but only the first is active at eps=1e-12. Use a scale-safe
rank-one norm ||V_j||*||U_j|| to determine activity, and distinguish
exact zeros, numerically unrepresentable factors and deliberate pruning.
Canonical sign/order only handles reciprocal scaling and permutation,
not arbitrary GL(C) factor mixing or rotations within degenerate SVD
subspaces. Rounded component IDs/basis hashes are approximate identities
and can alias different maps. Separate content hash, canonical approximate
match, and validated transport. Handle score ties and near ties explicitly.

10. The selected component outer-product sum P=U_S^T U_S is positive
semidefinite but is generally not an orthogonal projector. Its operator
norm may exceed one; duplicate components alter direction. Compare
sum-of-components with subspace projection using a stable QR/SVD or
Gram pseudoinverse (with rank/tolerance receipts).

11. Local cotangent alignment does not imply parameter-space descent.
Let shared scalar theta have loss
L(theta)=((theta+1)^2+(theta-1.1)^2)/2. At theta=0 the two activation
cotangents are 1 and -1.1, total parameter gradient -0.1. Inject +0.2
at the first activation, giving local dot +0.2 but new parameter
gradient +0.1. With SGD eta=.1 the initial loss 1.105 becomes 1.1041
under ordinary SGD, or 1.1061 under the steered update. This exact
counterexample was reproduced independently.
For g=grad L and parameter correction d=J_site^T delta, a smoothness
bound is L(theta-eta*(g+d)) <= L(theta)-eta<g,g+d> +
(L_smooth eta^2/2)||g+d||^2. For comparison with plain SGD, a sufficient
condition under an L_smooth bound is
<g,d> > (L_smooth eta/2)(||g+d||^2+||g||^2).
AdamW needs its actual stateful update, not an SGD surrogate. Audit
alignment in parameter/update space, task-loss stability, and net held-out
learning gain. No mathematical guarantee of generalization follows.

12. Scale model for an output-gradient row of shape (T,d):
row bytes=2*T*d+16, capacity=floor(M/row bytes); actual storage includes
scratch and maps. Projection traffic and work depend on batch, tokens,
width, rank and selected components, not just model step duration.
Under a fixed 4 MiB bank, T=d=128 gives only 127 rows; that is a
hypothetical shape, not a measured RMP configuration. Benchmark actual
RMP tensor layouts before choosing RAM tiers. Test exact cache as a
distinct intervention from approximate gradient replay.

## Disposition

Do not promote neural MMS on LBF-20D. 1.5s SVD-frozen and dense are
2/3 positive seeds; utility loses all three. Fixed exposure ordinary
wins the observed comparison. Positive exact-cache query efficiency
is scoped to the measured development workload. Run a new prospective
MMS-07 with common budgets, fresh roots, matched early stopping and
versioned mechanisms. No results from the repaired hash/ledger are
claimed for historical fits until rerun.
