# DG-Gym 0.2: execution and competence calculus

Status: public-development proposal, additive planning primitives. No paid compute,
no production launch authority, no corpus admission. The existing 1.0 trainer
contract is unchanged.

A gym G = <E,D,A,I,L,Pi,H,B,Q> compiles to a typed dependency graph,
placement and a reservation plan. A job v has cumulative work/traffic
counters and a separate exclusive reservation (RAM bytes, VRAM bytes, CPU
cores). Durations are estimated positive integer ticks; measured values,
estimates, proposals and unavailable values are distinct. Shared resident
pools must be reserved outside the job scheduler. Multi-device placement,
bandwidth contention and activation lifetime extraction require calibration
and additional adapters.

For physical storage s with byte size m_s, memory pool d_s and half-open
lifetime [a_s,b_s), peak memory in pool d is

    M_d = max_t sum_{s: d_s=d, a_s<=t<b_s} m_s.

Storage identities are physical; identical alias records are counted once.
Conflicting records for one identity are rejected. The event sweep combines
all arrivals and releases at a time boundary before measuring the peak.
A storage retained across visits must retain its lifetime across those visits.

For job start times t_v, duration tau_v and demand r_v,d, require

    t_v >= t_u + tau_u                   for every dependency u -> v
    sum_{v: t_v<=t<t_v+tau_v} r_v,d <= B_d for every resource d and time t.

The included deterministic list scheduler starts ready jobs in sorted key order
when their reservations fit, then advances to the next completion. Reservation
safety follows by induction: a start is permitted only if all coordinate sums
fit, and a completion releases demand. For a finite DAG with individually
feasible jobs, a nonempty remaining DAG always has a ready vertex when no job
is active; positive durations and finite jobs imply termination. Cycles and
unknown dependencies are rejected. Makespan optimality is not claimed.

The cumulative work and transfer counters should be additive even when execution
overlaps. A deadline must include preparation and checkpoint work. The estimated
scheduler is not an OS resource enforcer or a performance benchmark.

## Paired intervention receipts

For aligned binary outcomes (A,B), record n00,n01,n10,n11. Then

    gain = (n01-n10)/N
    recovery = n01/(n00+n01)
    corruption = n10/(n10+n11)
    gain = (1-q_A)*recovery - q_A*corruption.

Zero conditional denominators are null. Dependence across training fits and
shared roots precludes interpreting all rows as independent observations.
The implementation computes the counts and effects only, leaving uncertainty
and promotion to a registered analysis.

Before a mechanistic campaign, preregister a competence threshold relative
to the strongest relevant heuristic, with independent public-development
roots, and require meaningful headroom. Also identify informative roots where
changing the concept can change the verifier target.

MOD7 illustrates this condition. An affine suffix over F_7 has slope equal
to the product of the remaining multipliers. If any multiplier is zero, the
suffix is constant; if all are nonzero, it is bijective. Under a fixed bijective
suffix, distinct intermediate residues yield distinct terminal targets. This
does not establish that a learned model represents the residue.

## Advisory RAM reclaim policy

Only rank owned disposable leaf workers under memory pressure, after
an expired lease, stale heartbeat, low CPU work over the specified interval,
zero observed I/O and known inactive GPU status. Unknown GPU activity protects
the worker. RSS ranks candidates but is not necessarily reclaimable memory.
Use (pid, process-start identity) and revalidate ownership, identity,
lease and activity immediately before any OS termination. The reference
function returns candidates; it performs no process signaling, telemetry
collection or automatic kill. A supervisor must implement graceful stop,
timeout/escalation, checkpoint protection and auditing.

## Validation

The standard-library implementation has six focused tests, including 256
paired binary patterns of length four, 27 resource-cap combinations, lifetime
aliasing, dependency rejection and watchdog protections. A standalone copy
passed these checks locally; repository CI is still required.

A separate CPU public-development pilot used a 16,782-parameter GRU, 3 seeds,
150 steps per arm, 96 examples per update, four training visits, length-five
calibration and length-six development. It is a fixed-exposure pilot, not
the proposed v0.1 fixed-wall-time experiment. Each arm and seed solved
124/512 development cases, tying the last additive operand heuristic.
Across the three fits the residue-probe scores were 349/1536 (task loss),
370/1536 (concept supervision), 354/1536 (shuffled labels). Every intervention
and control scored 48/318 on changed-target dev roots in each fit. There was
no measured recovery or causal specificity. Disposition: redesign before
promotion; prefer nonzero multipliers and a competence screen. No hardware
throughput or peak RAM claim transfers to a user's devices.

Next adapters: compile DG-Gym/0.1 validated manifests to jobs, record live
allocated and reserved memory by device, perform bounded subprocess
supervision, bind measured receipts to spec and checkpoint hashes, then
compare matched wall-budget arms on the bijective fixture. Retain the
existing verifier registry and Model Atlas evidence/corpus separation.
