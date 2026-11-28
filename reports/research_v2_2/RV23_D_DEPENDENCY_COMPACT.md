# RV23-D: dependency-mask compactification

Parent RV23-C 3fbf8367 passed both hosted workflows (827 passed, 16
skipped, clean lint). CPU-only synthetic public experiments. Frozen
native upstream 7fa19c90372b13d71676e6fef13819d2a18b9be3.

## Mathematical constructions

Let S be a closed selected set, R the unselected separators for one pair,
d its positive deficit, and A_i=C(i) minus S for i in R. Every feasible
completion for the pair contains the union of at least d distinct
separator closures. Equivalently minimize cost(U) over unions U of A_i
such that |U intersect R| >= d. One direction follows because any d
separator choices cover d separators. In the other direction choose
d separators in U: closure of each is contained in S union U, since
the latter is closed. Their residual union is a subset of U and
cannot cost more. This equivalence handles nested closures and cycles
(co-selection constraints, not a feasible chronological schedule).

Exact quotient: graph states are residual union masks, edges U -> U OR A_i,
absolute cost nondecreasing with strictly positive item costs on strict
growth. Best-first expansion by union cost, deduplicate exact masks,
and stop on the first union satisfying the deficit. This gives the
same exact lower bound as combination enumeration. The within-world
cache key is (remaining separator mask, deficit, sorted distinct
residual closures); costs are fixed within the solve. Equal signatures
imply identical quotient graphs and objectives. No cross-world reuse.
Cap at 256 entries; retain only exact completed bounds.

Union atlas: enumerate all original need=2e+1 separator combinations
for a pair, deduplicate full dependency union masks, and evaluate
min cost(U minus S) at later selected states. Any pair-feasible
completion contains at least one original need-set closure; any
such closure joined to S covers the pair. Thus the atlas gives the
same exact residual bound, even if some separators are already
selected. Identical original separator masks share the atlas; cap
retained atlas masks at 4096. A nonretained atlas can still be used
for the current bound. No unbounded memo.

Cheap control bound: sort the residual closures' total prices and
their prices restricted to nonseparators. Any completion must contain
d separators, at least the sum of d cheapest direct separator costs,
and at least the d-th smallest nonseparator closure price. These
cost classes are disjoint. It must also pay at least the d-th
smallest total closure price. Max these two lower bounds. Independently
verify this new witness by set-based reconstruction in v2 proof
checker. Never sum across pairs. Old v1 proofs remain replayable.

## Work and evaluation

Same RV23-C proof-producing search for all arms. Cover, forced, original
all-pair closure enumeration, cheap control bound, residual quotient
with/without exact cache, full-union atlas. New exact arms first
compute cheap control bound and avoid exact refinement if it closes
the node. 20,000 exact work terms per solve (quotient candidate OR
attempts, or atlas construction combinations plus scan terms).
Legacy enumeration counts combinations separately. Partial
minimizations never authorize pruning; fallback to previous bound.
Also reserve original combination count for every newly selected
exact witness, conservatively capped at 20,000, so the independent
enumerating proof checker can replay it. Checker uses original
combination enumeration for closure witnesses, not quotient/atlas.
CPU includes solver, proof production, cache accounting; verification
CPU separately measured. Shallow retained bound-cache estimate includes
memo/atlas keys and values, may double count shared integers, excludes
transient heap, proof, common solver and RSS. Count caps not RAM caps.
One randomized timing per arm/root/cap, paired bootstrap over roots,
no multiplicity adjustment.

Fresh 24 native frontier roots, 16 each shared/transitive and bundled
structured 16-item worlds, 72 roots; caps 64 and 65536, 7 arms,
1008 searches. Bundled cases deliberately use co-selection cycles.
Exclude all 176 previous A/B/C raw root identities. No structural
isomorphism claim. Full-cap exact oracle and certificate replay
required on all roots, plus native reference comparison on native
worlds. Independent tests compare residual and atlas bounds with
explicit combinations on small closed states; verify cheap bound
inequality, cached result, exact full search, budget fallback and
tampering. Native result pending. No learned model, promotion,
GPU or paid inference.
