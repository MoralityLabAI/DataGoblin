# ZKL-21: cached symbolic preparation and closed learned-certificate replay

ZKL-20 preserved the full recurrent action trace and all 93/96 certificates
at 96 columns. It improved deadline coverage from 77/96 to 85/96. Search
CPU fell from 483.1 to 275.4 ms, but setup comprised about 71% of the
optimized end-to-end time. ZKL-21 caches row-level and multiplier-level
invariants and evaluates only the seven terms used by the original static
ranking. Original float association, monomial supports, tie values and RNG
state remain unchanged. Both row-heap recurrent and reset arms are compared
with the cached versions. No new model parameters, no changed training.

Fresh root blocks 21101–21108 and 21201–21208; 128 tasks (96 certifiable,
32 essential), F11/BN254, eight arms, three protocols (3072 evaluation
runs). Pinned SatsHedgeZero checker 84c1681a99ef5897f0f6b90140a0ac9e3293f5cb. Every uncapped pair must
match full trace, certificate, work and heap metrics; deadline traces
must share their entire common prefix. Separate setup/search/verification
CPU, 512 MiB virtual-address cap, whole-process RSS only.

The first successful cached-recurrent c96 certificate per nonessential
family/field, in ascending root order across the two blocks, is chosen for
Lean replay. All candidate seeds and successful seeds are recorded, with
explicit abstention if no success. Reference certificates are never
substituted. Regenerate the exact shape, reverify it with the pinned
Python checker, generate Lean source, discharge concrete wellFormed, and
prove accepts(original,x) iff accepts(replacement,x) for every x with
no external wellFormed premise. Pinned MathCan a29a46d465ddecb3a94a0367707befdc7f27ff28. The replay gate
requires all 12 family/field cells, successful Lean kernel checks and
the axiom audit; Python receipts remain unpromoted. This is still a
synthetic ordinary-R1CS campaign. No actual Nova production-matrix binding,
key reuse or production authorization.
