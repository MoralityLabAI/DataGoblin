# ZKL-21: authenticated structural context jumps

Measured ZKL-20 Nova baseline: 25,952 raw constraints per row at depth 6, 36,386 at depth 9, 11,472 bytes per compressed proof. Both verified: https://github.com/patrickdugan/zk-rusto-no-bitcoino/actions/runs/45594791750 . ZKL-21 adds a bounded power-of-two ancestor record and a local lookup rule. The public host walks parent pointers to validate each record before commitment. Proofs authenticate the source, destination, distance, and earlier lookup judgment. The explicit bulk shift from ZKL-20 is preserved. Public syntax and the used context jumps remain visible.

The native rewrite takes a valid existing bulk DAG, replaces long unit lookups by checked binary jumps, and removes only unused lookup rows. Independent expansion of the jumps back to unit rules and semantic replay confirm validity. All 576 paired public exercises pass, 500 improve and 76 tie; 44,495 different-signature predecessor substitutions and 66 malformed public jump records rejected in the fixed/stress audit. 192 of the generated roots are disjoint fresh diagnostics; the others reuse ZKL-20 cohorts.

| cohort | bulk rows | jump rows | bulk objects | jump objects |
|---|---:|---:|---:|---:|
| 22000 | 22135 | 19491 | 9701 | 11320 |
| 32000 | 22424 | 19722 | 9835 | 11498 |
| 42000 fresh | 23127 | 20244 | 10066 | 11784 |

Each cohort uses 64 roots each for weakening, dependent application, and conversion (seed offsets 0,1000,2000). Stress seed 23004 shrinks 270->240 rows, 112->131 objects. All fit the 512-slot profile, but capacity and extra public metadata must be accounted for alongside row count. The verifier's structural host work has not been measured. A logarithmic number of local lookup transitions is not an end-to-end complexity claim. No learned policy, statistical confirmation, or training release. Fixed rule table and generator source c80325149aec7a2255dc8803e3a81bf530663574 in zk-rusto-no-bitcoino. Actual new Nova constraints and verification await the depth 6/9 jobs; do not equate old measured cost per step with new measured cost.

Curriculum implication: the stronger native baseline now uses bulk lifting and binary context jumps. Learn to choose context and substitution judgments only after an independent checker admits the entire certificate. Compare to this deterministic rewrite under fixed capacity and profile, keeping every failed/capped root in the denominator. A private witness profile requires authenticated construction of the syntax arena and ancestor relations, rather than trusted public structural admission.
