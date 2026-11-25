# ZKL-15 native affine bridge

Source index expressions in the original grammar contain 0, n, succ and addition. Structural recursion computes their affine coefficients (a,b), denoting a*n+b. This identity follows directly by induction. The transport accepts exactly coefficient 1 for universal rule indices and 0 for ground rules, and encodes the two families as atom 2*index+family. An independent replay validates one shared binder across the rule, all backward premise references, capacity, integer bounds and the goal. A SHA-256 environment ID is a transport identifier, not a cryptographic proof or Nova's Poseidon rule root.

Local CPU evaluation: 41 focused tests passed. 256 fresh public generated environments were solved with ordered-height A*, replayed using the source relation, then translated and checked by the independent packed-integer reference.

| Split | Replayed | Total certificate rows | Largest certificate | Rules per environment |
|---|---:|---:|---:|---:|
| 12-index (90000–90127) | 128/128 | 714 | 13 | 16 |
| shifted 24-index (91000–91127) | 128/128 | 1130 | 18 | 22 |

No capped cases. Tests cover all four affine syntax styles, 24 additional independent source instances, the common P(3)→Q(4)→P(5) fixture, premise/binder/goal attacks and overflow. Run python affine_bridge.py and python -m pytest -q test_affine_bridge.py from the indexed directory. No model trained, no paid compute, no sealed confirmation claimed.

The Rust ZKL-15 candidate is in [zk-rusto-no-bitcoino PR 34](https://github.com/patrickdugan/zk-rusto-no-bitcoino/pull/34). It currently uses a four-rule fixture; the Python transport does not yet emit a proof to that Rust binary. Actual circuit synthesis, cross-language Poseidon transport for 16/22 rule environments and general dependent term reduction are independent next gates. Do not apply ground Horn constraint measurements to this new fragment as measured costs.

For EvalMath: report checked solve coverage, optimal row count, actual rows, admissible lower bound and verified cost identity separately. Invalid or missing certificates cannot receive cost credit. When exact optimization caps, report an interval instead of pretending the best found cost is optimal. Bind the cost profile to a particular circuit and version. No claims about model generalization follow from this symbolic finite-domain experiment.
