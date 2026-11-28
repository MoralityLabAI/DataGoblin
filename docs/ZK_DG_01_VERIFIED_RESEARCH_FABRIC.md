# ZK-DG-01: Verified Research Fabric, replay baseline

Status: transparent development oracle. No SNARK, privacy, succinctness,
hardware attestation, Lean checker execution, or Lucid integration is claimed.
Do not admit replay receipts under a cryptographic-proof policy.

## Typed plan and graph semantics

The relying party must authorize the exact Plan independently of the worker.
Plan binds tenant, run ID, policy ID and a topologically ordered list of jobs.
A job has a unique name, ordered parent names, a source vector, integer
weights, bias, and recurrent depth. A graph is limited to 128 jobs, common
width 1..16 and depth 1..64. There is exactly one sink, the final job.
All jobs reach it: following outgoing edges in a finite DAG must terminate
at a sink. This is an elementary graph argument, not a Lean-checked theorem.

For each job, x = clamp(source + sum(parent outputs)), h0 = x, and

    h[t+1] = clamp(W h[t] + b + x)

componentwise, with clamp to [-127,127]. Every vector and weight entry is
an integer in this interval; booleans are not integers for admission.
This is a small weight-tied recurrent arithmetic surrogate, not an
implementation of attention, a trained TRM, BPD, or Inky Mini.

A statement binds backend, full plan hash, job name, ordered parent receipt
hashes, model hash, input and output hashes, and exact depth.
The replay receipt includes all states in clear. Hashes are domain-separated
SHA-256 over compact sorted-key ASCII JSON, limited to the fixture's typed
values. Tuples encode as JSON arrays. The API assumes bounded, typed trusted
plans; a production network must add strict decoding, byte limits, versioned
schemas and authenticated authorization. No signatures or hiding commitments
are provided. Low-entropy values are subject to dictionary guessing.

Verification recomputes all expected receipts from the authorized plan.
Induction on the job order and recurrence depth establishes deterministic
agreement for the fixture relation, subject to correct implementation.
A relying party must independently pin the tenant, unique run ID, plan and
policy, and maintain an atomic consumed-run ledger if one-time admission is
required. Replaying the same valid plan is intentionally idempotent here.

## Local benchmark and validation

Isolated Python execution, 38 tests passing. At width 2 and depth 32, nine
verification repetitions per size (median milliseconds; illustrative only):

| Jobs | Steps | Clear JSON bytes | Median replay ms |
| ---: | ---: | ---: | ---: |
| 8 | 256 | 7109 | 0.896 |
| 32 | 1024 | 29259 | 3.121 |
| 128 | 4096 | 117895 | 12.814 |

The DAG joins the previous two outputs where available. These timings and
sizes measure full replay, not ZK proof size or verification. Unit tests include
a hand-computed trace, a factorial job/depth sweep, statement/trace tampering,
missing and reordered receipts, changed tenant/run/policy, and invalid plans.
No checkout-wide suite or hosted CI was run locally.

    PYTHONPATH=src python -m pytest -q tests/test_zk_replay.py

## Next gate: native Nova arithmetic circuit

Use a pinned microsoft/Nova revision, initially a width-2 circuit with a
fixed shape and Pedersen/IPA parameters. Carry x, h, counter, and immutable
model/policy context across steps. A step must constrain every multiplication,
sum, signed range, clamp case, counter increment, and immutable state carry.
For width <=16 the absolute pre-clamp sum is <= 16*127^2+254 = 258318.
Use signed range decomposition and mutually exclusive clamp cases; field
arithmetic without these constraints does not establish integer semantics.
The initial state and final output must be bound to the authorized statement.
If using a field-friendly hash inside the circuit, specify a versioned
binding to the SHA-256 envelope or use a new manifest format; do not silently
reinterpret digests. A public z0 or zn may expose model/input/state values.
Private witnesses require hiding commitments and the appropriate final
zero-knowledge compression, plus tests for public-input leakage.

Compare native execution, recursive proof and compressed proof at depths
1,2,4,8,16,32; report setup, constraints, peak RSS, prover time, verifier time,
serialized proof and exact public inputs. Require verification failure for
wrong initial state, final state, depth, model, job context and mutated proof.
Do not promote beyond the replay backend until an independent circuit audit.

A separate graph relation must constrain each authorized edge, ordered input
opening, parent verifier identity, and exactly the required job set. Single
chain IVC is insufficient for general joins. First benchmark batch proof
verification of each vertex and then investigate recursive proof-carrying data.
Bind relation and verification-key identities; worker-provided keys are not
trusted. Make failure and cancellation explicit in the production manifest.

## Lean lane

Use the existing docs/rlt_external_lean_mcp_verifiers.md interface. Introduce
a distinct lean-checker statement containing theorem identity/type, checker
and environment hashes, axiom policy, dependency closure and result.
Require the checker to verify the exact proof term in the pinned environment,
including axiom collection over the transitive closure; reject sorryAx and
unapproved dependencies. Check that the statement does not resolve to a
different theorem. Keep policy choices such as Classical.choice explicit.
A successful process exit or hash of a proof file is insufficient.

Measure supported zkPi theorems separately from a pinned checker in a zkVM.
Retain unsupported cases and failures in denominators. Proving a checker
execution inherits its logical assumptions and implementation correctness.
Metamath cross-checking is a separate predicate, where a translation exists.

## Lucid and federation

Lucid's documented registry currently lists groth16, plonk, fflonk and stark.
Nova compressed proofs are not a listed native proof type. Use an explicit
adapter only after format, public-input encoding, circuit/key registration,
and verification behavior are demonstrated. Attestation evidence and
cryptographic computation proofs must retain distinct backend IDs and policy
classes. OpenSilico authorizes plans; Tesseract schedules and transports
typed messages; DataGoblin emits and consumes verified receipts.
No live integration or production security claim is made.

References:
- https://github.com/microsoft/Nova
- https://eprint.iacr.org/2024/267
- https://docs.lucidcomputing.ai/api-reference
- https://lucidcomputing.ai/blog/introducing-inference-proofs
