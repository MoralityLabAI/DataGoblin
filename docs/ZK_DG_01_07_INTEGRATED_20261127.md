# ZK-DG-01–07 integrated development campaign

Date: 2026-11-27. All seven lanes have source-level integration; the
development Ristretto backend and public graph/corpus paths ran locally.
Native Nova, native Lean/zkPi, recursive DAG aggregation, trained routing,
and live Lucid/OpenSilico/Tesseract deployment remain unexecuted.
This is an unaudited research prototype, not production cryptography.

## Reproduction

Requires Python 3.10+ and a system libsodium exposing Ristretto255.
The external verifier mock test uses a POSIX executable script.
No network calls or external teacher inference are needed.

    PYTHONPATH=src python -m pytest -q tests/test_zk_replay.py tests/test_zk_fabric.py

Executed in isolated Python 3.13.5 with libsodium.so.23:
74 passed in 3.15 s (38 prior replay cases + 36 integrated cases).
The repository-wide suite, lint and mypy were not executed in this environment.
No Cargo or Lean toolchain; cloning and package downloads were unavailable.
Keep the PR in draft until checkout-wide verification and independent review.

## 01 and 02: execution and arithmetic proofs

zk_replay.py remains the transparent reference oracle.
zk_sigma.py uses libsodium's Ristretto255 group and SHA-512-derived
independent generator H. For a witness v, commitment C(v,r)=vG+rH.
A linear equality is proved by knowledge of the H-opening of its
homomorphic commitment. For A=C(a,r), B=C(b,s), D=C(d,t), the product
relation ab=d is checked by a common Fiat-Shamir challenge e:

    T1=uG+vH, T2=uB+wH
    z1=u+ea, z2=v+er, z3=w+e(t-as)
    z1G+z2H=T1+eA
    z1B+z3H=T2+eD

The transcript includes the backend, exact public statement, all commitments
and all announcements. The verifier constructs the same relation itself
and rejects malformed scalar/point encodings and changed public commitments.
Blinding and nonce randomness come from the system CSPRNG.
This prototype has not received a formal security proof or third-party audit.
Group arithmetic alone does not validate the surrounding protocol.

For pre-clamp q and result y, use:

    -127 <= y <= 127
    q-y=a-b
    a>=0, b>=0
    a(127-y)=0
    b(127+y)=0

Both inequalities on y and the bounds on a,b use constrained binary
decomposition; each bit satisfies b*b=b. If y is interior, a=b=0;
if y=127 then b=0 and q>=127; if y=-127 then a=0 and q<=-127.
Conversely any correctly clamped result admits a,b. For authorized
integer inputs, the field modulus is large enough to preclude wraparound
in these bounded relations.

The experimental sigma backend is non-succinct, with public model,
input and output. Commitments hide intermediate witness openings,
but the deterministic states can be recomputed from public inputs.
It does not provide confidentiality of a proprietary input or model,
and does not prove full attention, actual TRM, BPD or Inky Mini execution.

Width 2, model W=[[1,2],[-1,1]], bias=[0,1], source=[5,-3].
Median of three wall-time repetitions, one local CPU environment,
compact JSON proof length (decimal bytes):

| Depth | Prove ms | Verify ms | Proof bytes |
|---:|---:|---:|---:|
| 1 | 26.49 | 38.98 | 31843 |
| 2 | 50.81 | 76.75 | 63401 |
| 4 | 98.64 | 148.57 | 126517 |
| 8 | 200.99 | 302.61 | 252749 |
| 16 | 397.47 | 605.68 | 505213 |
| 32 | 798.37 | 1200.93 | 1010141 |

These are measurements of the development proof system, not Nova.
No peak-RSS or binary compression measurements were collected.

The fixed-shape Nova export uses the same clamp relation with 19-bit
slacks and fixed model coefficients; state = (x,h,counter,context limbs).
The x and h range constraints, step counter increment and immutable
input/context carry are explicit. Context is split into two 128-bit limbs;
the external verifier must pin the exact initial state and final state,
model-derived circuit, number of steps, and authorized context. Do not
accept caller-supplied circuits as authoritative. The export includes
the witness in clear for development. At widths 1 and 2 the exported
circuits have 107 and 205 R1CS constraints respectively, excluding Nova
folding/verification overhead. The R1CS witness checker passed positive
fixtures and rejected an altered output. It checks an exported instance
only; it is not a proof verifier.

Native integration: use a reviewed, pinned nova-snark version, preferably
a Pallas/Vesta or BN254/Grumpkin Pedersen/IPA configuration with no
test-utils setup. Map wire zero to CS::one, state_in to allocated
recursive state, allocate remaining witness wires, enforce every sparse
A*B=C constraint, return state_out. Build constraints independently from
the authorized model; never trust the export's witness/constraint file
as a production verification key. Run RecursiveSNARK::prove_step,
verify the returned final state against the authorized expected output,
then compress and independently verify the compressed proof.
The source API references are https://github.com/microsoft/Nova and
https://github.com/microsoft/Nova/blob/main/examples/minroot.rs .
Actual Rust integration and benchmark are blocked here.

## 03: Lean ZK statement

zk_fabric.lean_statement binds theorem/type identity, environment,
checker, proof term digest, dependency closure, sorted permitted axiom
policy and context. The external_verify adapter accepts only a relying
party-pinned executable and verification key. It requires an exact
request-digest acknowledgment, not just a successful exit.
The external executable must implement the actual cryptographic verifier
for this complete statement. The mock subprocess test intentionally does
not implement zkPi or Lean. It tests transport, pin mismatch, wrong relation,
empty proof, changed executable and sorryAx policy rejection only.

To use zkPi (https://eprint.iacr.org/2024/267), map the complete pinned
Lean environment/theorem/proof to its supported fragment. Retain unsupported
cases in the denominator. Proving that a Lean checker accepted a certificate
inherits the checker and axiom assumptions. A proof-term digest is not a
hiding commitment for low-entropy certificates. For a zkVM lane pin the
actual checker program/image, exact environment dependency manifest and
verified axiom collection. Existing LeanKernelChecker remains the integration
seam. Metamath cross-check is an independent additional predicate.

The subprocess boundary is a development adapter. Production must use
immutable executable/runtime/dependency images, strict JSON schemas,
resource limits, sandboxed process execution, authenticated authorization
and atomic nonce-consumption. Hashing an executable path before executing it
does not itself prevent TOCTOU or untrusted shared-library substitution.

## 04: verified graph, pending recursive aggregation

Each node proves its local arithmetic against the digest of the exact
authorized plan, job and ordered parent receipt digests. The graph verifier
checks every local proof, exact node set and order, recomputes each public
join, and binds output. A single terminal hash authenticates the ordered
reachable receipts under the supplied manifest; a hash alone is not a proof.
The verifier requires all graph receipts and has linear verification cost.
Recursive proof-carrying data for heterogeneous joins remains pending.

Width 1, two recurrent steps per node, previous-two-parent joins,
single illustrative run per size, all outputs checked against zk_replay:

| Nodes | Prove ms | Verify ms | Receipt JSON bytes |
|---:|---:|---:|---:|
| 1 | 30.34 | 37.80 | 31002 |
| 8 | 210.83 | 306.41 | 248016 |
| 32 | 819.19 | 1221.44 | 992086 |

A separate aggregation circuit must verify exact parent proof/key identities,
graph completeness and the ordered joins. Binding only a terminal digest
does not suffice. Use a persistent run ledger for one-time semantics.

## 05: proof-aware routing baseline

fit_route_cost fits a nonnegative integer affine envelope on supplied
training observations. choose_route selects the least predicted cost among
routes whose proof class matches the requested class and whose cost fits
the budget, otherwise abstains. It cannot replace a live resource watchdog
and an envelope over training points is not a bound on future runtime.

Fit total prover + verifier microseconds using only depths 1,2,4,8 above:
a=2880 us, b=62590 us/pass. Held-out development depths:

| Depth | Predicted us | Observed us |
|---:|---:|---:|
| 16 | 1004320 | 1003150 |
| 32 | 2005760 | 1999300 |

This is a descriptive within-machine extrapolation on two depths, not a
statistical generalization claim. No trained TRM/BPD controller has run.
Use the route oracle as the label generator in the native gym; lock
train/cal/test graph roots and compare equal compute against heuristic,
TRM and BPD policies. Charge failed proofs and retries; apply proof-class
admission separately from learned predictions.

## 06: Lucid-compatible metadata

lucid_registration builds documented registry metadata with base64 key,
proof-system, curve, version, public input count and setup hash. It rejects
unlisted Nova and sigma formats, mismatched keys, and missing Groth16 setup
pins. This is a schema/format gate, not live registration or a key parser.
The current Lucid API registry lists groth16, plonk, fflonk and stark:
https://docs.lucidcomputing.ai/api-reference
https://docs.lucidcomputing.ai/concepts/zk-proofs

A real adapter must verify relation/key identity, exact public input
encoding and the backend proof under the registered circuit, then test
live submission with operator authorization. No API calls, credentials
or attestation service were used here.

## 07: corpus certification

certify_corpus verifies the entire graph first, requires the exact public
payload of each selected job, an authorized source-root mapping, TRAIN
allowlist, and unique job identity. The manifest binds the plan, terminal
receipt, admission policy, count and ordered length-bound Merkle root.
The same verified graph can certify a selected corpus without re-running
the recurrent computation; verification still checks every proof.
Source-root authorization is external and the corpus manifest is public.
It does not establish source truth, hidden-data lineage, exhaustive
completeness of all eligible examples, or training/checkpoint provenance.
An independent verifier needs the authorized plan, roots, items and proof
receipts. Private inputs, hiding commitments, membership proofs, optimizer
execution and cross-run policy enforcement are separate next gates.

## Deployment contract

OpenSilico owns plan authorization, policy, task and source admission;
Tesseract owns scheduling, transport, immutable job identity and resource
admission; DataGoblin owns proposed examples and proof-aware routing.
No verified deployment of OpenSilico, Tesseract or Lucid is claimed.
Keep proof classes explicit: transparent replay, experimental sigma,
native recursive proof (only when independently verified), hardware
attestation, and trusted checker. Never silently downgrade an admission
requirement or label a development fixture as a production proof.
