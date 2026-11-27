# ZK-DG-10: authenticated public output journal

Date: 2026-11-27. Native Nova succeeded for the preceding frontier commit
dec7463d. New journal implementation and tests committed; the new native
journal and matched scaling campaign require their own successful CI runs.

## Recovered native execution receipt

At dec7463de6be0b15ecb4058ef0267e2e735c6ddb:
- Python CI run 45967565595: 809 passed, 13 skipped, 54.91 s; Ruff passed.
- Nova CI run 45967565459: Rust 1.100.0, native release build succeeded.
  Three workers, depth cap 2, 480 application constraints per step.
  Nova augmented constraint counts (11858, 10349).
  Parameter setup 2371 ms (prover), 2369 ms (independent verifier).
  Folding 333 ms; compression 5450 ms; compressed proof 10056 bytes.
  Verification 116 ms, excluding setup and key generation.
  Wrong pin, wrong output, trailing bytes, truncated proof and mutated
  proof all rejected. Uploaded artifact step succeeded.
- https://github.com/MoralityLabAI/DataGoblin/actions/runs/45967565595
- https://github.com/MoralityLabAI/DataGoblin/actions/runs/45967565459

This is a real native proof for the preceding public, integer frontier
circuit. It is not evidence for the new journal circuit, actual transformer
inference, Lean proving, or distributed production deployment. Timings
are a single hosted smoke with uncached parameter setup.

## Exact packed journal

For worker j, coordinate i, width w, define p = j*w+i, limb l=floor(p/16),
digit offset d=p mod 16. The output h_i is already constrained by the
frontier circuit to [-127,127], so h_i+127 belongs to [0,254].
Initialize all journal words to zero. Let s_j be the constrained one-hot
worker selector already present in the frontier circuit. For each limb:

    J_next[l] = J[l] + sum_i (h_i+127) * sum_{j: floor((j*w+i)/16)=l}
                                                   s_j * 256^((j*w+i)%16)

This uses w products and one linear equality per limb per step. The
compiler directly references the original constrained selectors and
worker outputs; these are exported as internal metadata, not extra public
inputs. No witness-supplied layout or position is accepted.

Induction from the pinned all-zero initial journal and counter: after t
workers, exactly the first t*w byte positions hold their output digits
and all later positions are zero. The update writes previously unused
positions; each word remains between 0 and 2^128-1, far below either
Pasta modulus. Thus no modular wraparound or carry across digits occurs.
With exactly N steps the terminal words injectively encode all outputs.
The verifier checks final counter, context, sink, canonical integer
bounds, byte digits and padding. No new bit-range gadget for journal
limbs is needed within the inductive circuit because the authorized
initial state and transition force the bounds.

This is a public, lossless packing rather than a hiding commitment.
The compressed proof authenticates these public words when the native
proof verifies. No Merkle hash is computed inside the circuit. An
off-circuit SHA-256 Merkle root can subsequently bind selected records
to the proof-verified journal. Unlike a hash-based journal, this scheme
exposes every packed output; it is limited to small bounded coordinates.
It does not give selective disclosure or privacy.

For N workers, width w, L=ceil(N*w/16) words, frontier size K:

    arity = 3 + K*w + L
    constraints = frontier_rows + L*(w+1)

At N=32, w=2, D=4, L=4, compared with dense arity 67 and 2023
application rows:

| Topology | Frontier K | Journal arity | Journal rows |
|---|---:|---:|---:|
| chain | 1 | 9 | 797 |
| previous two | 2 | 11 | 839 |
| all-to-sink | 31 | 69 | 2057 |

These are exact application constraint counts, not native Nova timings.
For the two-parent graph the application rows drop by 58.53% and the
state arity drops by 83.58% while retaining every output. The dense
scheme can be preferable when the live frontier is nearly the full graph.

An independent BigInt packing check covered 256 configurations across
N=1..32, w=1..2 and four output patterns (minimum, maximum, zero and a
deterministic mixed pattern), checking 4,224 prefixes for exact decoding,
zero padding, 128-bit range and both Pasta moduli. This checks the codec
argument separately; it is not an execution of the Python integration.

## Authorized proof-to-corpus gate

The relying party supplies the authorized plan and a claimed terminal
state. authorized_statement validates structural encoding and synthesizes
one fixed-shape initial step. It constructs the R1CS and initial state
from the authorized plan, not from the worker's statement. It does not
replay the full graph or assume that the claim is correct. The claim
becomes the expected terminal state of the pinned native Nova verifier.

claim_manifest checks the proposed selection against the decoded claim
and the relying party's authorized job/root mapping and TRAIN allowlist.
It rejects duplicate jobs and changed payloads. The selected leaves are
ordered by plan position and bind the plan, journal state and record.
This is not a proof status until verify_and_certify succeeds.

verify_and_certify pins the local verifier executable digest, writes
the locally reconstructed statement and proof into a temporary directory,
checks the exact native success response and emits a receipt binding
statement, proof, verifier and corpus manifest. A runtime with mutable
dependencies or executable path is not secure merely because the
executable is hashed. Production requires an immutable trusted runtime,
resource bounds, atomic nonce consumption, source admission policy and
reviewed proof/key handling. No claims of source truth, full selection
completeness, private corpus membership or verified training.

The new tests/test_zk_journal.py contains 36 cases, including 30 graph
fixtures with both widths, limb boundaries, chain/window/star topology,
both Pasta fields, prefix decoding against the replay oracle, witness
corruption, malformed terminal encodings and fail-closed corpus gates.
The subprocess mock is explicitly a transport fixture, not Nova.

## Native execution gate

The scoped CI now builds Nova, runs the existing frontier smoke and
a three-worker journal proof with corpus admission. It constructs a
structurally valid altered claim changing an earlier worker's output
and the matching selected payload, while preserving the final sink.
Native verification must reject the old proof against that claim.
The native statement arity guard expands from 67 to 71 to allow the
additional journal words.

It also runs a matched 32-worker dense versus frontier+journal
proof and independent verification benchmark on the same fixture
(benchmark-only mode skips the redundant negative tests). Preserve
the public statements, proofs, cargo lock and corpus receipt artifacts.

New hosted results are pending at this writing. The prior native smoke
does not establish correctness of the modified bridge or journal.
The PR remains draft; independent cryptographic review is still needed.

## Next step

For high-dimensional outputs, a reviewed field-friendly hash or
authenticated memory construction can replace exact packing. Compare
a hash/Merkle accumulator, packed journal and full-register circuit
under matched output contracts. Retain the exact packing as the
small-model correctness and cost baseline. An independently verified
Lean checker remains a separate lane.
