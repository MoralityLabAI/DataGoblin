# Builder protocol

Builders are untrusted implementers. A `BuilderTicket` contains one component purpose, required public inputs, declared outputs, interface contract, invariants, property tests, dependencies, allowed and forbidden filesystem roots, permissions, finite resources, acceptance criteria, expected artifacts, and evaluator-separation requirements.

The default `DryRunBuilder` emits manifests only. `TemplateBuilder` is trusted repository code for `causal_switches_v1`. The optional command builder uses `shell=False`, an argv list, an isolated working directory, a timeout, a receipt-output cap, and an environment allowlist. It rejects symlinks and resolved paths outside the build root and denies network in v0.

Tickets never contain private evaluation seeds, expected answers, hidden results, or evaluator source. The builder cannot write the ledger or evaluator directories. A build receipt hashes outputs and reports process/network use, but it does not certify scientific results.

