# Architecture

DataGoblin separates proposal authority from evidence authority. Backends consume only recursively frozen public
`ResearchState` records. A backend may have private recurrence, but its only output is a schema-valid proposal. The
membrane records before/after state, claimed and verified evidence, disagreement, disposition, and any bound verification
receipt.

The durable lifecycle is `DRAFT -> CRITIQUED -> VALIDATED -> TICKETED -> BUILT -> EXECUTED -> EVALUATED -> PROMOTED|REJECTED`.
Non-rejection transitions must be adjacent and use a typed authority. Models cannot mark execution, evaluation, or
promotion. Promotion requires an environment-owned or independent-harness receipt bound to the exact prior state and
proposal. Evaluated records are immutable except for a receipt-backed terminal transition or append-only annotations.
Verifier identities and the evidence types they may certify are configured by the environment owner; an otherwise valid
receipt from an untrusted identity fails closed.

The vertical slice uses five boundaries:

1. Domain records are frozen dataclasses with explicit JSON conversion.
2. The constitutional validator fails closed and emits a structured decision whose receipt binds the constitution,
   recipe, hypotheses, and tickets by SHA-256.
3. Builder tickets disclose only public contracts and assign finite resources and filesystem roots.
4. The trusted toy environment owns mechanics and labels; a separate harness verifies them and holds evaluation seeds.
5. SQLite stores record envelopes while artifact bytes are addressed by SHA-256.

`CommandGoblinBackend` and `CommandBuilderBackend` are opt-in local-process seams. They use argv lists, `shell=False`, timeouts, output caps, isolated working directories, and allowlisted environments. They are not strong sandboxes.
