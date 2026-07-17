# ADR 0001: Public and private research state

Status: accepted

DataGoblin exposes only frozen, serializable public research state (`a_t`) to orchestration and audit code. A backend may maintain private recurrence or scratch state (`h_t`), but it cannot place hidden reasoning into durable records and DataGoblin never requires access to it.

All cross-boundary proposals are typed objects. Durable records contain concise observable rationales, evidence references, critiques, and decisions—not hidden chain-of-thought.

