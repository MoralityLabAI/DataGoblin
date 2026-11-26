# AGN-MEM-01: authoring worlds with declared memory demands

This offline port gives Agonist an executable way to ask whether a proposed
world needs a running statistic or several independent facts remembered for
a later question. It extends the authoring contract described in
[AGN-03](https://github.com/MoralityLabAI/DataGoblin/blob/703380b0c46e68758b664711ae679c221676bbb2/docs/storyworld_continual_learning.md).
AGN-03 remains a separate draft. The files here run on the current default
branch without importing that draft or a story engine.

`Agonist/memory_curriculum.py` provides a trusted finite permit world,
versioned author tickets and exhaustive controller replay. The public
JSON schema lives in `Agonist/schemas/memory_ticket.schema.json`.
Every ticket belongs to the same exposed development generator family and
sets `training_ready=false`. Teacher calls and student-training updates are
zero in this implementation.

## Two authoring tasks

A courier visits the registry and sees `J` binary permit marks, one at a
time. The marks arrive in coordinate order, from most significant bit to
least significant bit. The world uses the same actor and office names in
every instance. The late action menu is always `decline`, `authorize`.

| Task | Rule for the final action | Useful memory |
| --- | --- | --- |
| Parity permit | Authorize exactly when an odd number of marks were one | A two-state parity accumulator |
| Queried permit | After memory commitment, the registry asks for coordinate `q`; authorize when that mark was one | Enough retained distinctions to answer the later coordinate |

For parity, each mark updates a single toggle and two states give perfect
performance at every tested width. In the queried task, a zero-error
controller must distinguish all `2**J` words after the last update. If two
different words share a code, some coordinate differs; asking that
coordinate forces the same answer for two different targets. A `J`-bit
register attains the bound.

The two-state compiler evaluates the remaining partial-information cases.
For `J=3`, its exact causal optimum is `17/24`. A one-bit window attains
`16/24`, and a complete register attains `24/24`. These are finite control
values, recorded by the replay command rather than committed as a generated
run archive. They measure the world and controller interface. A learner
uplift claim needs a later experiment with admitted training views and
separate source families.

## Author ticket fields

| Field | Required content |
| --- | --- |
| `source_family` | Generator identity, development split and original-source rights |
| `mechanism` | Query or parity rule and a width from 1 through 8 |
| `memory_bounds` | Number of retained states and `ceil(log2(K))` bit ceiling |
| `query_timing` | All observations finish and the finite memory code commits before query disclosure |
| `allowed_side_information` | Public phase, public rule, fixed controller tables and late query index |
| `retrieval_cost` | Retrieval disabled, budget zero, abstract per-bit access charge reserved as one |
| `actor_view` | Current observation at update time; committed code and query at action time |
| `teacher_budget` | Zero calls and zero tokens for this reference adapter |
| `max_presentations_per_root` | Four, shared with any other curriculum views of that root |

`author_ticket(J, K, task)` constructs the ticket. The Python validator also
checks the relationships between width, state capacity and bit ceiling.
The JSON schema rejects extra fields and fixes custody and timing metadata.
The source family and width are curriculum metadata; the actor packet
carries no root identity or instance name that could encode a hidden word.

The tables used by a compiled controller count as static public program
storage. The charged memory is the retained code between updates. A future
model comparison should report parameter storage and context cost alongside
this state bound. Enabling retrieval requires an explicit adapter revision
with matched retrieval budgets and charged observations.

## Public observations and private records

The trusted `PermitEpisode` retains the full word. Its acquisition API
returns one packet with the current mark and public phase. The actor update
reads its previous code, that mark and phase, then returns a code in the
declared range. Earlier observations remain outside this interface.

After all `J` updates, `commit_memory(code)` accepts exactly one code.
`reveal_query(q)` becomes available only after that commitment. Its packet
contains the committed code, rule and query index, together with constant
names, menu and resource fields. The decoder receives this packet alone.

`score(action)` emits an evaluator-owned receipt with the private word and
expected answer. That receipt remains outside the acting interface and is
never fed back during an episode. An exporter for a later training study
must keep these private records separate from actor input views and retain
the existing shared exposure cap.

Tests enumerate every four-bit word and query to verify the constant
surface fields. They assert exact public packet keys, check query-independent
observation prefixes and reject premature queries, repeated commitments and
out-of-range codes. The independent story transition computes the target
from its private word and agrees with the vendored MIC-18 evaluator across
all replayed words and late queries.

This trusted adapter executes authored Python mechanics. A future teacher
can propose narrative surfaces and typed mechanisms for independent review.
Generated code requires its own admission and verification step before it
can take this evaluator's role. The current renderer uses fixed text and
does not call GPTStoryworld's native engine or Bonsai/Inky.

## Reproduce and extend

From the repository root, using Python 3.10 or newer:

```sh
python -m unittest discover -s tests/unit -p 'test_agonist_memory.py' -v
python -m Agonist.memory_curriculum --check
python -m pytest
ruff check .
```

The focused checks use the standard library. The existing development
environment supplies pytest and Ruff for the repository-wide checks.
`Agonist/MEMORY_PROVENANCE.json` pins the shared core and source contracts;
`Agonist/LICENSE.MIC18` retains its Apache-2.0 license. A narrow Ruff exception
preserves the exact vendored bytes, following the repository's existing
provenance practice.

For an AGN-03 author-quality study, add independent fictional mechanism
families under this ticket contract, then compare their verified memory
demand, shortcut sensitivity and valid worlds per measured teacher cost.
Student transfer and retention should be evaluated after separate corpus
admission. Every width and word in this port is part of one already exposed
development family.
