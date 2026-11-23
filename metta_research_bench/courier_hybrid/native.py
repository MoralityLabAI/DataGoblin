"""Default-deny native courier verification with immutable transaction state."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

PLACES = 7
ISSUE, COLLECT, DELIVER, ATTEST_FIRST, ATTEST_SECOND, STOP = range(PLACES, PLACES + 6)
ACTION_NAMES = (
    *[f"move:{i}" for i in range(PLACES)],
    "permit",
    "pickup",
    "deliver",
    "attest:first",
    "attest:second",
    "stop",
)
STORYWORLD = "4cfe1713d73b4e508833b668477921adfbe570a6"
HYBRID_RULES = Path(__file__).with_name("hybrid.metta")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_storyworld(checkout: Path) -> tuple[ModuleType, dict[str, str]]:
    checkout = checkout.resolve()

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    if git("rev-parse", "HEAD").decode().strip() != STORYWORLD:
        raise ValueError("Use the pinned Storyworld revision")
    files = ("metta-storyworld/bridge/courier_planner.py", "metta-storyworld/engine/courier.metta")
    hashes = {}
    for filename in files:
        pinned = git("show", f"{STORYWORLD}:{filename}")
        if (checkout / filename).read_bytes().replace(b"\r\n", b"\n") != pinned:
            raise ValueError(f"Storyworld source differs from pinned blob: {filename}")
        hashes[filename] = hashlib.sha256(pinned).hexdigest()
    spec = importlib.util.spec_from_file_location("dg_metta13_pinned_storyworld", checkout / files[0])
    if spec is None or spec.loader is None:
        raise ValueError("Could not import the pinned Storyworld bridge")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, hashes


@dataclass(frozen=True, order=True)
class State:
    place: int
    permit: bool = False
    cargo: int = 0
    evidence: int = 0

    @property
    def complete(self) -> bool:
        return self.cargo == 2 and self.evidence == 3


@dataclass(frozen=True)
class World:
    arcs: tuple[tuple[int, int, bool], ...]
    offices: tuple[int, ...]
    depots: tuple[int, ...]
    desk: int
    entry: int
    sources: tuple[tuple[int, ...], tuple[int, ...]]

    @property
    def initial(self) -> State:
        return State(self.entry)

    @property
    def sha256(self) -> str:
        return digest(["courier-hybrid-world-v1", asdict(self)])

    def facts(self) -> list[str]:
        return [
            *(f"(CourierArc Site{a} Site{b} {'PermitOnly' if gate else 'Open'})" for a, b, gate in self.arcs),
            *(f"(CourierPermitOffice Site{i})" for i in self.offices),
            *(f"(CourierDepot Site{i})" for i in self.depots),
            f"(CourierDesk Site{self.desk})",
            *(
                f"(HybridSource Site{i} {label})"
                for label, places in zip(("First", "Second"), self.sources, strict=True)
                for i in places
            ),
        ]


def successors(world: World, state: State) -> dict[int, State]:
    """Independent public transition projection used for candidates and differential checking."""
    if state.cargo == 2:
        return {}
    place, permit, cargo, evidence = state.place, state.permit, state.cargo, state.evidence
    result = {b: State(b, permit, cargo, evidence) for a, b, gate in world.arcs if a == place and (permit or not gate)}
    if place in world.offices and not permit:
        result[ISSUE] = State(place, True, cargo, evidence)
    if place in world.depots and cargo == 0:
        result[COLLECT] = State(place, permit, 1, evidence)
    if place == world.desk and cargo == 1 and evidence == 3:
        result[DELIVER] = State(place, permit, 2, evidence)
    for index, sites in enumerate(world.sources):
        bit = 1 << index
        if place in sites and not (evidence & bit):
            result[ATTEST_FIRST + index] = State(place, permit, cargo, evidence | bit)
    return result


def action_atom(action: int) -> str:
    if action < PLACES:
        return f"(Move Site{action})"
    return ("IssuePermit", "Collect", "Deliver", "AttestFirst", "AttestSecond")[action - PLACES]


def state_atom(state: State) -> str:
    permit = "HasPermit" if state.permit else "NoPermit"
    cargo = ("Waiting", "Carrying", "Delivered")[state.cargo]
    first = "True" if state.evidence & 1 else "False"
    second = "True" if state.evidence & 2 else "False"
    return f"(HybridState (CourierState Site{state.place} {permit} {cargo}) {first} {second})"


def state_digest(world: World, state: State) -> str:
    return digest(["courier-hybrid-state-v1", world.sha256, asdict(state)])


@dataclass(frozen=True)
class Receipt:
    accepted: bool
    kind: str
    before_sha256: str
    after_sha256: str
    actions: tuple[int, ...]
    query_sha256: str
    native_answers: tuple[str, ...]
    step_sha256: tuple[str, ...]
    reason: str


class NativeCourier:
    """Compose the pinned NativeCourier backend with budgeted evidence and replay queries."""

    def __init__(self, upstream: ModuleType, *, auxiliary_search: bool = False):
        self.source = upstream.RULES.read_text(encoding="utf-8") + "\n" + HYBRID_RULES.read_text(encoding="utf-8")
        if auxiliary_search:
            self.source += "\n" + HYBRID_RULES.with_name("aux_search.metta").read_text(encoding="utf-8")
        self.runtime = upstream.NativeCourier(self.source)
        self.rules_sha256 = hashlib.sha256(self.source.encode()).hexdigest()
        self.calls = 0

    def query(self, world: World, expression: str) -> set[str]:
        self.calls += 1
        return self.runtime.query(world, expression)

    def checked_successors(self, world: World, state: State) -> dict[int, State]:
        expected = successors(world, state)
        answers = self.query(world, f"(HybridSuccessors {state_atom(state)})")
        if answers != {f"(HybridTransition {action_atom(a)} {state_atom(s)})" for a, s in expected.items()}:
            raise RuntimeError("Native courier successors disagree with the public projection")
        return expected

    def session(self, world: World, ceiling: int, horizon: int) -> Session:
        return Session(self, world, ceiling, horizon)


class Session:
    """One episode. Failed transactions preserve state and evict no charged calls."""

    def __init__(self, native: NativeCourier, world: World, ceiling: int, horizon: int):
        if ceiling < 0 or horizon < 0:
            raise ValueError("Courier caps must be nonnegative")
        self.native, self.world = native, world
        self.ceiling, self.horizon = ceiling, horizon
        self.state = world.initial
        self.actions: tuple[int, ...] = ()
        self.receipts: list[Receipt] = []
        self.calls = 0
        self.stopped = False

    @property
    def remaining(self) -> int:
        return self.ceiling - self.calls

    @property
    def moves_left(self) -> int:
        return self.horizon - len(self.actions)

    def _check(
        self, kind: str, actions: tuple[int, ...], expression: str, expected: set[str], states: list[State]
    ) -> Receipt:
        if not self.remaining or self.stopped:
            raise ValueError("Courier verification is unavailable after STOP or budget exhaustion")
        before = state_digest(self.world, self.state)
        identity = digest([self.native.rules_sha256, self.world.facts(), expression])
        self.calls += 1
        answers = self.native.query(self.world, expression)
        if answers != expected:
            raise RuntimeError("Native courier replay disagrees with the public projection")
        accepted = bool(answers)
        after = state_digest(self.world, states[-1]) if accepted else before
        steps = (
            tuple(
                digest([identity, index, action, state_digest(self.world, state)])
                for index, (action, state) in enumerate(zip(actions, states, strict=True))
            )
            if accepted and actions
            else ()
        )
        receipt = Receipt(
            accepted,
            kind,
            before,
            after,
            actions,
            identity,
            tuple(sorted(answers)),
            steps,
            "" if accepted else "native_no_derivation",
        )
        self.receipts.append(receipt)
        if accepted:
            self.state = states[-1]
            self.actions += actions
            self.stopped = kind == "stop"
        return receipt

    def certify(self, actions: tuple[int, ...], *, macro: bool = False) -> Receipt:
        if (
            not actions
            or len(actions) > self.moves_left
            or any(type(a) is not int or not 0 <= a < STOP for a in actions)
        ):
            raise ValueError("Expected a bounded, typed, nonempty courier action sequence")
        if not macro and len(actions) != 1:
            raise ValueError("Primitive verification accepts one action")
        state = self.state
        states = []
        for action in actions:
            after = successors(self.world, state).get(action)
            if after is None:
                break
            states.append(after)
            state = after
        expected = set()
        if len(states) == len(actions):
            proof = f"(HybridEnd {state_atom(state)})"
            for action, after in reversed(list(zip(actions, states, strict=True))):
                proof = f"(HybridProof {action_atom(action)} {state_atom(after)} {proof})"
            expected.add(proof)
        expression = f"(HybridReplay {state_atom(self.state)} ({' '.join(map(action_atom, actions))}))"
        return self._check("macro" if macro else "primitive", actions, expression, expected, states)

    def stop(self) -> Receipt:
        state = state_atom(self.state)
        expected = {f"(HybridDone {state})"} if self.state.complete else set()
        return self._check("stop", (), f"(HybridAcceptStop {state})", expected, [self.state])
