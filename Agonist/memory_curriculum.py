"""AGN-MEM-01: offline author tickets and a finite actor-local permit world.

The trusted engine owns the word and outcome labels. A compiled actor carries
one finite code between observations; the coordinate query arrives afterward.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

TICKET_SCHEMA = "AGN-MEM-01/ticket/v1"
FAMILY = "agn-memory-permit/v1"
NAMES = {"actor": "Courier", "office": "Registry"}
MENU = ["decline", "authorize"]
RESOURCES = {"retrieval_remaining": 0}


def author_ticket(width: int, capacity: int, task: str = "query") -> dict:
    ticket = {
        "schema": TICKET_SCHEMA,
        "source_family": {"id": FAMILY, "split": "dev", "rights": "project_original"},
        "mechanism": {"task": task, "width": width},
        "memory_bounds": {"states": capacity, "bits_ceiling": (capacity - 1).bit_length()},
        "query_timing": "after_all_observations_and_memory_commit",
        "allowed_side_information": ["public phase", "public rule", "fixed controller tables", "late query index"],
        "retrieval_cost": {"enabled": False, "per_bit": 1, "budget": 0},
        "actor_view": "current observation during update; retained code and late query during action",
        "teacher_budget": {"calls": 0, "tokens": 0},
        "training_ready": False,
        "max_presentations_per_root": 4,
    }
    validate_ticket(ticket)
    return ticket


def validate_ticket(ticket: dict) -> None:
    required = {
        "schema", "source_family", "mechanism", "memory_bounds", "query_timing",
        "allowed_side_information", "retrieval_cost", "actor_view", "teacher_budget",
        "training_ready", "max_presentations_per_root",
    }
    if set(ticket) != required or ticket["schema"] != TICKET_SCHEMA:
        raise ValueError("ticket shape")
    mechanism, memory = ticket["mechanism"], ticket["memory_bounds"]
    if set(mechanism) != {"task", "width"} or mechanism["task"] not in ("query", "parity"):
        raise ValueError("mechanism")
    width = mechanism["width"]
    if type(width) is not int or not 1 <= width <= 8:
        raise ValueError("width budget")
    if set(memory) != {"states", "bits_ceiling"}:
        raise ValueError("memory fields")
    capacity = memory["states"]
    if type(capacity) is not int or not 1 <= capacity <= 1 << width:
        raise ValueError("capacity budget")
    if type(memory["bits_ceiling"]) is not int or memory["bits_ceiling"] != (capacity - 1).bit_length():
        raise ValueError("bit accounting")
    if ticket["source_family"] != {"id": FAMILY, "split": "dev", "rights": "project_original"}:
        raise ValueError("source-family custody")
    if ticket["query_timing"] != "after_all_observations_and_memory_commit":
        raise ValueError("query timing")
    if ticket["allowed_side_information"] != [
        "public phase", "public rule", "fixed controller tables", "late query index",
    ]:
        raise ValueError("side information")
    if ticket["retrieval_cost"] != {"enabled": False, "per_bit": 1, "budget": 0}:
        raise ValueError("retrieval budget")
    if ticket["teacher_budget"] != {"calls": 0, "tokens": 0} or ticket["training_ready"] is not False:
        raise ValueError("offline development scope")
    if ticket["max_presentations_per_root"] != 4:
        raise ValueError("shared root exposure cap")
    if ticket["actor_view"] != "current observation during update; retained code and late query during action":
        raise ValueError("actor interface")


def _surface(stage: str, phase: int) -> dict:
    return {
        "schema": "AGN-MEM-01/actor/v1", "stage": stage, "phase": phase,
        "names": dict(NAMES), "resources": dict(RESOURCES),
        "menu": list(MENU) if stage == "query" else [],
    }


class PermitEpisode:
    """Trusted evaluator-owned episode. Never pass this object to an actor."""

    def __init__(self, ticket: dict, word: int):
        validate_ticket(ticket)
        self.ticket = copy.deepcopy(ticket)
        self.width = ticket["mechanism"]["width"]
        if type(word) is not int or not 0 <= word < 1 << self.width:
            raise ValueError("private word")
        self._word = word
        self._phase = 0
        self._code = None
        self._query = None
        self._scored = False

    def next_observation(self) -> dict:
        if self._phase >= self.width or self._code is not None:
            raise ValueError("observation stream exhausted")
        packet = _surface("observe", self._phase)
        packet["observation"] = {
            "kind": "permit_mark",
            "bit": (self._word >> (self.width - 1 - self._phase)) & 1,
        }
        self._phase += 1
        return packet

    def commit_memory(self, code: int) -> None:
        if self._phase != self.width or self._code is not None:
            raise ValueError("memory commits exactly once after all observations")
        if type(code) is not int or not 0 <= code < self.ticket["memory_bounds"]["states"]:
            raise ValueError("retained code exceeds declared memory")
        self._code = code

    def reveal_query(self, query: int) -> dict:
        if self._code is None or self._query is not None:
            raise ValueError("late query requires a committed memory code")
        task = self.ticket["mechanism"]["task"]
        limit = self.width if task == "query" else 1
        if type(query) is not int or not 0 <= query < limit:
            raise ValueError("query index")
        self._query = query
        packet = _surface("query", self.width)
        packet.update({"retained_code": self._code, "query_index": query, "rule": task})
        return packet

    def score(self, action: int) -> dict:
        if self._query is None or self._scored:
            raise ValueError("score exactly one action after the query")
        if type(action) is not int or action not in (0, 1):
            raise ValueError("action")
        self._scored = True
        task = self.ticket["mechanism"]["task"]
        expected = ((self._word >> (self.width - 1 - self._query)) & 1
                    if task == "query" else self._word.bit_count() % 2)
        return {"schema": "AGN-MEM-01/private_receipt/v1", "private_word": self._word,
                "expected_action": expected, "chosen_action": action,
                "success": int(expected == action), "query_index": self._query,
                "training_ready": False}


def validate_controller(controller: dict) -> None:
    if controller.get("schema") != "MIC-18/controller/v1":
        raise ValueError("controller schema")
    ticket = author_ticket(controller["width"], controller["capacity"], controller["task"])
    width, capacity = ticket["mechanism"]["width"], ticket["memory_bounds"]["states"]
    if controller["initial_state"] != 0 or type(controller["initial_state"]) is not int:
        raise ValueError("initial state")
    if len(controller["transitions"]) != width or len(controller["outputs"]) != capacity:
        raise ValueError("controller dimensions")
    for layer in controller["transitions"]:
        if len(layer) != capacity:
            raise ValueError("controller layer")
        for row in layer:
            if len(row) != 2 or any(type(x) is not int or not 0 <= x < capacity for x in row):
                raise ValueError("controller transition")
    columns = width if controller["task"] == "query" else 1
    for row in controller["outputs"]:
        if len(row) != columns or any(type(x) is not int or x not in (0, 1) for x in row):
            raise ValueError("controller output")


def actor_update(controller: dict, code: int, observation: dict) -> int:
    if observation["stage"] != "observe":
        raise ValueError("update packet")
    return controller["transitions"][observation["phase"]][code][observation["observation"]["bit"]]


def actor_action(controller: dict, packet: dict) -> int:
    if packet["stage"] != "query":
        raise ValueError("decision packet")
    return controller["outputs"][packet["retained_code"]][packet["query_index"]]


def replay(controller: dict) -> dict:
    """Independent world replay of every uniform word and every late query."""
    validate_controller(controller)
    ticket = author_ticket(controller["width"], controller["capacity"], controller["task"])
    successes = total = 0
    queries = range(controller["width"] if controller["task"] == "query" else 1)
    for word in range(1 << controller["width"]):
        for query in queries:
            engine = PermitEpisode(ticket, word)
            code = controller["initial_state"]
            for _ in range(controller["width"]):
                code = actor_update(controller, code, engine.next_observation())
            engine.commit_memory(code)
            packet = engine.reveal_query(query)
            receipt = engine.score(actor_action(controller, packet))
            successes += receipt["success"]
            total += 1
    return {"successes": successes, "decisions": total, "reward": str(Fraction(successes, total)),
            "capacity": controller["capacity"], "family_id": FAMILY, "training_ready": False,
            "retrieval_operations": 0, "teacher_calls": 0}


def checks() -> dict:
    source = Path(__file__).with_name("mic18_core.py")
    expected = "9fa386bd52c6925e41e7151e163b34df01f553eb6a214fe4737960dfd2109abc"
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("vendored MIC-18 source digest changed")
    from Agonist.mic18_core import compile_exact, evaluate, make_parity, make_register, make_window

    results = {}
    for width in (2, 3, 4):
        controllers = [make_register(width), make_parity(width), make_window(width, 1),
                       compile_exact(width, 2)["controller"]]
        for index, controller in enumerate(controllers):
            metric = replay(controller)
            shared = evaluate(controller)
            if (metric["successes"], metric["decisions"]) != (shared["score"], shared["total"]):
                raise AssertionError("trusted story replay differs from shared finite oracle")
            results[f"J{width}_{index}_{controller['task']}"] = metric
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(checks(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
