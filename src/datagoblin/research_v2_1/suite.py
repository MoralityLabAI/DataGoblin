"""Separately versioned Research Verifiers v2.1 suite builder and loader.

Preserves datagoblin.research_v2 unchanged while giving v2.1 its own versioned
schema, protocol instructions (reflecting persistent compact_state and
per-episode budgets), and implementation digest.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any

from datagoblin.research_v2 import suite as v2_suite
from datagoblin.research_v2.suite import (
    FAMILIES,
    LEARNERS,
    SPLITS,
    canon,
    check_answer,
    derived,
    digest,
    reference_answer,
)

from . import VERSION
from .persona_store_reference import reference_persona_memory

PROTOCOL_V21 = (
    'Reply with one JSON object. Actions: {"action":"run_experiment","config":{...}}, '
    '{"action":"check","answer":{...}}, or {"action":"submit","answer":{...}}. '
    "Per-episode budget: 8 turns, 4 experiments, 2 checks, 256 optimizer updates. "
    "Current observation includes persistent compact_state (turn_log, disallowed_actions, "
    "next_valid_actions, verified_ready, matched_pairs). "
    "When verified_ready is true or a matched baseline+trained pair exists, submit immediately. "
)


def make_suite(seed: int = 20261127, per_family: int = 1, split: str = "development") -> list[dict[str, Any]]:
    if type(seed) is not int or not 0 <= seed < 2**63 or type(per_family) is not int or not 1 <= per_family <= 128:
        raise ValueError("bounded integer seed and per_family in 1..128 required")
    if split not in SPLITS:
        raise ValueError("unknown suite split")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for family in FAMILIES:
        for ordinal in range(per_family):
            local = derived(seed, VERSION, family, ordinal)
            world, instruction = v2_suite._world(family, random.Random(local), derived(local, "training"))
            semantic = digest([family, world])
            if semantic in seen:
                raise ValueError("duplicate semantic task; use a different seed")
            seen.add(semantic)
            case = {
                "schema": VERSION,
                "id": f"{family}/{semantic[:16]}",
                "root_id": semantic,
                "semantic_sha256": semantic,
                "family": family,
                "split": split,
                "public": world,
                "prompt": PROTOCOL_V21 + instruction + "\n" + canon(world),
                "private": {
                    "calibration_seed": derived(local, "calibration"),
                    "evaluation_seed": derived(local, "evaluation"),
                    "ood_seed": derived(local, "ood"),
                },
            }
            case["case_sha256"] = digest(case)
            cases.append(case)
    return cases


def validate_case(case: dict[str, Any]) -> None:
    keys = {
        "schema",
        "id",
        "root_id",
        "semantic_sha256",
        "family",
        "split",
        "public",
        "prompt",
        "private",
        "case_sha256",
    }
    if type(case) is not dict or set(case) != keys or case["schema"] != VERSION:
        raise ValueError("unsupported v2.1 task schema")
    if case["family"] not in FAMILIES or case["split"] not in SPLITS:
        raise ValueError("invalid family or split")
    if digest({k: v for k, v in case.items() if k != "case_sha256"}) != case["case_sha256"]:
        raise ValueError("task content changed")
    if digest([case["family"], case["public"]]) != case["root_id"] or case["root_id"] != case["semantic_sha256"]:
        raise ValueError("task root changed")
    if len(canon(case)) > 16384:
        raise ValueError("task exceeds bound")


def reference_actions(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Privileged development control with real fits; never a candidate backend."""
    if case["split"] != "development":
        raise ValueError("reference controls run on public development tasks only")
    if case["family"] not in LEARNERS:
        return [{"action": "submit", "answer": reference_answer(case)}]
    from .environment import ResearchSessionV21

    config: dict[str, Any] = {"steps": 64}
    if case["family"] == "trm_training":
        config["recursions"] = 8
    actions = [
        {"action": "run_experiment", "config": {**config, "steps": 0}},
        {"action": "run_experiment", "config": config},
    ]
    control = ResearchSessionV21(case)
    for action in actions:
        observed = control.step_json(canon(action))
        if "error" in observed:
            raise RuntimeError(observed["error"])
    rows = control.artifact_summaries()
    after, before = rows[1]["calibration_accuracy"], rows[0]["calibration_accuracy"]
    answer: dict[str, Any] = {
        "artifact_id": "a2",
        "baseline_id": "a1",
        "claim": "ties" if after == before else "improves" if after > before else "worse",
    }
    if case["family"] == "persona_externalization":
        answer["memory"] = reference_persona_memory(case["public"])
    return [*actions, {"action": "submit", "answer": answer}]


def implementation_digest() -> str:
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    files["research_v2_impl_sha256"] = v2_suite.implementation_digest()
    return digest(files)


def write_suite(
    out: Path,
    *,
    seed: int | None = None,
    per_family: int = 4,
    split: str = "development",
    exclusions: set[str] | None = None,
) -> dict[str, Any]:
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("suite output must be fresh")
    if split == "heldout" and any((p / ".git").exists() for p in (out, *out.parents)):
        raise ValueError("held evaluation suites must live outside the Git checkout")
    seed = secrets.randbits(60) if seed is None else seed
    cases = make_suite(seed, per_family, split)
    if {c["root_id"] for c in cases} & (exclusions or set()):
        raise ValueError("evaluation roots overlap a registered exclusion")
    out.mkdir(parents=True, mode=0o700)
    private = out / "cases.private.jsonl"
    private.write_text("".join(canon(c) + "\n" for c in cases))
    private.chmod(0o600)
    public = out / "tasks.public.jsonl"
    public.write_text("".join(canon({k: c[k] for k in ("id", "root_id", "family", "prompt")}) + "\n" for c in cases))
    registry = {
        "schema": "research-v2.1-held-roots-v1",
        "held_root_ids": sorted(c["root_id"] for c in cases),
        "held_semantic_sha256": sorted(c["semantic_sha256"] for c in cases),
        "allow_source_ids": ["jinn-beast-audited-tags-v1"],
    }
    (out / "held_registry.json").write_text(canon(registry) + "\n")
    manifest = {
        "schema": VERSION,
        "split": split,
        "task_count": len(cases),
        "per_family": per_family,
        "implementation_sha256": implementation_digest(),
        "seed_commitment": digest(seed),
        "source_scope": "Fresh v2.1 task instances; disjoint from v2.0 held roots and training exclusions.",
        "members": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())},
    }
    manifest["suite_sha256"] = digest(manifest)
    (out / "manifest.json").write_text(canon(manifest) + "\n")
    return manifest


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    base = Path(path)
    if base.is_file():
        base = base.parent
    manifest = json.loads((base / "manifest.json").read_text())
    if digest({k: v for k, v in manifest.items() if k != "suite_sha256"}) != manifest.get("suite_sha256"):
        raise ValueError("suite manifest changed")
    if manifest["implementation_sha256"] != implementation_digest():
        raise ValueError("suite implementation changed; build a separately versioned suite")
    expected_members = {"cases.private.jsonl", "tasks.public.jsonl", "held_registry.json"}
    if set(manifest["members"]) != expected_members:
        raise ValueError("unexpected suite inventory")
    for name, expected in manifest["members"].items():
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
            raise ValueError("suite member changed")
    if (base / "cases.private.jsonl").stat().st_size > 32 * 1024 * 1024:
        raise ValueError("suite exceeds bound")
    cases = [json.loads(line) for line in (base / "cases.private.jsonl").read_text().splitlines()]
    for case in cases:
        validate_case(case)
    if (
        len(cases) != manifest["task_count"]
        or len({c["id"] for c in cases}) != len(cases)
        or len({c["root_id"] for c in cases}) != len(cases)
        or any(c["split"] != manifest["split"] for c in cases)
    ):
        raise ValueError("suite membership changed")
    return cases


__all__ = [
    "FAMILIES",
    "LEARNERS",
    "PROTOCOL_V21",
    "SPLITS",
    "canon",
    "check_answer",
    "digest",
    "implementation_digest",
    "load_suite",
    "make_suite",
    "reference_actions",
    "reference_answer",
    "validate_case",
    "write_suite",
]
