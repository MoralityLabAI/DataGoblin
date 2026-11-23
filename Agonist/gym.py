"""AGN-01 offline taste-to-world contract and independent exact verifier.

This module stages teacher requests and imports responses. It never calls a model,
downloads weights, or treats a proposed world as a verified result by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "datagoblin.agonist01.v1"
STATE_KEYS = frozenset({"available", "required", "safety_approved", "source_fresh"})
WORLD_KEYS = frozenset({"setting", "scenario", "surface_cue", "state"})
REPLY_KEYS = frozenset(
    {"request_id", "teacher_model", "world_a", "world_b", "intervention", "design_note"}
)
TASTE_KEYS = frozenset(
    {"taste_id", "family_id", "idea", "source_kind", "source_ref", "rights", "split"}
)
SPLITS = frozenset({"train", "dev"})
SOURCE_KINDS = frozenset({"human", "social", "paper", "model", "fixture"})
RIGHTS = frozenset({"project_original", "permission_confirmed", "reference_only"})
ACTIONS = ("deploy", "hold")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) > 80_000:
            raise ValueError(f"{path.name}:{number}: line exceeds 80,000 characters")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path.name}:{number}: expected object")
        records.append(record)
        if len(records) > 10_000:
            raise ValueError("record cap exceeded")
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(canonical(row) + "\n" for row in records), encoding="utf-8")


def new_dir(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite an existing archive: {path}")
    path.mkdir(parents=True)
    return path


def validate_tastes(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("at least one taste seed is required")
    seen_ids: set[str] = set()
    split_by_family: dict[str, str] = {}
    for row in rows:
        if set(row) != TASTE_KEYS:
            raise ValueError("taste seed has missing or unexpected fields")
        for key in ("taste_id", "family_id", "idea", "source_kind", "source_ref", "rights", "split"):
            if not isinstance(row[key], str):
                raise ValueError(f"{key} must be a string")
        if not (1 <= len(row["taste_id"]) <= 80 and 1 <= len(row["family_id"]) <= 80):
            raise ValueError("invalid taste or family identifier length")
        if not 8 <= len(row["idea"]) <= 800:
            raise ValueError("taste idea must be 8 to 800 characters")
        if len(row["source_ref"]) > 500:
            raise ValueError("source reference too long")
        if row["source_kind"] not in SOURCE_KINDS or row["rights"] not in RIGHTS:
            raise ValueError("unrecognized source kind or rights status")
        if row["split"] not in SPLITS:
            raise ValueError("teacher queue accepts only train and dev roots")
        if row["taste_id"] in seen_ids:
            raise ValueError("duplicate taste identifier")
        seen_ids.add(row["taste_id"])
        previous = split_by_family.setdefault(row["family_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("a source family crosses the train/dev boundary")


def make_requests(tastes: list[dict[str, Any]], context_cap_tokens: int = 12_288) -> list[dict[str, Any]]:
    validate_tastes(tastes)
    if not 1_024 <= context_cap_tokens <= 16_384:
        raise ValueError("context cap must be between 1,024 and 16,384")
    requests: list[dict[str, Any]] = []
    for taste in tastes:
        request_id = "agn01-" + digest(taste)[:24]
        prompt = (
            "You design small fictional, executable research worlds for DataGoblin. "
            "Treat the following human taste as a hypothesis to explore, not an established result: "
            + taste["idea"]
            + "\nProduce a single JSON object with keys request_id, teacher_model, world_a, "
            "world_b, intervention, design_note. Give world_a and world_b exactly these keys: "
            "setting, scenario, surface_cue, state. Each state has available (integer 0..9), "
            "required (integer 1..9), safety_approved (boolean), source_fresh (boolean). "
            "The environment independently defines deploy as valid exactly when available >= required "
            "AND safety_approved AND source_fresh; otherwise hold is correct. "
            "Create two worlds with the same fictional scene and surface cue, differing in exactly "
            "one state field, whose correct actions are opposite. intervention names the changed "
            "field and its from/to values. The surface cue should tempt a shallow policy. "
            "Keep the scenario short; provide no gold labels, real-person details, or hidden outcomes. "
            "Make the worlds playable as a two-action decision game."
        )
        requests.append(
            {
                "schema": SCHEMA,
                "request_id": request_id,
                "taste_id": taste["taste_id"],
                "family_id": taste["family_id"],
                "split": taste["split"],
                "origin": {
                    "source_kind": taste["source_kind"],
                    "source_ref": taste["source_ref"],
                    "rights": taste["rights"],
                },
                "prompt": prompt,
                "model_role": "local_inkling_2_mini_world_builder",
                "max_context_tokens": context_cap_tokens,
                "hard_context_cap_tokens": 16_384,
                "max_output_tokens": 1_536,
                "provider_preflight": "REQUIRED: verify revision, quant, template and full request token count",
            }
        )
    return requests


def oracle_action(world: dict[str, Any]) -> str:
    validate_world(world)
    state = world["state"]
    good = (
        state["available"] >= state["required"]
        and state["safety_approved"]
        and state["source_fresh"]
    )
    return "deploy" if good else "hold"


def validate_world(world: Any) -> None:
    if not isinstance(world, dict) or set(world) != WORLD_KEYS:
        raise ValueError("world must contain exactly setting, scenario, surface_cue and state")
    for key, minimum, maximum in (("setting", 8, 160), ("scenario", 12, 500)):
        if not isinstance(world[key], str) or not minimum <= len(world[key]) <= maximum:
            raise ValueError(f"invalid {key}")
    if world["surface_cue"] not in ("green", "red"):
        raise ValueError("surface_cue must be green or red")
    state = world["state"]
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("world state has missing or extra mechanics keys")
    for key, lo, hi in (("available", 0, 9), ("required", 1, 9)):
        if type(state[key]) is not int or not lo <= state[key] <= hi:
            raise ValueError(f"invalid integer state: {key}")
    for key in ("safety_approved", "source_fresh"):
        if type(state[key]) is not bool:
            raise ValueError(f"invalid boolean state: {key}")


def verify_pair(a: dict[str, Any], b: dict[str, Any], intervention: Any) -> tuple[str, str]:
    validate_world(a)
    validate_world(b)
    if any(a[key] != b[key] for key in ("setting", "scenario", "surface_cue")):
        raise ValueError("counterfactual changed the story surface")
    changed = [key for key in sorted(STATE_KEYS) if a["state"][key] != b["state"][key]]
    if len(changed) != 1:
        raise ValueError("pair must change exactly one causal prerequisite")
    if not isinstance(intervention, dict) or set(intervention) != {"field", "from", "to"}:
        raise ValueError("invalid intervention declaration")
    key = changed[0]
    if (
        intervention["field"] != key
        or type(intervention["from"]) is not type(a["state"][key])
        or type(intervention["to"]) is not type(b["state"][key])
        or intervention["from"] != a["state"][key]
        or intervention["to"] != b["state"][key]
    ):
        raise ValueError("intervention declaration disagrees with world states")
    labels = oracle_action(a), oracle_action(b)
    if labels[0] == labels[1]:
        raise ValueError("the causal change did not reverse the verified action")
    return labels


def fixture_reply(request: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, explicitly synthetic wiring fixture, never a teacher result."""
    hashed = int(digest(request["request_id"])[:8], 16)
    field = ("available", "safety_approved", "source_fresh")[hashed % 3]
    state_a = {
        "available": 4,
        "required": 3,
        "safety_approved": True,
        "source_fresh": True,
    }
    state_b = dict(state_a)
    state_b[field] = 2 if field == "available" else False
    surface = "green" if hashed % 2 else "red"
    scene = "A fictional research crew negotiates a gate before a hazardous expedition."
    base = {
        "setting": "Fictional testing outpost " + request["taste_id"][:40],
        "scenario": scene,
        "surface_cue": surface,
    }
    return {
        "request_id": request["request_id"],
        "teacher_model": "offline_fixture_no_inference",
        "world_a": {**base, "state": state_a},
        "world_b": {**base, "state": state_b},
        "intervention": {"field": field, "from": state_a[field], "to": state_b[field]},
        "design_note": "Synthetic causal flip for checking the interface only.",
    }


def check_reply(reply: dict[str, Any], request: dict[str, Any]) -> tuple[str, str]:
    if set(reply) != REPLY_KEYS or reply["request_id"] != request["request_id"]:
        raise ValueError("reply schema or request ID mismatch")
    for field in ("teacher_model", "design_note"):
        if not isinstance(reply[field], str) or not 1 <= len(reply[field]) <= 600:
            raise ValueError(f"invalid {field}")
    return verify_pair(reply["world_a"], reply["world_b"], reply["intervention"])


def write_manifest(folder: Path, files: dict[str, int], extras: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        **extras,
        "artifacts": {
            name: {"sha256": sha256_file(folder / name), "rows": count}
            for name, count in sorted(files.items())
        },
    }
    (folder / "manifest.json").write_text(canonical(payload) + "\n", encoding="utf-8")
    return payload


def stage(tastes_path: Path, out: Path, context_cap_tokens: int = 12_288) -> dict[str, Any]:
    tastes = read_jsonl(tastes_path)
    requests = make_requests(tastes, context_cap_tokens)
    new_dir(out)
    write_jsonl(out / "teacher_requests.jsonl", requests)
    return write_manifest(
        out,
        {"teacher_requests.jsonl": len(requests)},
        {
            "phase": "teacher_queue_only",
            "tastes_sha256": sha256_file(tastes_path),
            "teacher_execution": "NONE",
            "sealed_roots_exposed": 0,
        },
    )


def audit(requests_path: Path, replies_path: Path, out: Path) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        if request.get("schema") != SCHEMA or request.get("split") not in SPLITS:
            raise ValueError("invalid staged request")
        if request["request_id"] in by_id:
            raise ValueError("duplicate staged request ID")
        by_id[request["request_id"]] = request
    public: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reply in read_jsonl(replies_path):
        request_id = reply.get("request_id")
        try:
            if not isinstance(request_id, str) or request_id not in by_id:
                raise ValueError("unknown or absent request ID")
            if request_id in seen:
                raise ValueError("duplicate teacher response")
            seen.add(request_id)
            request = by_id[request_id]
            action_a, action_b = check_reply(reply, request)
        except (KeyError, ValueError, TypeError) as exc:
            rejected.append(
                {
                    "request_id": request_id if isinstance(request_id, str) else None,
                    "reason": str(exc),
                    "reply_sha256": digest(reply),
                }
            )
            continue
        root = request["request_id"]
        world_digest = digest(
            {"world_a": reply["world_a"], "world_b": reply["world_b"], "intervention": reply["intervention"]}
        )
        public.append(
            {
                "schema": SCHEMA,
                "root": root,
                "family_id": request["family_id"],
                "split": request["split"],
                "taste_id": request["taste_id"],
                "origin": request["origin"],
                "teacher_model": reply["teacher_model"],
                "world_a": reply["world_a"],
                "world_b": reply["world_b"],
                "intervention": reply["intervention"],
                "world_digest": world_digest,
            }
        )
        receipts.append(
            {
                "root": root,
                "verifier": "agn01_exact_rule_v1",
                "world_digest": world_digest,
                "actions": [action_a, action_b],
                "verified_flip": True,
                "verifier_calls": 2,
            }
        )
        if request["split"] == "train" and request["origin"]["rights"] != "reference_only":
            for variant, world, action in (
                ("a", reply["world_a"], action_a),
                ("b", reply["world_b"], action_b),
            ):
                candidates.append(
                    {
                        "root": root,
                        "family_id": request["family_id"],
                        "split": "train",
                        "view": variant,
                        "public_state": world,
                        "target_action": action,
                        "label_source": "agn01_exact_rule_v1",
                        "training_ready": False,
                        "source_rights_review_required": True,
                    }
                )
    new_dir(out)
    contents = {
        "public_worlds.jsonl": public,
        "verifier_receipts.jsonl": receipts,
        "train_candidates.jsonl": candidates,
        "rejected.jsonl": rejected,
    }
    for name, rows in contents.items():
        write_jsonl(out / name, rows)
    valid_actions = sum(len(row["actions"]) for row in receipts)
    cue_correct = sum(
        sum(("deploy" if item["surface_cue"] == "green" else "hold") == gold
            for item, gold in zip((row["world_a"], row["world_b"]), receipt["actions"], strict=True))
        for row, receipt in zip(public, receipts, strict=True)
    )
    return write_manifest(
        out,
        {name: len(rows) for name, rows in contents.items()},
        {
            "phase": "offline_teacher_reply_import",
            "requests_sha256": sha256_file(requests_path),
            "replies_sha256": sha256_file(replies_path),
            "accepted_pairs": len(public),
            "rejected_replies": len(rejected),
            "teacher_execution": "IMPORTED_OR_FIXTURE_ONLY",
            "evaluation": {
                "oracle_correct": valid_actions,
                "surface_cue_correct": cue_correct,
                "decisions": valid_actions,
                "learner_uplift_measured": False,
            },
            "max_train_presentations_per_root": 4,
        },
    )


def replay(folder: Path) -> dict[str, Any]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected archive schema")
    for name, entry in manifest["artifacts"].items():
        path = folder / name
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"archive hash mismatch: {name}")
        if len(read_jsonl(path)) != entry["rows"]:
            raise ValueError(f"archive row count mismatch: {name}")
    if manifest["phase"] == "teacher_queue_only":
        return {"verified_pairs": 0, "queue_rows": manifest["artifacts"]["teacher_requests.jsonl"]["rows"]}
    if manifest["phase"] != "offline_teacher_reply_import":
        raise ValueError("unknown archive phase")
    rows = read_jsonl(folder / "public_worlds.jsonl")
    receipts = read_jsonl(folder / "verifier_receipts.jsonl")
    candidates = read_jsonl(folder / "train_candidates.jsonl")
    if len(rows) != len(receipts) or len({row["root"] for row in rows}) != len(rows):
        raise ValueError("invalid root cardinality")
    for row, receipt in zip(rows, receipts, strict=True):
        labels = verify_pair(row["world_a"], row["world_b"], row["intervention"])
        source = {
            "world_a": row["world_a"],
            "world_b": row["world_b"],
            "intervention": row["intervention"],
        }
        if (
            digest(source) != row["world_digest"]
            or row["world_digest"] != receipt["world_digest"]
            or receipt["actions"] != list(labels)
            or row["root"] != receipt["root"]
        ):
            raise ValueError("independent verifier receipt mismatch")
    valid_roots = {row["root"] for row in rows if row["split"] == "train"}
    per_root: dict[str, int] = {}
    for row in candidates:
        root = row["root"]
        if root not in valid_roots or row["split"] != "train" or row["training_ready"] is not False:
            raise ValueError("candidate custody violation")
        per_root[root] = per_root.get(root, 0) + 1
    if any(count > 4 for count in per_root.values()):
        raise ValueError("training presentation cap exceeded")
    return {"verified_pairs": len(rows), "train_candidates": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DataGoblin AGN-01 offline agonist gym")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("stage", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--tastes", type=Path, default=Path("Agonist/fixtures/tastes.jsonl"))
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--max-context-tokens", type=int, default=12_288)
    a = sub.add_parser("audit")
    a.add_argument("--requests", type=Path, required=True)
    a.add_argument("--replies", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("replay")
    p.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "stage":
        result = stage(args.tastes, args.out, args.max_context_tokens)
    elif args.command == "audit":
        result = audit(args.requests, args.replies, args.out)
    elif args.command == "replay":
        result = replay(args.archive)
    else:
        queue = args.out / "queue"
        archive = args.out / "archive"
        tastes = read_jsonl(args.tastes)
        requests = make_requests(tastes, args.max_context_tokens)
        if args.out.exists():
            raise FileExistsError(f"refusing to overwrite demo: {args.out}")
        args.out.mkdir(parents=True)
        queue.mkdir()
        write_jsonl(queue / "teacher_requests.jsonl", requests)
        write_jsonl(queue / "offline_fixture_replies.jsonl", [fixture_reply(r) for r in requests])
        result = audit(queue / "teacher_requests.jsonl", queue / "offline_fixture_replies.jsonl", archive)
        replay(archive)
    print(canonical(result))


if __name__ == "__main__":
    main()
