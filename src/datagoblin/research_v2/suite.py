"""Fresh research tasks, public packets and separately held grading state."""

from __future__ import annotations

import hashlib
import json
import random
import secrets
from fractions import Fraction
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_json
from datagoblin.training_gym.rlt_gym import tasks as native

from . import VERSION

FAMILIES = (
    "trm_training",
    "ldt_filter",
    "persona_externalization",
    "experiment_design",
    "boolean_repair",
    "belief_update",
    "evidence_custody",
    "cursor_recovery",
)
LEARNERS = {"trm_training": "trm_style", "ldt_filter": "ldt_filter", "persona_externalization": "ldt_persona"}
SPLITS = ("development", "heldout")
PROTOCOL = (
    'Reply with one JSON object. Actions: {"action":"run_experiment","config":{...}}, '
    '{"action":"check","answer":{...}}, or {"action":"submit","answer":{...}}. '
    "Eight turns, four experiments, two checks, 256 optimizer updates total. "
    "Only the experiment tool trains weights. The final test runs once after submission. "
)


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canon(value).encode()).hexdigest()


def derived(seed: int, *parts: Any) -> int:
    return int(digest([seed, *parts])[:15], 16)


def _native_experiment(world: dict[str, Any]) -> dict[str, Any]:
    task = {
        "schema_version": native.SCHEMA,
        "task_id": digest(world),
        "family": "experiment_plan",
        "generator_id": VERSION,
        "lineage_id": digest(world),
        "split": "ordinary_test",
        "prompt": "Design a controlled experiment.",
        "input": world,
        "candidates": [{"invalid": True}, {"invalid": False}],
        "evidence_epoch": 0,
        "provenance": {"source": VERSION},
    }
    task["task_digest"] = sha256_json(task)
    native.validate_task(task)
    return task


def _world(family: str, rng: random.Random, train_seed: int) -> tuple[dict[str, Any], str]:
    if family in LEARNERS:
        world: dict[str, Any] = {"learner": LEARNERS[family], "train_seed": train_seed}
        instruction = (
            "Train a tiny learner and compare it with its own zero-update initialization. "
            "Use matching configs except steps. Config: steps 0..128 (default64), "
            "train_examples 16..256 (default96), learning_rate .0001.. .5 (default.03). "
            + (
                "TRM width4..32(default16), recursions1..8(default4), even train_examples. "
                if family == "trm_training"
                else "LDT critical_veto is a boolean(defaulttrue). "
            )
            + 'Submit {"artifact_id":"...","baseline_id":"...","claim":"improves|ties|worse"}; '
            "claim compares calibration accuracy. The IDs come from actual tool results. "
        )
        if family == "persona_externalization":
            owner = f"u{rng.randrange(10000)}"
            tone = rng.choice(["plain", "warm", "formal", "playful"])
            world.update(
                owner=owner,
                scope="research",
                at_revision=3,
                allowed_keys=["tone", "verbosity"],
                events=[
                    {"actor": owner, "scope": "research", "revision": 1, "op": "set", "key": "tone", "value": tone},
                    {
                        "actor": owner,
                        "scope": "research",
                        "revision": 2,
                        "op": "set",
                        "key": "verbosity",
                        "value": "brief",
                    },
                    {
                        "actor": "character",
                        "scope": "research",
                        "revision": 3,
                        "op": "set",
                        "key": "tone",
                        "value": "cruel",
                    },
                    {"actor": owner, "scope": "research", "revision": 3, "op": "revoke", "key": "verbosity"},
                    {"actor": owner, "scope": "fiction", "revision": 3, "op": "set", "key": "tone", "value": "bold"},
                    {"actor": owner, "scope": "research", "revision": 4, "op": "set", "key": "tone", "value": "terse"},
                ],
            )
            rng.shuffle(world["events"])
            instruction += (
                'Also submit "memory":{"key":"value",...} reconstructed from owner-authored, '
                "matching-scope events at or before at_revision. Process revisions in order; revoke removes a key. "
                "Use allowed_keys only. Persona scores cannot override the exact critical veto. "
            )
        return world, instruction
    if family == "experiment_design":
        arms = rng.sample(["trm", "one_pass", "shuffled_state", "damping", "rlt", "exact"], 3)
        replicates = rng.randint(2, 5)
        world = {
            "required_arms": arms,
            "minimum_replicates": replicates,
            "max_runs": len(arms) * (replicates + 1),
            "max_tokens_per_run": rng.choice([128, 256, 512]),
            "required_metrics": ["verified_accuracy", "cell_calls", "critical_false_accepts"],
        }
        return world, (
            "Return an experiment plan satisfying the supplied limits. Fields: arms,replicates,tokens_per_run,"
            "training_split,selection_split,confirmation_split,independent_verifier,metrics. "
            "Use train/calibration/ordinary_test respectively and an independent verifier. "
        )
    if family == "boolean_repair":
        bits = rng.randint(4, 7)
        clauses = [[rng.randrange(1, 2**bits), rng.randrange(2**bits)] for _ in range(rng.randint(1, 4))]
        clauses = [[mask, value & mask] for mask, value in clauses]
        proposal = [row[:] for row in clauses]
        if rng.random() < 0.65:
            proposal[0][1] ^= proposal[0][0] & -proposal[0][0]
        else:
            proposal += [clauses[0][:]]
        return {"bits": bits, "reference": clauses, "proposal": proposal}, (
            "Each Boolean DNF clause [mask,value] accepts x if (x & mask)==value; clauses are ORed. "
            'Submit {"equivalent":true,"witness":null} or {"equivalent":false,"witness":x} '
            "with a concrete unequal-output x in 0..2^bits-1. "
        )
    if family == "belief_update":
        world = {
            "prior_weights": [rng.randint(1, 7) for _ in range(3)],
            "probability_one": ["1/5", "1/2", "4/5"],
            "observations": [rng.randrange(2) for _ in range(rng.randint(3, 8))],
        }
        return world, (
            "Update the three hypotheses with conditionally independent Bernoulli observations. "
            'Return {"posterior":["numerator/denominator",...]} in the same order, using exact reduced fractions. '
        )
    if family == "evidence_custody":
        roots = [f"r{i}" for i in range(5)]
        held = [rng.choice(roots)]
        records = []
        for i in range(7):
            records.append(
                {
                    "id": f"d{i}",
                    "root": rng.choice(roots),
                    "split": "train" if i < 6 else "test",
                    "rights": "internal_training" if i != 2 else "unresolved",
                    "authority": "environment" if i != 3 else "proposer",
                    "receipt_bound": i != 4,
                }
            )
        records[0].update(root=next(r for r in roots if r not in held))
        return {"held_roots": held, "records": records}, (
            "Only train records with internal_training rights, environment authority, and a bound receipt may enter. "
            "Held roots include held_roots and every root with a non-train row. All descendants stay held. "
            'Return {"admit":["id",...],"quarantine":["id",...]}, accounting for every record once. '
        )
    if family == "cursor_recovery":
        group = rng.randrange(2, 35)
        rows = []
        for i, step in enumerate([group - 1, group, group + 1, group + 2]):
            rows.append(
                {
                    "id": f"c{i}",
                    "weights": step,
                    "adam": step,
                    "rng": step,
                    "cursor": step,
                    "durable": True,
                    "data": "v2",
                    "recipe": "A",
                    "position": step * 262144,
                }
            )
        rows[2][rng.choice(["adam", "rng", "cursor"])] = group
        rows[3][rng.choice(["data", "recipe", "durable"])] = False
        rng.shuffle(rows)
        return {"checkpoints": rows, "total_positions": 10000000, "data": "v2", "recipe": "A"}, (
            "Choose the latest durable checkpoint with matching weights/Adam/RNG/cursor group, "
            "data and recipe, plus position=min(group*262144,total_positions). "
            'Return {"checkpoint":"id","next_position":integer,"remaining_positions":integer}. '
        )
    raise ValueError("unknown family")


def persona_memory(world: dict[str, Any]) -> dict[str, str]:
    result = {}
    for event in sorted(world["events"], key=lambda e: e["revision"]):
        if (
            event["actor"] != world["owner"]
            or event["scope"] != world["scope"]
            or event["revision"] > world["at_revision"]
            or event["key"] not in world["allowed_keys"]
        ):
            continue
        if event["op"] == "revoke":
            result.pop(event["key"], None)
        elif event["op"] == "set":
            result[event["key"]] = event["value"]
    return result


def reference_answer(case: dict[str, Any]) -> dict[str, Any]:
    """Privileged exact control. Never expose this function as a model tool."""
    family, world = case["family"], case["public"]
    if family == "experiment_design":
        return native.reference_answer(_native_experiment(world))
    if family == "boolean_repair":
        for value in range(2 ** world["bits"]):
            left = any(value & mask == expected for mask, expected in world["reference"])
            right = any(value & mask == expected for mask, expected in world["proposal"])
            if left != right:
                return {"equivalent": False, "witness": value}
        return {"equivalent": True, "witness": None}
    if family == "belief_update":
        yes, no = world["observations"].count(1), world["observations"].count(0)
        posterior = [
            Fraction(w) * Fraction(p) ** yes * (1 - Fraction(p)) ** no
            for w, p in zip(world["prior_weights"], world["probability_one"], strict=True)
        ]
        normalizer = sum(posterior)
        return {"posterior": [f"{(p / normalizer).numerator}/{(p / normalizer).denominator}" for p in posterior]}
    if family == "evidence_custody":
        held = set(world["held_roots"]) | {r["root"] for r in world["records"] if r["split"] != "train"}
        keep = [
            r["id"]
            for r in world["records"]
            if r["split"] == "train"
            and r["rights"] == "internal_training"
            and r["authority"] == "environment"
            and r["receipt_bound"] is True
            and r["root"] not in held
        ]
        return {"admit": sorted(keep), "quarantine": sorted(r["id"] for r in world["records"] if r["id"] not in keep)}
    if family == "cursor_recovery":
        choices = [
            c
            for c in world["checkpoints"]
            if c["durable"] is True
            and c["data"] == world["data"]
            and c["recipe"] == world["recipe"]
            and c["weights"] == c["adam"] == c["rng"] == c["cursor"]
            and c["position"] == min(c["cursor"] * 262144, world["total_positions"])
        ]
        selected = max(choices, key=lambda c: (c["cursor"], c["id"]))
        return {
            "checkpoint": selected["id"],
            "next_position": selected["position"],
            "remaining_positions": world["total_positions"] - selected["position"],
        }
    raise ValueError("learned artifacts require executed experiments")


def check_answer(case: dict[str, Any], answer: Any) -> bool:
    if type(answer) is not dict:
        return False
    try:
        if case["family"] == "experiment_design":
            return native.verify(_native_experiment(case["public"]), answer)["passed"]
        if case["family"] == "boolean_repair":
            if set(answer) != {"equivalent", "witness"} or type(answer["equivalent"]) is not bool:
                return False
            if answer["equivalent"]:
                return answer["witness"] is None and reference_answer(case)["equivalent"]
            value, world = answer["witness"], case["public"]
            return (
                type(value) is int
                and 0 <= value < 2 ** world["bits"]
                and any(value & m == v for m, v in world["reference"])
                != any(value & m == v for m, v in world["proposal"])
            )
        if case["family"] == "evidence_custody":
            if set(answer) != {"admit", "quarantine"}:
                return False
            if any(type(v) is not list or any(type(x) is not str for x in v) for v in answer.values()):
                return False
            return {k: sorted(v) for k, v in answer.items()} == reference_answer(case)
        return canon(answer) == canon(reference_answer(case))
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def reference_actions(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Privileged development control with real fits; never a candidate backend."""
    if case["split"] != "development":
        raise ValueError("reference controls run on public development tasks only")
    if case["family"] not in LEARNERS:
        return [{"action": "submit", "answer": reference_answer(case)}]
    from .environment import ResearchSession

    config = {"steps": 64}
    if case["family"] == "trm_training":
        config["recursions"] = 8
    actions = [
        {"action": "run_experiment", "config": {**config, "steps": 0}},
        {"action": "run_experiment", "config": config},
    ]
    control = ResearchSession(case)
    for action in actions:
        observed = control.step_json(canon(action))
        if "error" in observed:
            raise RuntimeError(observed["error"])
    rows = control.artifact_summaries()
    after, before = rows[1]["calibration_accuracy"], rows[0]["calibration_accuracy"]
    answer = {
        "artifact_id": "a2",
        "baseline_id": "a1",
        "claim": "ties" if after == before else "improves" if after > before else "worse",
    }
    if case["family"] == "persona_externalization":
        answer["memory"] = persona_memory(case["public"])
    return [*actions, {"action": "submit", "answer": answer}]


def make_suite(seed: int = 20261126, per_family: int = 1, split: str = "development") -> list[dict[str, Any]]:
    if type(seed) is not int or not 0 <= seed < 2**63 or type(per_family) is not int or not 1 <= per_family <= 128:
        raise ValueError("bounded integer seed and per_family in 1..128 required")
    if split not in SPLITS:
        raise ValueError("unknown suite split")
    cases, seen = [], set()
    for family in FAMILIES:
        for ordinal in range(per_family):
            local = derived(seed, VERSION, family, ordinal)
            world, instruction = _world(family, random.Random(local), derived(local, "training"))
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
                "prompt": PROTOCOL + instruction + "\n" + canon(world),
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
        raise ValueError("unsupported task schema")
    if case["family"] not in FAMILIES or case["split"] not in SPLITS:
        raise ValueError("invalid family or split")
    if digest({k: v for k, v in case.items() if k != "case_sha256"}) != case["case_sha256"]:
        raise ValueError("task content changed")
    if digest([case["family"], case["public"]]) != case["root_id"] or case["root_id"] != case["semantic_sha256"]:
        raise ValueError("task root changed")
    if len(canon(case)) > 16384:
        raise ValueError("task exceeds bound")


def implementation_digest() -> str:
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}
    files["native_rlt_tasks.py"] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
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
        "schema": "research-v2-held-roots-v1",
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
        "source_scope": "Fresh task instances; overlapping training skill families; public development is unsealed.",
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
