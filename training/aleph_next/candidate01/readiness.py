"""Verify the frozen operator data supply, measure where the exact mixture runs dry, and report blockers."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from production import HERE, digest, read
from snapshot import MixedStream, check, next_allocation

PIN = HERE / "SUPPLY.json"


def milestone_supply(cfg, ix):
    goals = ix["target_by_lane"]
    target = cfg["stream"]["checkpoints"][0]
    batch = cfg["stream"]["batch_nonpadding"]
    counts = dict.fromkeys(goals, 0)
    for start in range(0, target, batch):
        stop = min(start + batch, target)
        while (plan := next_allocation(counts, goals, stop)) is not None:
            lane, positions = plan
            counts[lane] += positions
    return {
        name: {"required": count, "missing": max(0, count - ix["lanes"][name]["positions"])}
        for name, count in counts.items()
    }


def capacity(base, index, expected):
    cfg = read(HERE / "RECIPE.json")
    step = cfg["stream"]["batch_nonpadding"]
    pilot = cfg["stream"]["checkpoints"][0]
    completed = 0
    with ExitStack() as stack:
        stream = MixedStream(base, index, expected, stack)
        while sum(stream.totals.values()) < pilot:
            target = min(completed + step, pilot)
            try:
                while stream.next_pack(target) is not None:
                    pass
            except ValueError as exc:
                if not str(exc).startswith("insufficient admitted positions in "):
                    raise
                return dict(
                    fully_committed_positions=completed,
                    fully_committed_steps=completed // step,
                    emitted_including_incomplete_update=sum(stream.totals.values()),
                    first_stop=str(exc),
                    consumed_by_lane=stream.totals.copy(),
                )
            completed = target
    return dict(fully_committed_positions=completed, first_stop="reached 10M pilot")


def audit(base, pin, measure=False):
    data = read(pin)
    for path, stamp in data["artifacts_sha256"].items():
        member = (base / path).resolve()
        if not member.is_relative_to(base.resolve()) or digest(member) != stamp:
            raise ValueError("protected source changed: " + path)
    if digest(HERE / "RECIPE.json") != data["recipe_sha256"]:
        raise ValueError("frozen optimization or architecture changed")
    a = read(base / "verified/admissions.json")
    sources = {"verified": a}
    if "tournament-train/admissions.json" in data["artifacts_sha256"]:
        extra = read(base / "tournament-train/admissions.json")
        if (
            extra["base_manifest_sha256"] != digest(base / "verified/manifest.json")
            or not extra["no_held_files_opened"]
        ):
            raise ValueError("tournament supplement custody differs")
        sources["tournament-train"] = extra
    for source, admissions in sources.items():
        parent = (base / source).resolve()
        for name, h in admissions["members"].items():
            path = (parent / name).resolve()
            if not path.is_relative_to(parent) or digest(path) != h:
                raise ValueError("admissions member changed: " + name)
    independent = read(base / "verified/verified-replay.json")
    if (
        independent["admissions_sha256"] != digest(base / "verified/admissions.json")
        or independent["recompiled_lean"] != a["kernel_accepted"]
    ):
        raise ValueError("verification replay not bound to this corpus")
    name = data.get("index_path", "snapshot/index.json")
    index = base / name
    ix = read(index)
    facts = check(base, data["artifacts_sha256"][name], "supplement_manifest" in ix)
    facts["first_10m_supply"] = milestone_supply(read(HERE / "RECIPE.json"), ix)
    if measure:
        facts["capacity"] = capacity(base, index, digest(index))
    gaps = {name: cfg["deficit"] for name, cfg in ix["lanes"].items()}
    restart = read(base / "cpu-restart/restart.json")
    code = restart["contract"]["code"]
    reference = HERE.parents[1] / "aleph_rlt_placement"
    if code != {p.name: digest(p) for p in (HERE / "restart.py", reference / "gym.py", reference / "packing.py")}:
        raise ValueError("restart implementation differs from the tested fixture")
    if (
        restart["contract"]["recipe"] != data["recipe_sha256"]
        or restart["contract"]["manifest"] != ix["lanes"]["general"]["manifest_sha256"]
    ):
        raise ValueError("restart fixture used a different recipe or text stream")
    if not restart["exact_cpu_resume"] or restart["geometry"]["trainable_parameters"] != 999352320:
        raise ValueError("missing direct-A probe")
    facts.update(
        training_positions=ix["total_train_positions"],
        missing_positions=sum(gaps.values()),
        gaps=gaps,
        architecture="direct-A",
        cpu_restart=True,
        accelerator_validation=False,
        launch_ready=False,
    )
    return facts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--pin", type=Path, default=PIN)
    p.add_argument("--measure-capacity", action="store_true")
    p.add_argument("--require-ready", action="store_true")
    args = p.parse_args()
    facts = audit(args.base, args.pin, args.measure_capacity)
    print(json.dumps(facts, indent=2, sort_keys=True))
    if args.require_ready and not facts["launch_ready"]:
        raise SystemExit(2)
