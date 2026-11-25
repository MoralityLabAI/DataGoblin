"""Bounded, disjoint Lean replay shards and independent union/axiom/cost auditing."""

import argparse
import json
from collections import Counter
from pathlib import Path

from state15_audit import costs
from state15_balanced import build
from state15_kernel import COMMIT, PIN, compile_query, digest, proof_axioms, toolchain
from state15_profile import declarations
from state15_replay import EXPECTED, HEADER, PINS

THRESHOLD = 12_000


def plan(prepared):
    meta = json.loads((prepared / "PREPARED.json").read_bytes())
    if meta["archive_pins"] != {str(n): files for n, files in PINS.items()}:
        raise ValueError("archive pins changed")
    tasks = []
    for stage, count in EXPECTED.items():
        terms = declarations(prepared / f"STATE{stage}_REPLAY.lean", meta["prepared"][str(stage)]["source_sha256"])
        if len(terms) != count or len({n for n, _ in terms}) != count:
            raise ValueError("theorem inventory changed")
        pending = []

        def flush(batch, stage=stage):
            if not batch:
                return
            names = [n for n, _ in batch]
            source = (
                HEADER + "\n".join(t for _, t in batch) + "\n" + "\n".join("#print axioms " + n for n in names) + "\n"
            )
            tasks.append((f"s{stage}_raw_{names[0]}", stage, "raw_positive", names, names, source, {}))
            batch.clear()

        for name, term in terms:
            if stage in (13, 14) and len(term.encode()) >= THRESHOLD:
                flush(pending)
                source, names, origin = build(name, term)
                tasks.append((f"s{stage}_balanced_{name}", stage, "balanced_positive", [name], names, source, origin))
            else:
                pending.append((name, term))
                if len(pending) == 8:
                    flush(pending)
        flush(pending)
    raw = (prepared / "NEGATIVES.json").read_bytes()
    if digest(raw) != meta["negative_sha256"]:
        raise ValueError("negative inventory changed")
    controls = json.loads(raw)
    if len(controls) != 136:
        raise ValueError("wrong negative control count")
    for i, row in enumerate(controls):
        pre = HEADER + row["context"]
        for kind, source in (("valid_context", pre), ("invalid_transport", pre + "\n" + row["invalid"])):
            tasks.append((f"neg_{i:03d}_{kind}", row["stage"], kind, [], [], source, {"id": row["id"]}))
    return tasks


def allocate(tasks, count):
    if count < 1:
        raise ValueError("invalid shard count")
    lanes = [[] for _ in range(count)]
    loads = [0] * count
    for t in sorted(tasks, key=lambda t: (-len(t[5]), t[0])):
        i = min(range(count), key=loads.__getitem__)
        lanes[i].append(t)
        loads[i] += 2000 + len(t[5].encode())
    return lanes


def outcome(kind, names, receipt, log):
    if receipt["timed_out"]:
        return False, {}
    if kind == "invalid_transport":
        return receipt["exit_code"] != 0 and "type mismatch" in log.lower(), {}
    audited, ax = proof_axioms(log, names)
    return receipt["exit_code"] == 0 and audited, ax


def write_json(path, value):
    data = (json.dumps(value, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"receipt exists: {path}")
    path.write_bytes(data)


def run(prepared, out, index, count, executable, timeout):
    if not (0 <= index < count and 1 <= timeout <= 900):
        raise ValueError("invalid resource allocation")
    chain = toolchain(executable)
    tasks = allocate(plan(prepared), count)[index]
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / f"shard-{index}.json",
        {"index": index, "count": count, "timeout": timeout, "toolchain": chain, "jobs": [t[0] for t in tasks]},
    )
    for key, stage, kind, roots, names, source, origin in tasks:
        path = out / (key + ".lean")
        dest = out / (key + ".json")
        if dest.exists():
            raise FileExistsError(f"existing immutable query receipt: {key}")
        path.write_bytes(source.encode())
        cost, log = compile_query(chain["executable"], path, timeout)
        ok, ax = outcome(kind, names, cost, log)
        write_json(
            dest,
            {
                "key": key,
                "kind": kind,
                "stage": stage,
                "index": index,
                "roots": roots,
                "origin": origin,
                "passed": ok,
                "audited_declarations": len(ax),
                "axioms": sorted({a for vals in ax.values() for a in vals}),
                **cost,
            },
        )
        print(key, ok, round(cost["wall_seconds"], 3), flush=True)


def collect(prepared, out, count):
    lanes = allocate(plan(prepared), count)
    entries = []
    failures = []
    chains = set()
    helper_count = 0
    root_counts = Counter()
    successful_axioms = Counter()
    toolchains = {}
    for idx, shard in enumerate(lanes):
        manifest = json.loads((out / f"shard-{idx}.json").read_bytes())
        if manifest["index"] != idx or manifest["count"] != count or manifest["jobs"] != [t[0] for t in shard]:
            raise ValueError("shard inventory mismatch")
        chain = manifest["toolchain"]
        if f"version {PIN}," not in chain["version"] or COMMIT not in chain["version"]:
            raise ValueError("shard used an unpinned Lean compiler")
        chains.add(chain["version"])
        toolchains[idx] = chain["executable_sha256"]
        for key, stage, kind, roots, names, source, origin in shard:
            receipt = json.loads((out / (key + ".json")).read_bytes())
            log_bytes = (out / (key + ".log")).read_bytes()
            if (
                receipt["key"],
                receipt["stage"],
                receipt["kind"],
                receipt["index"],
                receipt["roots"],
                receipt["origin"],
            ) != (key, stage, kind, idx, roots, origin):
                raise ValueError(f"wrong query identity: {key}")
            if digest(source.encode()) != receipt["source_sha256"] or digest(log_bytes) != receipt["log_sha256"]:
                raise ValueError(f"hash mismatch: {key}")
            if receipt["wall_seconds"] <= 0 or receipt["cpu_seconds"] < 0:
                raise ValueError(f"unmeasured query: {key}")
            ok, ax = outcome(kind, names, receipt, log_bytes.decode("utf-8").replace("\r\n", "\n"))
            union = sorted({a for vals in ax.values() for a in vals})
            if ok != receipt["passed"] or receipt["axioms"] != union or receipt["audited_declarations"] != len(ax):
                raise ValueError(f"invalid gate: {key}")
            if ok:
                root_counts[(stage, kind)] += len(roots) if roots else 1
                helper_count += len(ax) - len(roots)
                for n in roots:
                    successful_axioms[",".join(ax[n]) or "none"] += 1
            else:
                failures.append(
                    {
                        "key": key,
                        "kind": kind,
                        "stage": stage,
                        "timed_out": receipt["timed_out"],
                        "exit_code": receipt["exit_code"],
                        "log_sha256": receipt["log_sha256"],
                    }
                )
            entries.append(receipt)
    if len(chains) != 1 or len(set(toolchains.values())) != 1 or len({k["key"] for k in entries}) != len(entries):
        raise ValueError("mixed toolchain versions or duplicate queries")
    extra = {p.stem for p in out.glob("*.log")} - {r["key"] for r in entries}
    if extra:
        raise ValueError("extra unregistered query logs")
    stages = {
        k: {t: root_counts[k, t] for t in ("raw_positive", "balanced_positive", "valid_context", "invalid_transport")}
        for k in EXPECTED
    }
    positive = sum(s["raw_positive"] + s["balanced_positive"] for s in stages.values())
    controls = sum(s["valid_context"] for s in stages.values())
    rejected = sum(s["invalid_transport"] for s in stages.values())
    manifest_sha = digest((prepared / "PREPARED.json").read_bytes())
    return {
        "schema": "datagoblin.state15.disjoint-kernel-replay.v1",
        "prepared_manifest_sha256": manifest_sha,
        "lean_version": chains.pop(),
        "lean_executable_sha256_by_shard": toolchains,
        "shards": count,
        "positive_declarations_verified": positive,
        "valid_contexts": controls,
        "invalid_transports_rejected": rejected,
        "kernel_verified": positive == sum(EXPECTED.values()) and controls == rejected == 136 and not failures,
        "stage_counts": stages,
        "auxiliary_declarations_audited": helper_count,
        "positive_root_axiom_sets": dict(successful_axioms),
        "failures": failures,
        "costs": {k: costs([r for r in entries if r["kind"] == k]) for k in sorted({r["kind"] for r in entries})},
        "all_charged_queries": len(entries),
        "training_admission": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("run", "collect"))
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--lean", type=Path)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--summary", type=Path)
    args = p.parse_args()
    if args.command == "run":
        run(args.prepared, args.out, args.index, args.count, args.lean, args.timeout)
        return 0
    if not args.summary:
        p.error("--summary required for collection")
    value = collect(args.prepared, args.out, args.count)
    write_json(args.summary, value)
    print(json.dumps(value, indent=2))
    return 0 if value["kernel_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
