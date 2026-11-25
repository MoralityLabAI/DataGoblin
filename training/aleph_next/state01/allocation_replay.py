"""Measure checked primitive attempts on the train split, then on frozen held families."""

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from allocation_tasks import (
    ACTIONS,
    declaration,
    digest,
    equal,
    features,
    generate,
    historical,
    inside,
    invalid_declaration,
    pair_key,
    solve,
)
from state15_kernel import compile_query, proof_axioms

HEADER = "set_option autoImplicit false\n"

HERE = Path(__file__).resolve().parent
DEFAULT = Path("../aleph-allocation/final")
LEAN = Path("../aleph-state15/lean-4.19.0-windows/bin/lean.exe")
MEDIAN = 1.24528


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pin():
    names = (
        "ALLOCATION_PROTOCOL.md",
        "allocation_tasks.py",
        "allocation_replay.py",
        "state11.py",
        "state12.py",
        "state13.py",
    )
    import ast

    kernel = ast.parse((HERE / "state15_kernel.py").read_text())
    relevant = [
        ast.dump(node)
        for node in kernel.body
        if getattr(node, "name", "") in {"compile_query", "_child_cpu", "proof_axioms"} or isinstance(node, ast.Assign)
    ]
    return digest({"files": {name: file_hash(HERE / name) for name in names}, "kernel": relevant})


def accepted(code, log, name):
    valid, _ = proof_axioms(log, [name])
    return code == 0 and valid and "sorry" not in log.lower()


@contextlib.contextmanager
def exclusive(path):
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 60
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Kernel:
    def __init__(self, lean, root):
        self.lean = lean.resolve()
        self.version = subprocess.check_output([str(self.lean), "--version"], encoding="utf-8").strip()
        if "4.19.0" not in self.version or "6caaee842e94" not in self.version:
            raise ValueError("Lean toolchain mismatch: " + self.version)
        self.binary = file_hash(self.lean)
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def run(self, source, name, negative=False):
        if re.search(r"\b(sorry|admit|unsafe|axiom)\b", source):
            raise ValueError("untrusted declaration token")
        content = (HEADER + source + "\n#print axioms " + name + "\n").encode()
        key = hashlib.sha256(content).hexdigest()
        path = self.root / (key + ".lean")
        with exclusive(path.with_suffix(".lock")):
            return self._locked(path, content, key, name, negative)

    def _locked(self, path, content, key, name, negative):
        log_file = path.with_suffix(".log")
        rec = path.with_suffix(".json")
        if rec.exists():
            result = read(rec)
            if result["binary"] != self.binary or file_hash(path) != key or file_hash(log_file) != result["log"]:
                raise ValueError("cached Lean receipt changed")
            return result
        path.write_bytes(content)
        cost, log = compile_query(self.lean, path, timeout=12)
        code = 124 if cost["timed_out"] else cost["exit_code"]
        elapsed = cost["wall_seconds"]
        ok = accepted(code, log, name)
        rejected = negative and code not in (0, 124) and "type mismatch" in log.lower()
        result = dict(
            source=key,
            log=file_hash(log_file),
            binary=self.binary,
            seconds=elapsed,
            cpu_seconds=cost["cpu_seconds"],
            exit=code,
            accepted=ok,
            intended_negative=negative,
            rejected=rejected,
            bytes=len(content),
            version=self.version,
        )
        save(rec, result)
        return result


def action(task, kind, kernel, anchor=None, source=False):
    leaves = task.source if source else task.target
    p, work, exp, solver = solve(leaves, kind, task.source, task.wrappers, anchor)
    measured = None
    render = 0.0
    if p is not None:
        at = time.perf_counter()
        text = declaration("candidate", leaves, p)
        render = time.perf_counter() - at
        measured = kernel.run(text, "candidate")
    return dict(
        ok=bool(measured and measured["accepted"]),
        solver_seconds=solver + render,
        work=work,
        expansions=exp,
        kernel=measured,
        steps=sum(map(len, p)) if p is not None else 0,
    ), p


def anchor_measure(task, kernel):
    attempts = []
    for choice in ("frontiers", "normal"):
        item, path = action(task, choice, kernel, source=True)
        attempts.append(item)
        if item["ok"]:
            return attempts, path
    return attempts, None


def measure(args):
    root = args.out
    split = args.split
    kernel = Kernel(args.lean, root / "singleton")
    bound = pin()
    frozen = None
    if split == "held":
        freeze = root / "FROZEN.json"
        if not freeze.exists():
            raise ValueError("held panel requires frozen trained policies")
        frozen = read(freeze)
        if frozen["primitive_pin"] != bound or frozen["train_sha256"] != file_hash(root / "train.json"):
            raise ValueError("frozen input contract changed")
    archive = historical()
    training = generate("train", banned=archive)
    prior_keys = archive | {pair_key(t.target) for t in training}
    tasks = training if split == "train" else generate("held", banned=prior_keys)
    if split == "held" and {t.lineage for t in tasks} & {t.lineage for t in training}:
        raise ValueError("lineage crosses split")
    rows = []
    exports = []
    for start in range(0, len(tasks), 3):
        if not args.collect and (start // 3) % args.shards != args.shard:
            continue
        lineage = tasks[start : start + 3]
        output = root / "lineages" / split / (lineage[0].lineage + ".json")
        if output.exists():
            cached = read(output)
            if cached["pin"] != bound or cached["binary"] != kernel.binary:
                raise ValueError("lineage contract changed")
            rows += cached["rows"]
            exports += cached.get("export", [])
            continue
        if args.collect:
            raise ValueError("missing measured lineage: " + lineage[0].lineage)
        anchors, source_path = anchor_measure(lineage[0], kernel)
        witness = None
        context = None
        if source_path is not None:
            paths = [
                [(inside(w, z), i, (1, *p), rev) for z, i, p, rev in path]
                for w, path in zip(lineage[0].wrappers, source_path, strict=True)
            ]
            witness = declaration("witness", lineage[0].target, paths)
            context = kernel.run(witness, "witness")
        group, training_export = [], []
        for task in lineage:
            start_gate = time.perf_counter()
            gate = equal(task.target)
            gate_s = time.perf_counter() - start_gate
            if gate != (task.intervention != "invalid"):
                raise ValueError("goal generator failed independent denotation")
            ft = time.perf_counter()
            observed = features(task)
            feat_s = time.perf_counter() - ft
            row = dict(
                key=pair_key(task.target),
                lineage=task.lineage,
                family=task.family,
                intervention=task.intervention,
                components=len(task.target),
                public=task.public(),
                gate=gate,
                gate_s=gate_s,
                features=observed,
                features_s=feat_s,
                source=anchors,
                actions={},
                stale=None,
            )
            if witness is not None and task.intervention != "unchanged":
                stale = kernel.run(witness + "\n" + invalid_declaration(task.target), "stale", negative=True)
                row["stale"] = dict(context=context, invalid=stale)
            exported = False
            if gate:
                for kind in ACTIONS:
                    result, path = action(task, kind, kernel, anchor=source_path)
                    row["actions"][kind] = result
                    if split == "train" and result["ok"] and not exported:
                        training_export.append(
                            dict(
                                split="train",
                                family=task.family,
                                lineage=task.lineage,
                                public={"leaves": task.public()},
                                target={"paths": path, "lean_source_sha256": result["kernel"]["source"]},
                                checker="allocation_tasks.valid + Lean 4.19.0 singleton",
                                checker_sha256=file_hash(HERE / "allocation_tasks.py"),
                                source_sha256=result["kernel"]["source"],
                                rights="locally generated synthetic obligations and certificates",
                            )
                        )
                        exported = True
            group.append(row)
        save(output, dict(pin=bound, binary=kernel.binary, rows=group, export=training_export))
        rows += group
        exports += training_export
        print(split, start // 3 + 1, "/", len(tasks) // 3, flush=True)
    if not args.collect and args.shards > 1:
        return
    save(
        root / (split + ".json"),
        dict(
            schema="aleph-allocation-attempts-v1",
            split=split,
            pin=bound,
            lean=kernel.version,
            lean_binary=kernel.binary,
            historical_equations=len(archive),
            panel_sha256=digest([r["key"] for r in rows]),
            freeze_sha256=file_hash(root / "FROZEN.json") if frozen else None,
            rows=rows,
        ),
    )
    if split == "train":
        (root / "train-only.jsonl").write_bytes("".join(json.dumps(v, sort_keys=True) + "\n" for v in exports).encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("train", "held"), required=True)
    p.add_argument("--out", type=Path, default=DEFAULT)
    p.add_argument("--lean", type=Path, default=LEAN)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--collect", action="store_true")
    args = p.parse_args()
    if not 0 <= args.shard < args.shards <= 4:
        p.error("invalid shard")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch

    torch.set_num_threads(1)
    measure(args)


if __name__ == "__main__":
    main()
