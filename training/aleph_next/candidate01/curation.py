"""Admit and replay six train-only mathematical skills using the strict Aleph tokenizer."""

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path
from statistics import median

import curation_check as checked
import curation_tasks as tasks
from production import TOKENIZER_SHA256, digest, load_tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "aleph_rlt_placement"))
from packing import INDEX_SCHEMA, PackedStream  # noqa: E402

SCHEMA = "datagoblin.aleph.candidate01.checked-train.v1"
INTERNAL = "Original project-generated exercise and response. Authorized for internal DataGoblin training."
LICENSED = "Original exercise using Lean core declarations under Apache-2.0. See licenses/Lean-LICENSE."
LIVE = ("curation.py", "curation_tasks.py", "curation_check.py", "production.py")


def code_digests():
    return {name: hashlib.sha256((HERE / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in LIVE}


def raw(x):
    return tasks.canon(x).encode()


def append(handle, x):
    handle.write(raw(x) + b"\n")


def dump(path, x):
    path.write_bytes(raw(x) + b"\n")


def member(path):
    return dict(path=path.name, sha256=digest(path))


def render(row):
    skill = row["skill"]
    prompt = tasks.INSTRUCTIONS[skill] + "\n" + tasks.canon(row["public"])
    target = row["answer"] if skill == "lean" else tasks.canon(row["answer"])
    return prompt, target


def tokens(tokenizer, prompt, target):
    prefix = prompt + tasks.DELIMITER
    transcript = prefix + target + "\n"
    encoded = tokenizer.encode(transcript, add_special_tokens=False)
    if len(encoded.ids) > 2048 or not encoded.ids or any(i < 0 or i >= 49152 for i in encoded.ids):
        raise ValueError("invalid strict context length or vocabulary")
    if tokenizer.decode(encoded.ids, skip_special_tokens=False) != transcript:
        raise ValueError("strict tokenizer failed transcript round trip")
    mask = [lo >= len(prefix) and hi > lo for lo, hi in encoded.offsets]
    if not any(mask) or mask[0] or any(not b for b in mask[mask.index(True) :]):
        raise ValueError("invalid answer token mask")
    return encoded.ids, mask, mask.index(True)


def inspect_candidate(row, native, library):
    skill = row["skill"]
    fam = row["family"]
    origin = row["origin"]
    if (
        skill not in tasks.SKILLS
        or row.get("split", "train") != "train"
        or fam in set(itertools.chain.from_iterable(tasks.RESERVED.values()))
    ):
        raise ValueError("outside admitted train family")
    if skill != "lean":
        if origin != {"kind": "fresh-first-party"}:
            raise ValueError("unexpected public generator")
        ans, family, semantic, difficulty = checked.SOLVE[skill](row["public"])
        if row["answer"] != ans or fam != family:
            raise ValueError("public mathematical verifier disagrees")
        return tasks.digest([skill, semantic]), difficulty
    stmt = row["public"]["statement"]
    proof = row["answer"]
    if (
        checked.FORBIDDEN.search(stmt)
        or checked.FORBIDDEN.search(proof.replace("<;>", ""))
        or not proof.startswith("by\n")
    ):
        raise ValueError("unsupported Lean proof grammar")
    if origin["kind"] == "native-train":
        if native.get(origin["root"]) != row:
            raise ValueError("native train root or action changed")
    elif origin["kind"] == "lean-core":
        if library.get(tasks.digest([stmt, proof])) != row:
            raise ValueError("pinned library-derived row changed")
    elif origin != {"kind": "fresh-first-party"} or fam not in {"predicate-congruence", "ordered-transitivity"}:
        raise ValueError("unrecognized Lean source")
    normalized = checked.lean_template(stmt)
    return tasks.digest([skill, normalized]), dict(theorem_characters=len(stmt), proof_characters=len(proof))


def groups(pack, lean):
    native = {x["origin"]["root"]: x for x in tasks.native_lean(pack)}
    library = {tasks.digest([x["public"]["statement"], x["answer"]]): x for x in tasks.library_lean(lean)}
    return native, library


def exports(count, pack, lean, directory):
    native, library = groups(pack, lean)
    source = itertools.chain(tasks.candidates(count, pack), library.values())
    unique = {}
    counts = Counter()
    rejects = []
    distributions = defaultdict(list)
    for row in source:
        skill = row["skill"]
        counts[skill] += 1
        root, difficulty = inspect_candidate(row, native, library)
        if root in unique:
            rejects.append(
                dict(skill=skill, semantic_key=root, reason="semantic-duplicate", candidate_sha256=tasks.digest(row))
            )
            continue
        unique[root] = (row, difficulty)
    lean_rows = [
        (root, row["public"]["statement"], row["answer"]) for root, (row, _) in unique.items() if row["skill"] == "lean"
    ]
    kernel, negative = checked.kernel_batch(lean_rows, lean, directory / "lean-kernel")
    accepted = []
    for root, (row, difficulty) in sorted(unique.items()):
        if row["skill"] == "lean" and not kernel[root]["passed"]:
            rejects.append(dict(skill="lean", semantic_key=root, reason="kernel-reject", detail=kernel[root]))
            continue
        for k, v in difficulty.items():
            if v is not None:
                distributions[row["skill"] + "/" + k].append(v)
        accepted.append((root, row, difficulty, kernel.get(root)))
    metrics = dict(
        raw_candidates=dict(counts),
        fresh_requested_per_lane=count,
        native_archived_goals=36,
        native_closed_goal_candidates=len(native),
        missing_dfa_state_at_depth_two=count - (counts["repair"]),
        duplicate_or_failed=dict(Counter(r["reason"] for r in rejects)),
        kernel_compiled=len(kernel),
        kernel_accepted=sum(x["passed"] for x in kernel.values()),
        adversarial_kernel_rejection=negative,
        difficulty={k: dict(min=min(v), median=median(v), max=max(v)) for k, v in sorted(distributions.items())},
    )
    return accepted, rejects, metrics


def bind_registry(path, rows, shard_hash):
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript(INDEX_SCHEMA)
        db.executemany("INSERT INTO families VALUES (?, 'train')", [(f,) for f in sorted({r[1] for r in rows})])
        for i, (root, family, rec_sha, receipt_hash) in enumerate(rows):
            db.execute("INSERT INTO roots VALUES (?,?)", (root, family))
            db.execute("INSERT INTO views VALUES (?,1,?,?,?,?)", (root, shard_hash, i, rec_sha, receipt_hash))


def build(pack, lean, tok, out, count=1024):
    if count < 1 or out.exists():
        raise ValueError("Choose a positive count and fresh output directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        directory = Path(tmp).resolve() / "verified"
        directory.mkdir()
        tokenizer = load_tokenizer(tok)
        admitted, rejects, metrics = exports(count, pack, lean, directory)
        (directory / "licenses").mkdir()
        shutil.copyfile(lean.resolve().parents[1] / "LICENSE", directory / "licenses" / "Lean-LICENSE")
        shutil.copyfile(tok, directory / "tokenizer.json")
        rows = []
        counts = Counter()
        positions = Counter()
        targets = Counter()
        families = Counter()
        origins = Counter()
        samples = {}
        code = code_digests()
        with (
            (directory / "train.jsonl").open("wb") as shard,
            (directory / "records.jsonl").open("wb") as public,
            (directory / "receipts.jsonl").open("wb") as receipts,
            (directory / "rejections.jsonl").open("wb") as rejected,
        ):
            for drop in rejects:
                append(rejected, drop)
            for root, row, difficulty, kernel in admitted:
                lane = row["skill"]
                family = lane + "/" + row["family"]
                origin = row["origin"]
                right = LICENSED if origin["kind"] == "lean-core" else INTERNAL
                prompt, target = render(row)
                ids, mask, start = tokens(tokenizer, prompt, target)
                dataset_row = dict(
                    split="train",
                    skill=lane,
                    family=family,
                    semantic_key=root,
                    lineage=origin.get("lineage", root),
                    prompt=prompt,
                    target=target,
                    delimiter=tasks.DELIMITER,
                    origin=origin,
                    rights=right,
                    source_generator_sha256_lf=code["curation_tasks.py"],
                    difficulty=difficulty,
                )
                receipt = dict(
                    schema=SCHEMA,
                    split="train",
                    root=root,
                    record_sha256=tasks.digest(dataset_row),
                    verifier_sha256_lf=code["curation_check.py"],
                    accepted=True,
                    check="lean-kernel" if kernel else "public-independent-execution",
                    witness=kernel if kernel else difficulty,
                    tokenizer_sha256=TOKENIZER_SHA256,
                )
                rh = tasks.digest(receipt)
                item = dict(
                    root=root,
                    family=family,
                    split="train",
                    presentation=1,
                    lane=lane,
                    objective="checked_response",
                    token_ids=ids,
                    target_mask=mask,
                    response_start=start,
                    receipt_sha256=rh,
                )
                rows.append((root, family, tasks.digest(item), rh))
                append(shard, item)
                append(public, dataset_row)
                append(receipts, receipt)
                counts[lane] += 1
                positions[lane] += len(ids)
                targets[lane] += sum(mask)
                families[family] += 1
                origins[origin["kind"]] += 1
                if lane not in samples or len(prompt) < len(samples[lane]["prompt"]):
                    samples[lane] = dict(
                        family=family, prompt=prompt, target=target, semantic_key=root, receipt_sha256=rh
                    )
        if set(counts) != set(tasks.SKILLS):
            raise ValueError("A target skill has no admitted data")
        bind_registry(directory / "registry.sqlite3", rows, digest(directory / "train.jsonl"))
        manifest = dict(
            schema="aleph.packed-stream.v1",
            context=2048,
            vocab_size=49152,
            pad_id=0,
            tokenizer=member(directory / "tokenizer.json"),
            registry=member(directory / "registry.sqlite3"),
            shards=[dict(**member(directory / "train.jsonl"), records=len(rows), split="train")],
        )
        dump(directory / "manifest.json", manifest)
        dump(directory / "examples.json", samples)
        metrics.update(
            schema=SCHEMA,
            implementation_sha256_lf=code,
            seed=tasks.SEED,
            rows=len(rows),
            rows_by_lane=dict(counts),
            positions_by_lane=dict(positions),
            targets_by_lane=dict(targets),
            positions=sum(positions.values()),
            target_positions=sum(targets.values()),
            origins=dict(origins),
            train_families=dict(families),
            excluded_families=tasks.RESERVED,
            native_train_artifacts=tasks.NATIVE,
            members={
                n: digest(directory / n)
                for n in (
                    "manifest.json",
                    "train.jsonl",
                    "records.jsonl",
                    "receipts.jsonl",
                    "rejections.jsonl",
                    "registry.sqlite3",
                    "examples.json",
                    "tokenizer.json",
                    "licenses/Lean-LICENSE",
                )
            },
        )
        with PackedStream(directory / "manifest.json", metrics["members"]["manifest.json"], "train") as stream:
            while stream.next_pack(metrics["positions"]) is not None:
                pass
            if stream.state["processed"] != metrics["positions"] or stream.state["started_views"] != len(rows):
                raise ValueError("packed coverage")
            metrics["packed_supervised_positions"] = stream.state["targets"]
            metrics["packed_by_lane"] = stream.state["targets_by_lane"]
            metrics["padding"] = stream.state["padding"]
            metrics["packs"] = stream.state["packs"]
        dump(directory / "admissions.json", metrics)
        directory.rename(out)
    return metrics


def replay(pack, lean, folder, recompile=False):
    m = json.loads((folder / "admissions.json").read_bytes())
    if m["implementation_sha256_lf"] != code_digests():
        raise ValueError("source drift")
    for name, h in m["members"].items():
        if digest(folder / name) != h:
            raise ValueError("file mismatch: " + name)
    tok = load_tokenizer(folder / "tokenizer.json")
    native, library = groups(pack, lean)
    seen = set()
    counts = Counter()
    kernel_rows = []
    lane_positions = Counter()
    lane_labels = Counter()
    with (
        (folder / "records.jsonl").open("rb") as public,
        (folder / "receipts.jsonl").open("rb") as proofs,
        (folder / "train.jsonl").open("rb") as shards,
    ):
        for p, r, item in zip(public, proofs, shards, strict=True):
            row = json.loads(p)
            receipt = json.loads(r)
            packed = json.loads(item)
            skill = row["skill"]
            family = row["family"]
            prefix = tasks.INSTRUCTIONS[skill] + "\n"
            if not row["prompt"].startswith(prefix) or row["split"] != "train" or row["delimiter"] != tasks.DELIMITER:
                raise ValueError("invalid prompt")
            structured = json.loads(row["prompt"][len(prefix) :])
            response = row["target"] if skill == "lean" else json.loads(row["target"])
            rec = dict(
                skill=skill, family=family.split("/", 1)[1], public=structured, answer=response, origin=row["origin"]
            )
            root, difficulty = inspect_candidate(rec, native, library)
            right = LICENSED if row["origin"]["kind"] == "lean-core" else INTERNAL
            if (
                root in seen
                or root != row["semantic_key"]
                or root != packed["root"]
                or difficulty != row["difficulty"]
                or row["rights"] != right
                or receipt["record_sha256"] != tasks.digest(row)
                or packed["receipt_sha256"] != tasks.digest(receipt)
                or not receipt["accepted"]
            ):
                raise ValueError("semantic, rights or receipt mismatch")
            ids, mask, start = tokens(tok, row["prompt"], row["target"])
            if ids != packed["token_ids"] or mask != packed["target_mask"] or start != packed["response_start"]:
                raise ValueError("tokenization mismatch")
            seen.add(root)
            counts[skill] += 1
            lane_positions[skill] += len(ids)
            lane_labels[skill] += sum(mask)
            if skill == "lean":
                witness = receipt["witness"]
                batch = folder / "lean-kernel" / witness["file"]
                if (
                    digest(batch) != witness["source_sha256"]
                    or digest(Path(str(batch) + ".log")) != witness["compiler_output_sha256"]
                    or not witness["passed"]
                ):
                    raise ValueError("missing successful Lean artifact")
                kernel_rows.append((root, structured["statement"], response))
    if (
        dict(lane_positions) != m["positions_by_lane"]
        or dict(lane_labels) != m["targets_by_lane"]
        or dict(counts) != m["rows_by_lane"]
    ):
        raise ValueError("counts mismatch")
    if recompile:
        with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
            evidence, _ = checked.kernel_batch(kernel_rows, lean, Path(tmp))
            if not all(e["passed"] for e in evidence.values()):
                raise ValueError("independent Lean replay failed")
    return dict(
        schema="aleph.checked-corpus-replay.v1",
        rows=len(seen),
        lanes=dict(counts),
        token_ids=sum(lane_positions.values()),
        response_positions=sum(lane_labels.values()),
        recompiled_lean=len(kernel_rows) if recompile else 0,
        manifest_sha256=m["members"]["manifest.json"],
        admissions_sha256=digest(folder / "admissions.json"),
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("build", "verify"))
    cli.add_argument("--pack", type=Path, required=True)
    cli.add_argument("--lean", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--tokenizer", type=Path)
    cli.add_argument("--count", type=int, default=1024)
    cli.add_argument("--recompile", action="store_true")
    args = cli.parse_args()
    if args.action == "build":
        if args.tokenizer is None:
            cli.error("--tokenizer required for build")
        r = build(args.pack, args.lean, args.tokenizer, args.out.resolve(), args.count)
        print(
            tasks.canon(
                {
                    k: r[k]
                    for k in (
                        "rows",
                        "rows_by_lane",
                        "positions",
                        "target_positions",
                        "packed_supervised_positions",
                        "origins",
                    )
                }
            )
        )
    else:
        r = replay(args.pack, args.lean, args.out.resolve(), args.recompile)
        report = args.out / "verified-replay.json"
        if report.exists() and json.loads(report.read_bytes()) != r:
            raise ValueError("conflicting replay receipt")
        dump(report, r)
        print(tasks.canon(r))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
