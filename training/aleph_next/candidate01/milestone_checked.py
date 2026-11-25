"""Supply three mathematical training lanes up to their 10M milestone quotas, and replay every admission."""

import argparse
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from statistics import median

import curation as c
import snapshot
from production import TOKENIZER_SHA256, digest, load_tokenizer, read

SCHEMA = "datagoblin.aleph.checked-milestone.v1"
BASE_SHA = "cb2495da7078bdb5320fb348a007744cb2473711cedca6d34341d6622a04cb1d"
SOURCE_ADMISSIONS = {
    "verified": "94728f557b565e7a800e8f36082e5a0c56e34f7e060734e553c9aa3c13dbd51f",
    "tournament-train": "57d2e5eb149f4f618f84ed12979b4e0c2ff84c63f42f60d64207cccf5c3a8eed",
}
QUOTAS = {"procedural": 1150550, "verified": 1441344, "failure_directed": 259339}
SKILLS = {"procedural": ("state", "operator"), "verified": ("composition",), "failure_directed": ("repair",)}
SEED = 2026112601
FILES = (
    "manifest.json",
    "registry.sqlite3",
    "tokenizer.json",
    "train.jsonl",
    "records.jsonl",
    "receipts.jsonl",
    "inputs.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code():
    paths = (Path(__file__), Path(snapshot.__file__), c.HERE.parents[1] / "aleph_rlt_placement/packing.py")
    return c.code_digests() | {
        p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths
    }


def prior(base):
    index = base / "snapshot-final/index.json"
    require(digest(index) == BASE_SHA, "unrecognized starter supply")
    frozen = read(index)
    recipe = c.HERE / "RECIPE.json"
    require(digest(recipe) == frozen["recipe_sha256"], "recipe changed")
    mapping = read(recipe)["stream"]["subtasks"]
    require(all(mapping[s] == k for k, v in SKILLS.items() for s in v), "subtask mapping changed")
    prior_roots, families, registry_hashes = set(), set(), {}
    for lane, entry in frozen["lanes"].items():
        path = (base / entry["path"]).resolve()
        require(path.is_relative_to(base.resolve()) and digest(path) == entry["manifest_sha256"], "starter lane change")
        roots, groups = snapshot.membership(path, "train")
        require(prior_roots.isdisjoint(roots), "duplicate starter root across projections")
        registry_hashes[lane] = read(path)["registry"]["sha256"]
        prior_roots.update(roots)
        families.update(groups)
    prompts, semantics, records = set(), set(), {}
    for name, expected in SOURCE_ADMISSIONS.items():
        folder = base / name
        admission = folder / "admissions.json"
        require(digest(admission) == expected, "starter admissions changed")
        path = folder / "records.jsonl"
        records[name] = digest(path)
        require(records[name] == read(admission)["members"]["records.jsonl"], "starter records changed")
        with path.open("rb") as handle:
            for line in handle:
                row = json.loads(line)
                root = row["semantic_key"]
                require(
                    row["split"] == "train" and root in prior_roots and root not in semantics, "starter semantic root"
                )
                semantics.add(root)
                prompts.add(c.tasks.digest(row["prompt"]))
    require(len(prior_roots) == 7806 and len(semantics) == 6955, "starter coverage")
    reserved = {name for v in c.tasks.RESERVED.values() for name in v}
    for role in frozen["private_math"]["roles"].values():
        reserved.update(role["families"])
    train_families = {f for f in families if f.partition("/")[0] in {s for v in SKILLS.values() for s in v}}
    require(not {f.partition("/")[2] for f in train_families} & reserved, "reserved mathematical family")
    inputs = {
        "schema": SCHEMA,
        "starter_index_sha256": BASE_SHA,
        "recipe_sha256": frozen["recipe_sha256"],
        "starter_registry_sha256": registry_hashes,
        "starter_admissions_sha256": SOURCE_ADMISSIONS,
        "starter_records_sha256": records,
        "prior_training_roots": len(prior_roots),
        "prior_checked_semantic_keys": len(semantics),
        "prior_checked_prompt_hashes": len(prompts),
        "approved_train_families": sorted(train_families),
        "excluded_families": sorted(reserved),
        "code_sha256_lf": code(),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "seed": SEED,
        "rights": c.INTERNAL,
        "held_payload_files_opened": 0,
    }
    return prior_roots, prompts, inputs


def generated(skill, ordinal):
    rng = random.Random(f"milestone10m/{SEED}/{skill}/{ordinal}")
    return getattr(c.tasks, skill)(rng, ordinal)


def checked_row(row, ordinal, families, hashes):
    root, difficulty = c.inspect_candidate(row, {}, {})
    skill = row["skill"]
    prompt, target = c.render(row)
    family = skill + "/" + row["family"]
    require(family in families, "family outside the starter train grammar")
    public = {
        "split": "train",
        "skill": skill,
        "family": family,
        "semantic_key": root,
        "lineage": root,
        "prompt": prompt,
        "target": target,
        "delimiter": c.tasks.DELIMITER,
        "origin": row["origin"],
        "rights": c.INTERNAL,
        "difficulty": difficulty,
        "generator_seed": SEED,
        "generator_ordinal": ordinal,
        "source_generator_sha256_lf": hashes["curation_tasks.py"],
    }
    receipt = {
        "schema": SCHEMA,
        "split": "train",
        "root": root,
        "accepted": True,
        "check": "public-independent-execution",
        "verifier_sha256_lf": hashes["curation_check.py"],
        "record_sha256": c.tasks.digest(public),
        "tokenizer_sha256": TOKENIZER_SHA256,
        "witness": difficulty,
    }
    return public, receipt


def packed_row(public, receipt, tokenizer):
    ids, mask, start = c.tokens(tokenizer, public["prompt"], public["target"])
    return {
        "root": public["semantic_key"],
        "family": public["family"],
        "split": "train",
        "presentation": 1,
        "lane": public["skill"],
        "objective": "checked_response",
        "token_ids": ids,
        "target_mask": mask,
        "response_start": start,
        "receipt_sha256": c.tasks.digest(receipt),
    }


def admissions(lane, prior_state, tokenizer):
    prior_roots, prior_prompts, source = prior_state
    seen_roots, seen_prompts = set(), set()
    families = set(source["approved_train_families"])
    rejects = Counter()
    skills = SKILLS[lane]
    positions = 0
    for ix in range(200000):
        ordinal, slot = divmod(ix, len(skills))
        skill = skills[slot]
        row = generated(skill, ordinal)
        if row is None:
            rejects["no_eligible_reachable_fault"] += 1
            continue
        pub, receipt = checked_row(row, ordinal, families, source["code_sha256_lf"])
        root, prompt_hash = pub["semantic_key"], c.tasks.digest(pub["prompt"])
        if root in prior_roots or root in seen_roots:
            rejects["prior_semantic_duplicate" if root in prior_roots else "stream_semantic_duplicate"] += 1
            continue
        if prompt_hash in prior_prompts or prompt_hash in seen_prompts:
            rejects["prior_prompt_duplicate" if prompt_hash in prior_prompts else "stream_prompt_duplicate"] += 1
            continue
        packed = packed_row(pub, receipt, tokenizer)
        seen_roots.add(root)
        seen_prompts.add(prompt_hash)
        positions += len(packed["token_ids"])
        yield pub, receipt, packed, ix + 1, rejects
        if positions >= QUOTAS[lane]:
            return
    raise RuntimeError("candidate budget exhausted before reaching quota")


def tally(summary, pub, packed):
    skill = pub["skill"]
    for bucket, key, count in (
        ("rows_by_skill", skill, 1),
        ("rows_by_family", pub["family"], 1),
        ("positions_by_skill", skill, len(packed["token_ids"])),
        ("raw_targets_by_skill", skill, sum(packed["target_mask"])),
    ):
        summary[bucket][key] += count
    for name, value in pub["difficulty"].items():
        if value is not None:
            summary["difficulty"][skill + "/" + name].append(value)


def totals():
    return {k: Counter() for k in ("rows_by_skill", "rows_by_family", "positions_by_skill", "raw_targets_by_skill")} | {
        "difficulty": defaultdict(list)
    }


def finalize(totals):
    return {k: dict(v) for k, v in totals.items() if k != "difficulty"} | {
        "difficulty": {
            k: {"n": len(v), "min": min(v), "median": median(v), "max": max(v)} for k, v in totals["difficulty"].items()
        }
    }


def build_lane(lane, folder, tokenizer_path, tokenizer, prior_state):
    c.dump(folder / "inputs.json", prior_state[2])
    shutil.copyfile(tokenizer_path, folder / "tokenizer.json")
    registry = []
    counts = totals()
    with ExitStack() as stack:
        files = [
            stack.enter_context((folder / f).open("wb")) for f in ("records.jsonl", "receipts.jsonl", "train.jsonl")
        ]
        for admission in admissions(lane, prior_state, tokenizer):
            pub, receipt, item, draws, rejects = admission
            for stream, data in zip(files, (pub, receipt, item), strict=True):
                c.append(stream, data)
            registry.append((item["root"], item["family"], c.tasks.digest(item), item["receipt_sha256"]))
            tally(counts, pub, item)
    positions = sum(counts["positions_by_skill"].values())
    c.bind_registry(folder / "registry.sqlite3", registry, digest(folder / "train.jsonl"))
    c.dump(
        folder / "manifest.json",
        {
            "schema": "aleph.packed-stream.v1",
            "context": 2048,
            "vocab_size": 49152,
            "pad_id": 0,
            "tokenizer": c.member(folder / "tokenizer.json"),
            "registry": c.member(folder / "registry.sqlite3"),
            "shards": [{"split": "train", "records": len(registry), **c.member(folder / "train.jsonl")}],
        },
    )
    audit = {
        "schema": SCHEMA,
        "lane": lane,
        "rows": len(registry),
        "target_positions": QUOTAS[lane],
        "positions": positions,
        "whole_record_overhang": positions - QUOTAS[lane],
        "final_record_tokens": len(item["token_ids"]),
        "candidate_draws": draws,
        "rejections": dict(rejects),
        "source": c.INTERNAL,
        "selection": "First deterministic prefix of whole verified unique records meeting the position quota.",
        "members": {f: digest(folder / f) for f in FILES},
        **finalize(counts),
        "packed": snapshot.scan(folder / "manifest.json", "train", positions),
    }
    c.dump(folder / "admissions.json", audit)


def verify(lane, folder, tokenizer, prior_state):
    prior_roots, prior_prompts, src = prior_state
    audit = read(folder / "admissions.json")
    require(audit["members"] == {f: digest(folder / f) for f in FILES}, "artifact digest mismatch")
    require(read(folder / "inputs.json") == src, "original inputs or source code changed")
    require(audit["schema"] == SCHEMA and audit["lane"] == lane and audit["source"] == c.INTERNAL, "admission scope")
    roots, forms = snapshot.membership(folder / "manifest.json", "train")
    require(
        roots.isdisjoint(prior_roots) and forms <= set(src["approved_train_families"]), "starter or family collision"
    )
    # Re-enumerating the seeded candidate sequence also verifies that every exclusion applies and that no later
    # problem replaced an earlier eligible one. The checker inside checked_row independently solves each public task.
    counts = totals()
    seen, seen_prompts = set(), set()
    with ExitStack() as stack:
        files = [
            stack.enter_context((folder / f).open("rb")) for f in ("records.jsonl", "receipts.jsonl", "train.jsonl")
        ]
        for *lines, expected in zip(*files, admissions(lane, prior_state, tokenizer), strict=True):
            pub, receipt, packed = (json.loads(line) for line in lines)
            expected_pub, expected_receipt, expected_pack, draws, rejects = expected
            require(
                (pub, receipt, packed) == (expected_pub, expected_receipt, expected_pack),
                "seed, proof or token labels changed",
            )
            root, ph = pub["semantic_key"], c.tasks.digest(pub["prompt"])
            require(
                root in roots and root not in seen and ph not in seen_prompts and ph not in prior_prompts,
                "duplicate or missing root",
            )
            seen.add(root)
            seen_prompts.add(ph)
            tally(counts, pub, packed)
    positions = sum(counts["positions_by_skill"].values())
    require(all(audit[k] == v for k, v in finalize(counts).items()), "summaries differ")
    require(len(seen) == audit["rows"] and seen == roots and positions == audit["positions"], "row or position count")
    require(draws == audit["candidate_draws"] and dict(rejects) == audit["rejections"], "rejection audit")
    last = len(packed["token_ids"])
    require(
        audit["target_positions"] == QUOTAS[lane] and positions - last < QUOTAS[lane] <= positions,
        "whole-record prefix",
    )
    require(
        audit["whole_record_overhang"] == positions - QUOTAS[lane] and audit["final_record_tokens"] == last, "overhang"
    )
    ledger = snapshot.scan(folder / "manifest.json", "train", positions)
    require(audit["packed"] == ledger and ledger["started_views"] == len(seen), "native registry or mask")
    return {
        "schema": SCHEMA,
        "lane": lane,
        "rows": len(seen),
        "positions": positions,
        "packed_targets": ledger["targets"],
        "prior_root_collisions": 0,
        "prior_checked_prompt_collisions": 0,
        "generated_root_collisions_admitted": 0,
        "independently_recomputed_responses": len(seen),
        "candidate_draws": draws,
        "manifest_sha256": digest(folder / "manifest.json"),
        "admissions_sha256": digest(folder / "admissions.json"),
        "inputs_sha256": digest(folder / "inputs.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Original candidate01 data root")
    parser.add_argument("--out", type=Path, required=True, help="Parent of the additional production lane directories")
    parser.add_argument("--lane", action="append", choices=QUOTAS)
    parser.add_argument(
        "--verify", action="store_true", help="Detached replay of stored sources and all candidate draws"
    )
    args = parser.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", TOKENIZERS_PARALLELISM="false", RAYON_NUM_THREADS="1")
    base = args.base.resolve()
    out = args.out.resolve()
    roots = prior(base)
    tok_path = base / "prose/train/tokenizer.json"
    tokenizer = load_tokenizer(tok_path)
    if not args.verify:
        out.mkdir(parents=True, exist_ok=True)
    for lane in args.lane or QUOTAS:
        folder = out / lane
        if args.verify or folder.exists():
            result = verify(lane, folder, tokenizer, roots)
            require(read(folder / "verified-replay.json") == result, "stored replay differs")
        else:
            with tempfile.TemporaryDirectory(dir=out, prefix=lane + "-") as tmp:
                stage = Path(tmp) / "build"
                stage.mkdir()
                build_lane(lane, stage, tok_path, tokenizer, roots)
                result = verify(lane, stage, tokenizer, roots)
                c.dump(stage / "verified-replay.json", result)
                stage.rename(folder)
        print(c.tasks.canon(result), flush=True)


if __name__ == "__main__":
    main()
