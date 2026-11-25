"""Admit only measured training slates and valid proof paths from the external tournament."""

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from curation import PackedStream, append, bind_registry, dump, member, tasks, tokens
from production import digest, load_tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "state01"))
TRAIN = ("nat_distribute", "nat_exchange", "list_reverse_append", "list_map_append")
ACTIONS = ("congruence", "feedback", "frontiers", "normal")
CAP_NS = 4981120000
OLD_LEAN = "f59b0c515f65dbf96526279d77fb43152f76bdd77f5fb94f387b75139fafe68c"
RIGHTS = "Original, locally generated DataGoblin theorem obligations, paths and train-only solver measurements."


def tuples(v):
    return tuple(tuples(a) for a in v) if isinstance(v, list) else v


def unify(pattern, value, bindings):
    if isinstance(pattern, str):
        if pattern.startswith("?"):
            if pattern in bindings:
                return bindings[pattern] == value
            bindings[pattern] = value
            return True
        return pattern == value
    return (
        isinstance(value, tuple)
        and len(pattern) == len(value)
        and all(unify(a, b, bindings) for a, b in zip(pattern, value, strict=True))
    )


def local(before, after, pos):
    for ix in pos:
        if (
            not isinstance(before, tuple)
            or not isinstance(after, tuple)
            or len(before) != len(after)
            or not 1 <= ix < len(before)
        ):
            raise ValueError("invalid child address")
        if before[:ix] != after[:ix] or before[ix + 1 :] != after[ix + 1 :]:
            raise ValueError("context changed")
        before, after = before[ix], after[ix]
    return before, after


def compact_path(domain, first, last, steps, rules):
    start = first
    compact = []
    for after, i, pos, reverse in steps:
        a, b = local(start, after, pos)
        if i not in (range(12) if domain == "nat" else range(12, len(rules))):
            raise ValueError("wrong rule domain")
        lhs, rhs = rules[i][2:]
        if reverse:
            lhs, rhs = rhs, lhs
        env = {}
        if not unify(lhs, a, env) or not unify(rhs, b, env):
            raise ValueError("invalid rewrite or substitution")
        compact.append([i, list(pos), b] if reverse else [i, list(pos)])
        start = after
    if start != last:
        raise ValueError("goal unproved")
    return compact


def canonical_goal(leaf, tournament):
    domain = leaf.domain
    names = ("x", "y", "z") if domain == "nat" else ("xs", "ys", "zs")
    keys = []

    def rename(x, mapping):
        return mapping.get(x, x) if isinstance(x, str) else tuple(rename(i, mapping) for i in x)

    for perm in itertools.permutations(names):
        for maps in [("mf", "mg")] if domain == "nat" else [("mf", "mg"), ("mg", "mf")]:
            substitution = dict(zip(names, perm, strict=True)) | dict(zip(("mf", "mg"), maps, strict=True))
            term = rename(leaf.left, substitution)
            form = tournament.prior.nf(domain, term)
            if isinstance(form, dict):
                form = sorted(form.items())
            keys.append(tasks.canon(form))
    return tasks.digest([domain, min(keys)])


def check_receipt(folder, entry):
    if entry is None:
        return
    src = entry["source"]
    if not re.fullmatch("[0-9a-f]{64}", src) or entry["binary"] != OLD_LEAN:
        raise ValueError("foreign source receipt")
    path = folder / "singleton" / src
    if (
        json.loads(path.with_suffix(".json").read_bytes()) != entry
        or digest(path.with_suffix(".lean")) != src
        or digest(path.with_suffix(".log")) != entry["log"]
    ):
        raise ValueError("bad singleton binding")


def quote(row, folder):
    """Expose retrospective calibration observations as quotes. Include every prerequisite cost."""
    if row["gate"] is not True or set(row["actions"]) != set(ACTIONS):
        raise ValueError("incomplete positive slate")

    def charge(attempt):
        receipt = attempt["kernel"]
        check_receipt(folder, receipt)
        if attempt["ok"] != bool(receipt and receipt["accepted"]):
            raise ValueError("kernel status mismatch")
        return [
            round(attempt["solver_seconds"] * 1e9),
            round(receipt["seconds"] * 1e9) if receipt else 0,
            int(receipt is not None),
        ]

    charges = {}
    for name in ACTIONS:
        attempt = row["actions"][name]
        seq = [*(row["source"] if name == "feedback" else []), attempt]
        charges[name] = dict(accepted=all(s["ok"] for s in seq), stages=[charge(s) for s in seq])
    return dict(allowance_ns=CAP_NS, call_cap=3, gate_ns=round(row["gate_s"] * 1e9), options=charges)


def choose(quote):
    valid = {}
    for name, option in quote["options"].items():
        cost = quote["gate_ns"] + sum(s + k for s, k, _ in option["stages"])
        calls = sum(n for _, _, n in option["stages"])
        if option["accepted"] and calls <= quote["call_cap"] and cost <= quote["allowance_ns"]:
            valid[name] = cost
    if len(valid) < 2:
        return None
    best = min(valid, key=lambda n: (valid[n], n))
    return dict(action=best, elapsed_ns=valid[best])


def verify_new_kernel(chosen, lean, stage):
    binary = lean.resolve()
    version = subprocess.check_output([str(binary), "--version"], encoding="utf8").strip()
    if "4.28.0" not in version:
        raise ValueError("new Lean toolchain mismatch")
    source = ["set_option autoImplicit false"]
    expected = set()
    for i, (_, _, original) in enumerate(chosen):
        name = "admission" + str(i)
        code = re.sub(r"\bcandidate(?=\b|_component_)", name, original)
        if re.search(r"\b(sorry|admit|unsafe|axiom)\b", code):
            raise ValueError("unsupported declaration")
        source.append(code)
        expected.add(name)
    f = stage / "recompiled.lean"
    f.write_bytes(("\n".join(source) + "\n").encode())
    p = subprocess.run(
        [str(binary), "-j", "1", f.name], cwd=stage, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150
    )
    output = p.stdout.decode("utf8", errors="replace").replace("\r\n", "\n")
    (stage / "recompiled.log").write_bytes(p.stdout)
    ax = {
        n: sorted(t.strip() for t in vals.split(",") if t.strip())
        for n, vals in re.findall(
            r"^'(admission\d+)' (?:does not depend on any axioms|depends on axioms: \[([^\]]*)\])$", output, re.M
        )
    }
    if (
        p.returncode != 0
        or set(ax) != expected
        or any(set(v) - {"propext", "Quot.sound", "Classical.choice"} for v in ax.values())
    ):
        raise ValueError("new Lean replay failed")
    return dict(
        binary_sha256=digest(binary),
        version=version,
        source_sha256=digest(f),
        log_sha256=hashlib.sha256(p.stdout).hexdigest(),
        axioms=ax,
    )


def build(folder, base, tokenizer_path, lean, out):
    import allocation_tasks as tour
    from allocation_routes import PRIMARY, route

    if round(PRIMARY * 1e9) != CAP_NS or tuple(tour.TRAIN) != TRAIN or out.exists():
        raise ValueError("frozen contract mismatch or existing output")
    tok = load_tokenizer(tokenizer_path)
    primary = json.loads((base / "admissions.json").read_bytes())
    if primary["members"]["manifest.json"] != digest(base / "manifest.json"):
        raise ValueError("base corpus changed")
    base_roots = {json.loads(line)["semantic_key"] for line in (base / "records.jsonl").read_bytes().splitlines()}
    src = folder / "train-only.jsonl"
    trainfile = folder / "train.json"
    data = json.loads(trainfile.read_bytes())
    if data["split"] != "train" or data["lean_binary"] != OLD_LEAN:
        raise ValueError("not original train evidence")
    collection = {(r["lineage"], tasks.digest(r["public"])): r for r in data["rows"] if r["gate"]}
    wanted = {}
    drops = []
    checked_sha = digest(Path(tour.__file__))
    candidates = []
    for row in map(json.loads, src.read_bytes().splitlines()):
        if row["split"] != "train" or row["family"] not in TRAIN or row["checker_sha256"] != checked_sha:
            raise ValueError("foreign or held example")
        g = row["public"]["leaves"]
        detail = collection[(row["lineage"], tasks.digest(g))]
        if len(g) != 1 or detail["family"] != row["family"] or detail["intervention"] not in ("unchanged", "edited"):
            raise ValueError("missing train intervention binding")
        leaves = tuple(tour.Leaf(d, tuples(x), tuples(y)) for d, x, y in g)
        paths = tuples(row["target"]["paths"])
        source_sha = row["source_sha256"]
        if not tour.valid(leaves, paths) or row["target"]["lean_source_sha256"] != source_sha:
            raise ValueError("invalid checked path")
        lit = folder / "singleton" / (source_sha + ".lean")
        original = lit.read_bytes()
        rebuilt = (
            "set_option autoImplicit false\n"
            + tour.declaration("candidate", leaves, paths)
            + "\n#print axioms candidate\n"
        )
        if original != rebuilt.encode() or digest(lit) != source_sha:
            raise ValueError("path does not recover exact Lean source")
        receipt = json.loads(lit.with_suffix(".json").read_bytes())
        check_receipt(folder, receipt)
        if not receipt["accepted"] or receipt["intended_negative"] or receipt["exit"] != 0:
            raise ValueError("not a kernel success")
        leaf = leaves[0]
        actions = compact_path(leaf.domain, leaf.left, leaf.right, paths[0], tour.rw.RULES)
        semantic = canonical_goal(leaf, tour)
        slate = quote(detail, folder)
        answer = choose(slate)
        choices = {n: route(detail, (n,)) for n in ACTIONS}
        original_ok = {n: v["seconds"] for n, v in choices.items() if v["ok"]}
        if (answer is None) != (len(original_ok) < 2) or (
            answer and answer["action"] != min(original_ok, key=lambda n: (original_ok[n], n))
        ):
            raise ValueError("quantized slate changes actual single-action optimum")
        candidates.append((row, detail, semantic, leaf, actions, slate, answer, receipt, original.decode()))
    # Edited and unchanged siblings share the same denotation and lineage. Prefer the shorter valid proof.
    for v in sorted(candidates, key=lambda v: (len(v[4]), v[0]["lineage"], v[0]["source_sha256"])):
        row, _, semantic, *_ = v
        if semantic in wanted or row["lineage"] in {x[0]["lineage"] for x in wanted.values()}:
            drops.append(
                dict(lineage=row["lineage"], semantic_key=semantic, reason="same denotation or source lineage")
            )
            continue
        wanted[semantic] = v
    works = []
    pending = []
    for row, detail, sem, leaf, actions, slate, answer, receipt, original in wanted.values():
        ids = range(12) if leaf.domain == "nat" else range(12, len(tour.rw.RULES))
        public = dict(domain=leaf.domain, equation=[leaf.left, leaf.right], rules={i: tour.rw.RULES[i][0] for i in ids})
        prompt = (
            "Compose a local rewrite certificate from left to right. Children are numbered from 1. "
            "Return steps [rule ID, address] for a forward rewrite or [rule ID, address, new local subtree] "
            "for a reverse rewrite. The empty address is the root. " + tasks.canon(public)
        )
        payload = (row, sem, detail["intervention"], receipt)
        name = tasks.digest(["tournament-composition", sem])
        pending.append((name, "composition", prompt, tasks.canon(actions), payload))
        works.append((name, actions, original))
        if answer is not None:
            question = (
                "Audit four measured one-action proof options. Each stage is [solver ns, kernel ns, kernel calls]. "
                "Feedback includes its source verification. Include the common gate once. Select the fastest "
                "accepted complete option within both caps, breaking ties by action name. "
                "Return action and elapsed_ns. "
                + tasks.canon(slate)
            )
            pending.append(
                (tasks.digest(["tournament-cost-audit", sem]), "allocation", question, tasks.canon(answer), payload)
            )
        else:
            drops.append(dict(lineage=row["lineage"], reason="fewer than two accepted affordable options"))
    out.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        stage = Path(tmp) / "data"
        stage.mkdir()
        audit = verify_new_kernel(works, lean, stage)
        dump(stage / "kernel.json", audit)
        shutil.copyfile(tokenizer_path, stage / "tokenizer.json")
        counts = Counter()
        pos = Counter()
        targets = Counter()
        indexes = []
        source_sha = digest(src)
        actions_sha = digest(trainfile)
        with (
            (stage / "train.jsonl").open("wb") as s,
            (stage / "records.jsonl").open("wb") as r,
            (stage / "receipts.jsonl").open("wb") as h,
        ):
            for root, skill, prompt, target, payload in sorted(pending):
                row, sem, mode, kernel = payload
                fam = skill + "/tournament-" + row["family"]
                if root in base_roots:
                    raise ValueError("base corpus collision")
                ids, mask, start = tokens(tok, prompt, target)
                record = dict(
                    split="train",
                    skill=skill,
                    family=fam,
                    lineage=row["lineage"],
                    semantic_key=root,
                    mathematical_denotation_sha256=sem,
                    prompt=prompt,
                    target=target,
                    delimiter=tasks.DELIMITER,
                    rights=RIGHTS,
                    source_sha256=row["source_sha256"],
                    intervention=mode,
                )
                receipt = dict(
                    split="train",
                    root=root,
                    record_sha256=tasks.digest(record),
                    skill=skill,
                    train_export_sha256=source_sha,
                    all_actions_train_sha256=actions_sha,
                    upstream_checker_sha256=checked_sha,
                    independent_checker_sha256=digest(Path(__file__)),
                    source_lean_419_compile=kernel,
                    new_lean_428_batch_sha256=audit["source_sha256"],
                    new_lean_428_log_sha256=audit["log_sha256"],
                )
                rh = tasks.digest(receipt)
                item = dict(
                    root=root,
                    family=fam,
                    split="train",
                    presentation=1,
                    lane=skill,
                    objective="checked_response",
                    token_ids=ids,
                    target_mask=mask,
                    response_start=start,
                    receipt_sha256=rh,
                )
                append(r, record)
                append(h, receipt)
                append(s, item)
                indexes.append((root, fam, tasks.digest(item), rh))
                counts[skill] += 1
                pos[skill] += len(ids)
                targets[skill] += sum(mask)
        bind_registry(stage / "registry.sqlite3", indexes, digest(stage / "train.jsonl"))
        manifest = dict(
            schema="aleph.packed-stream.v1",
            context=2048,
            pad_id=0,
            vocab_size=49152,
            tokenizer=member(stage / "tokenizer.json"),
            registry=member(stage / "registry.sqlite3"),
            shards=[dict(**member(stage / "train.jsonl"), split="train", records=len(indexes))],
        )
        dump(stage / "manifest.json", manifest)
        dump(stage / "rejections.json", drops)
        total = sum(pos.values())
        with PackedStream(stage / "manifest.json", digest(stage / "manifest.json"), "train") as stream:
            while stream.next_pack(total) is not None:
                pass
            if stream.state["processed"] != total or stream.state["started_views"] != len(indexes):
                raise ValueError("partial shard replay")
            packed = stream.state["targets_by_lane"]
        result = dict(
            rows=len(indexes),
            rows_by_lane=dict(counts),
            positions=total,
            positions_by_lane=dict(pos),
            targets_by_lane=dict(targets),
            packed_by_lane=packed,
            packed_supervised_positions=sum(packed.values()),
            target_positions=sum(targets.values()),
            raw_train_proofs=len(candidates),
            raw_measured_train_slates=len(data["rows"]),
            positive_measured_slates=len(collection),
            canonical_lineages=len(wanted),
            new_kernel_checked=len(works),
            rejections=dict(Counter(x["reason"] for x in drops)),
            no_held_files_opened=True,
            base_manifest_sha256=primary["members"]["manifest.json"],
            members={p.name: digest(p) for p in stage.iterdir() if p.is_file()},
        )
        dump(stage / "admissions.json", result)
        stage.rename(out)
    return result


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "base", "tokenizer", "lean", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    print(tasks.canon(build(args.source, args.base, args.tokenizer, args.lean, args.out.resolve())))
