"""Compact development views of the checked RLT pack and executable agonist tasks.

All target sets stay in the grader/loss. The actor sees prompt IDs only. These
small encodings are development interfaces, not the planned 49,152-token codec.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from datagoblin.training_gym import rlt_pretrain_data as frozen
from datagoblin.training_gym import rlt_revision as revision

TEXT, PROOF, BEFORE, AFTER, REPAIR, COUNTER, SEP, PAD = range(256, 264)
VOCAB = 512
NONE = 32


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Row:
    lane: str
    root: str
    prompt: tuple
    good: tuple
    choices: int
    group: str = ""
    order: int = 0

    def public(self):
        return self.prompt


def proof_tokens(text):
    lexemes = re.findall(r"[\w.']+|[^\s\w]", text)
    return [272 + int.from_bytes(hashlib.sha256(t.encode()).digest()[:2], "big") % 240 for t in lexemes]


def text_rows(records):
    for row in records:
        tokens = tuple(row["tokens"])
        if len(tokens) < 2 or any(type(t) is not int or not 0 <= t < 256 for t in tokens):
            raise ValueError("invalid byte window")
        yield Row("text", row.get("source_doc_id", row["id"]), tokens[:-1], tokens[1:], 256, row.get("corpus", "train"))


def lean_rows(records):
    for public, grades in records:
        if public["id"] != grades["id"] or public["epoch"] != grades["epoch"]:
            raise ValueError("stale Lean receipt")
        lookup = {c["id"]: c["tactic"] for c in public["candidates"]}
        if set(lookup) != set(grades["outcomes"]):
            raise ValueError("inconsistent Lean candidate set")
        # A frozen proposal pool independent of grades. Keep ungraded proposals in ranking.
        order = list(
            dict.fromkeys(chain.from_iterable(public["orders"][name][:12] for name in ("compose", "wider", "legacy")))
        )
        stmt = proof_tokens(public["statement"])
        for rank, candidate in enumerate(order):
            result = grades["outcomes"][candidate]
            label = result["label"]
            if label == 1 and not (
                result.get("kernel_ok")
                and result.get("closed")
                and result.get("kernel_calls") == 1
                and set(result.get("axioms", ())) <= {"propext", "Quot.sound", "Classical.choice"}
            ):
                raise ValueError("unverified positive Lean action")
            if label not in (0, 1, None):
                raise ValueError("invalid Lean grade")
            prefix = stmt if len(stmt) <= 40 else stmt[:12] + stmt[-28:]
            prompt = (PROOF, *prefix, SEP, *proof_tokens(lookup[candidate])[:30], SEP)
            yield Row("lean", public["id"], prompt, () if label is None else (int(label),), 2, public["lineage"], rank)


def rule_rows(records):
    for task, target in records:
        links = dict(task["links"])
        patch = task["patch"]
        new = links | {patch["source"]: patch["destination"]}
        nodes = sorted(links)
        if nodes != sorted(revision.NODES) or target["id"] != task["id"] or target["lineage"] != task["lineage"]:
            raise ValueError("invalid revision row")
        start, steps = task["start"], task["steps"]
        endpoints = [revision.follow(graph, start, steps) for graph in (links, new)]
        if endpoints != [target["before"], target["after"]]:
            raise ValueError("revision replay disagrees")
        table = [nodes.index(links[k]) for k in nodes]
        src, dst = nodes.index(patch["source"]), nodes.index(patch["destination"])
        before = (BEFORE, *table, SEP, nodes.index(start), steps, SEP)
        after = (AFTER, *before[1:-1], SEP, src, dst, SEP)
        for turn, prompt, endpoint in zip(("before", "after"), (before, after), endpoints, strict=True):
            yield Row("rule", task["lineage"], prompt, (nodes.index(endpoint),), 32, task["id"] + "/" + turn)


def generate_repair(rng):
    # An EDIT overwrites the register; ADD rotates it. Repair the first faulty trace value.
    start, length = rng.randrange(4), rng.randint(3, 6)
    edits = [(rng.randrange(2), rng.randrange(4)) for _ in range(length)]
    fault = rng.randrange(length)
    trace = [start]
    for i, (op, arg) in enumerate(edits):
        value = arg if op == 0 else (trace[-1] + arg) % 4
        trace.append((value + rng.randrange(1, 4)) % 4 if i == fault else value)
    valid = []
    for proposal in range(4):
        prefix = [*trace[: fault + 1], proposal]
        if all(
            b == (v if op == 0 else (a + v) % 4)
            for a, b, (op, v) in zip(prefix[:-1], prefix[1:], edits[: fault + 1], strict=True)
        ):
            valid.append(proposal)
    assert len(valid) == 1 and valid[0] != trace[fault + 1]
    op, arg = edits[fault]
    tokens = [REPAIR, start]
    for i, ((action, x), y) in enumerate(zip(edits, trace[1:], strict=True)):
        tokens.extend((16 + action + 2 * (i == fault), x, y))
    # Repeat the public disputed transition at the query. This is an explicit, cheap diagnostic.
    tokens.extend((SEP, trace[fault], 16 + op, arg, SEP))
    return Row("repair", identity([start, edits]), tuple(tokens), tuple(valid), 4, ("set", "add")[op])


def generate_counter(rng):
    # Construct two conjunctive rules on five bits. Generate any assignment on
    # which decisions differ, or NONE if they agree on the whole finite domain.
    def rule():
        pos = rng.randrange(32)
        neg = rng.randrange(32) & ~pos
        return pos, neg

    first = rule()
    second = first if rng.random() < 0.2 else rule()

    def truth(clauses, x):
        return all(((x >> i) & 1) == expect for i, expect in clauses)

    def literals(pos, neg):
        return [(i, int(bool(pos & 1 << i))) for i in range(5) if (pos | neg) & 1 << i]

    a, b = literals(*first), literals(*second)
    valid = tuple(x for x in range(32) if truth(a, x) != truth(b, x)) or (NONE,)

    def accepts(mask, x):
        pos, neg = mask
        return (x & pos) == pos and (x & neg) == 0

    verify = tuple(x for x in range(32) if accepts(first, x) != accepts(second, x)) or (NONE,)
    assert valid == verify
    return Row(
        "counter",
        identity(sorted((first, second))),
        (COUNTER, *first, SEP, *second, SEP),
        valid,
        33,
        "equivalent" if valid == (NONE,) else "witness",
    )


def agonist_rows(generator, train_n, dev_n):
    used = set()
    for split, size in (("train", train_n), ("development", dev_n)):
        rng = random.Random(f"placement-v1:{generator.__name__}:{split}")
        rows = []
        while len(rows) < size:
            row = generator(rng)
            if row.root not in used:
                rows.append(row)
                used.add(row.root)
        yield rows


def load_corpus(pack=None):
    train, dev = defaultdict(list), defaultdict(list)
    source = {"scope": "public development", "native": "absent"}
    if pack:
        pack = Path(pack)
        manifest = json.loads((pack / "dataset.json").read_text(encoding="utf-8-sig"))
        if manifest["schema"] not in ("datagoblin.mixed-pretrain-pilot.v1", "aleph-reconstructed-development-v1"):
            raise ValueError("unrecognized mixed pack")
        # Validate members before parsing or joining any grades.
        for name, sha in manifest["files"].items():
            member = (pack / name).resolve()
            if not member.is_relative_to(pack.resolve()) or hashlib.sha256(member.read_bytes()).hexdigest() != sha:
                raise ValueError(f"pack member changed: {name}")
        source.update(
            native=manifest["schema"], dataset_sha256=hashlib.sha256((pack / "dataset.json").read_bytes()).hexdigest()
        )
        for output, split, lean, rules in (
            (train, "train", "train", "train"),
            (dev, "held", "held_structures", "development"),
        ):
            name = f"text-{split}.jsonl"
            if name in manifest["files"]:
                output["text"].extend(text_rows(frozen.json_lines(pack / name)))
            public, grades = [f"lean-{lean}.{kind}.jsonl" for kind in ("public", "receipts")]
            if public in manifest["files"] and grades in manifest["files"]:
                records = zip(frozen.json_lines(pack / public), frozen.json_lines(pack / grades), strict=True)
                output["lean"].extend(lean_rows(records))
            if "revision/manifest.json" in manifest["files"]:
                output["rule"].extend(rule_rows(frozen.rule_rows(pack / "revision", rules)))
        if manifest["schema"] == "datagoblin.mixed-pretrain-pilot.v1" and any(
            not rows.get(lane) for rows in (train, dev) for lane in ("text", "lean", "rule")
        ):
            raise ValueError("incomplete frozen mixed pack")
    if not train["rule"]:
        for out, split, size in ((train, "train", 128), (dev, "development", 32)):
            out["rule"].extend(rule_rows(revision.episodes(7331, split, size)))
        source["rule"] = "new seeded replay of rlt_revision.v1"
    for generator in (generate_repair, generate_counter):
        a, b = agonist_rows(generator, 512, 96)
        train[a[0].lane].extend(a)
        dev[b[0].lane].extend(b)
    for lane in train:
        if {r.root for r in train[lane]} & {r.root for r in dev[lane]}:
            raise ValueError(f"{lane}: root overlap")
        if {r.prompt for r in train[lane]} & {r.prompt for r in dev[lane]}:
            raise ValueError(f"{lane}: identical prompt across splits")
        if lane == "lean" and {r.group for r in train[lane]} & {r.group for r in dev[lane]}:
            raise ValueError("Lean family across splits")
    source["view_digest"] = identity(
        [
            [lane, r.root, r.prompt, r.good, r.group, r.order]
            for out in (train, dev)
            for lane, rows in sorted(out.items())
            for r in rows
        ]
    )
    return train, dev, source
