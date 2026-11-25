"""Freeze paired terminals, create private native panels, and score the held 14M confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from confirm_reason import GRAPH_COUNT, HELD_WORDS, ORBIT_COUNT, SET, _orbit, _seed, revision, rule_rows
from corpus import AFTER, BEFORE, SEP, Row, agonist_rows, generate_repair, identity

HERE = Path(__file__).resolve().parent
ARMS = ("direct", "global_full_damp")
SEEDS = (137, 163, 191)
GRAPH_REGIMES = ("length_ood", "revision_ood")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if path.exists():
        if json.dumps(read(path), sort_keys=True, separators=(",", ":"), allow_nan=False) != serialized:
            raise ValueError(f"refusing to alter existing {path.name}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_text(serialized, encoding="utf-8")
        if os.name == "nt":
            os.rename(tmp, path)
        else:
            os.link(tmp, path)
    except FileExistsError:
        if json.dumps(read(path), sort_keys=True, separators=(",", ":"), allow_nan=False) != serialized:
            raise ValueError(f"conflicting concurrent record {path.name}") from None
    finally:
        tmp.unlink(missing_ok=True)


def sources():
    names = (
        "confirm_eval.py",
        "confirm_reason.py",
        "confirm_stats.py",
        "CONFIRM_REASON_PROTOCOL.md",
        "CONFIRM14_PROTOCOL.md",
        "CONFIRM_TEXT.md",
        "confirm_text_sources.json",
        "confirm_text.py",
    )
    files = [
        *(HERE / n for n in names),
        HERE.parent / "aleph_rlt_placement" / "study.py",
        HERE.parent / "aleph_rlt_placement" / "gym.py",
        HERE.parent / "aleph_rlt_placement" / "corpus.py",
        Path(revision.__file__),
    ]
    return {p.name: sha(p) for p in files}


def path_from(base, name):
    path = Path(name)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def lookup_text_manifest(directory):
    options = [p for p in (directory / "manifest.json", directory / "text-manifest.json") if p.is_file()]
    if len(options) != 1:
        raise ValueError("need one text manifest in the registered held-text directory")
    return options[0]


def check_native_pack(pack):
    manifest = read(pack / "dataset.json")
    if manifest["schema"] != "datagoblin.mixed-pretrain-pilot.v1":
        raise ValueError("unrecognized historical data pack")
    for name, wanted in manifest["files"].items():
        if (name.startswith("revision/") or name == "text-train.jsonl") and sha(pack / name) != wanted:
            raise ValueError("historical source digest mismatch")


def freeze(fit_manifest, text_dir, pack, out, extra_roots=()):
    out.mkdir(parents=True, exist_ok=True)
    cohort = read(fit_manifest)
    if cohort.get("reference_competent") is not True:
        raise ValueError("reference failed development competence; leave fresh panel unopened")
    if not isinstance(cohort.get("candidate_competent"), bool):
        raise ValueError("candidate development competence must be recorded")
    if not cohort.get("development_competence"):
        raise ValueError("missing development gate evidence")
    terminals, roots = {}, {Path(p).resolve() for p in extra_roots}
    paired = defaultdict(list)
    steps = set()
    for row in cohort["fits"]:
        arm, seed, step = row["arm"], row["seed"], row["step"]
        if arm not in ARMS or seed not in SEEDS or not isinstance(step, int) or step <= 0:
            raise ValueError("unregistered terminal")
        key = f"{arm}-{seed}"
        file = path_from(fit_manifest.parent, row["checkpoint"])
        if key in terminals or sha(file) != row["sha256"]:
            raise ValueError("duplicate terminal or incorrect checkpoint digest")
        default_roots = file.with_name(file.stem + "-roots.json")
        if row.get("train_roots"):
            roots.add(path_from(fit_manifest.parent, row["train_roots"]))
        elif default_roots.exists():
            roots.add(default_roots)
        elif not extra_roots:
            raise ValueError("supply reasoning training root hashes for every terminal")
        terminals[key] = {"checkpoint": str(file), "sha256": row["sha256"], "step": step, "arm": arm, "seed": seed}
        meta = file.with_suffix(".json")
        if meta.is_file():
            receipt = read(meta)
            if (
                receipt["arm"],
                receipt["seed"],
                receipt["step"],
                receipt["checkpoint_sha256"],
                receipt["params"],
                receipt["visits"],
            ) != (arm, seed, step, row["sha256"], 14179584, 24):
                raise ValueError("endpoint metadata mismatch")
            if sha(meta.parent / receipt["roots"]) != receipt["roots_sha256"]:
                raise ValueError("training root sidecar differs from endpoint receipt")
            paired[seed].append(
                {
                    k: receipt[k]
                    for k in (
                        "stream",
                        "tokens",
                        "padded",
                        "labels",
                        "prefix_stream",
                        "prefix_tokens",
                        "prefix_padded",
                        "roots_sha256",
                        "recipe",
                        "corpus",
                    )
                }
            )
            terminals[key]["endpoint_receipt_sha256"] = sha(meta)
        steps.add(step)
    if set(terminals) != {f"{a}-{s}" for a in ARMS for s in SEEDS} or len(steps) != 1:
        raise ValueError("need six matched terminals at one horizon")
    if paired and (set(paired) != set(SEEDS) or any(len(p) != 2 or p[0] != p[1] for p in paired.values())):
        raise ValueError("endpoint training streams, roots or recipes are unpaired")
    check_native_pack(pack)
    target = {
        "schema": "aleph-confirm-frozen-v1",
        "sources": sources(),
        "fits": terminals,
        "fit_manifest": str(fit_manifest.resolve()),
        "fit_manifest_sha256": sha(fit_manifest),
        "text_dir": str(text_dir.resolve()),
        "text_manifest_sha256": sha(lookup_text_manifest(text_dir)),
        "pack": str(pack.resolve()),
        "pack_manifest_sha256": sha(pack / "dataset.json"),
        "train_root_sources": {str(p): sha(p) for p in sorted(roots)},
        "reference_development_competent": True,
        "candidate_development_competent": cohort["candidate_competent"],
    }
    save(out / "freeze.json", target)
    print("frozen matched terminals", len(terminals), sha(out / "freeze.json"), flush=True)


def verify_freeze(out):
    lock = read(out / "freeze.json")
    pack, text_dir = Path(lock["pack"]), Path(lock["text_dir"])
    if lock["sources"] != sources() or sha(Path(lock["fit_manifest"])) != lock["fit_manifest_sha256"]:
        raise ValueError("frozen source code or fit manifest changed")
    if (
        sha(lookup_text_manifest(text_dir)) != lock["text_manifest_sha256"]
        or sha(pack / "dataset.json") != lock["pack_manifest_sha256"]
    ):
        raise ValueError("held text or native archive manifest changed")
    if any(sha(Path(p)) != h for p, h in lock["train_root_sources"].items()):
        raise ValueError("training root manifest changed")
    check_native_pack(pack)
    if any(sha(Path(v["checkpoint"])) != v["sha256"] for v in lock["fits"].values()):
        raise ValueError("frozen checkpoint changed")
    return lock


def outcome(row):
    if len(row.good) != 1:
        raise ValueError("single checked action required")
    return row.good[0]


def repair_cues(row):
    prompt = row.prompt
    if prompt[2] != SEP or prompt[-1] != SEP:
        raise ValueError("invalid register delimiters")
    proposal, start = prompt[1], prompt[3]
    ops = list(zip(prompt[4:-1:2], prompt[5:-1:2], strict=True))
    last_op, last_arg = ops[-1]
    args = [v for _, v in ops]
    anchor = start
    survivors = []
    for op, arg in ops:
        if op == SET:
            anchor = arg
            survivors = []
        else:
            survivors.append(arg)

    def last(x):
        return last_arg if last_op == SET else (x + last_arg) % 4

    counts = Counter([proposal, start, *args])
    return {
        "zero": 0,
        "proposal": proposal,
        "initial": start,
        "last_operand": last_arg,
        "last_reset_or_start": anchor,
        "mode": max(range(4), key=lambda k: counts[k]),
        "last_two_operands": sum(args[-2:]) % 4,
        "last_from_initial": last(start),
        "last_from_previous_operand": last(args[-2] if len(args) > 1 else start),
        "anchor_then_two_tail_adds": (anchor + sum(survivors[-2:])) % 4,
    }


def jump(table, start, steps):
    """Checked endpoint via binary powering of a finite map."""
    power = tuple(table)
    while steps:
        if steps & 1:
            start = power[start]
        power = tuple(power[i] for i in power)
        steps //= 2
    return start


def rule_cues(row):
    x = row.prompt
    table = x[1:33]
    start, steps = x[34:36]
    if x[0] not in (BEFORE, AFTER) or x[33] != SEP or x[36] != SEP or x[-1] != SEP:
        raise ValueError("invalid graph delimiters")
    src, dst = (x[-3], x[-2]) if x[0] == AFTER else (None, None)
    changed = list(table)
    if src is not None:
        changed[src] = dst
    if jump(changed, start, steps) != outcome(row):
        raise ValueError("independent graph transition powering disagrees")
    stale = jump(table, start, steps)
    path = [start]
    for _ in range(steps):
        path.append(table[path[-1]])
    repaired = stale
    if src in path[:-1]:
        remaining = steps - path.index(src) - 1
        if remaining <= 1:
            repaired = jump(changed, dst, remaining)
    return {
        "zero": 0,
        "start": start,
        "first_old_hop": table[start],
        "first_current_hop": changed[start],
        "two_current_hops": jump(changed, start, min(2, steps)),
        "first_then_patch_destination": table[start] if src is None else dst,
        "stale_original_walk": stale,
        "patch_with_one_tail_hop": repaired,
    }


def checked_row(row, task, family, template, variant, turn, changed=None):
    cues = (repair_cues if row.lane == "repair" else rule_cues)(row)
    return {
        "row": asdict(row),
        "task": task,
        "family": family,
        "template": template,
        "variant": variant,
        "turn": turn,
        "changed": changed,
        "cue_hits": {k: int(v == outcome(row)) for k, v in cues.items()},
    }


def graph_hash(task):
    return identity(sorted(tuple(edge) for edge in task["links"]))


def archive_rows(pack):
    directory = pack / "revision"
    source = read(directory / "manifest.json")
    records = {}
    for name, meta in source["splits"].items():
        tasks = directory / f"{name}.tasks.jsonl"
        targets = directory / f"{name}.targets.jsonl"
        if sha(tasks) != meta["tasks_sha256"] or sha(targets) != meta["targets_sha256"]:
            raise ValueError("native graph member changed")
        public = [json.loads(s) for s in tasks.read_bytes().splitlines()]
        truth = {v["id"]: v for s in targets.read_bytes().splitlines() if (v := json.loads(s))}
        if len(public) != len(truth) or len(public) != meta["items"]:
            raise ValueError("native graph member count mismatch")
        records[name] = [(t, truth[t["id"]]) for t in public]
    return records


def graph_entries(games, which, regime):
    grouped = defaultdict(list)
    for public, target in games:
        grouped[public["lineage"]].append((public, target))
    answer = []
    for root, variants in grouped.items():
        changed_set = set()
        for task, target in variants:
            changed = target["before"] != target["after"]
            changed_set.add(changed)
            pair = list(rule_rows([(task, target)]))
            for r, turn in zip(pair, ("before", "after"), strict=True):
                if r.root != root:
                    raise ValueError("graph lineage mismatch")
                answer.append(checked_row(r, which, regime, regime, task["id"], turn, changed))
        if len(variants) != 2 or changed_set != {True, False}:
            raise ValueError("invalid changed/no-op lineage")
    return answer


def root_values(value):
    if isinstance(value, str):
        if len(value) == 64 and all(c in "0123456789abcdef" for c in value):
            yield value
    elif isinstance(value, (list, dict)):
        for v in value.values() if isinstance(value, dict) else value:
            yield from root_values(v)


def prior_repair_programs():
    """Exclude all relevant computations appearing in old public train and development traces."""
    used = set()
    for batch in agonist_rows(generate_repair, 512, 96):
        for row in batch:
            p = row.prompt
            cut = p.index(SEP, 2)
            ops = [(16 + (v - 16) % 2, x) for v, x in zip(p[2:cut:3], p[3:cut:3], strict=True)]
            initial = p[1]
            for left, (op, arg) in enumerate(ops):
                for length in (4, 5, 6):
                    sub = ops[left : left + length]
                    if len(sub) == length:
                        used.add(identity([initial, sub]))
                initial = arg if op == SET else (initial + arg) % 4
    return used


def prepare(out):
    lock = verify_freeze(out)
    target = out / "private-panel.json"
    if target.exists() and (out / "panel.json").exists():
        index = read(out / "panel.json")
        if sha(target) != index["panel_sha256"] or sha(out / "freeze.json") != index["freeze_sha256"]:
            raise ValueError("existing private panel differs from index")
        return
    seed_file = out / "private.seed"
    if target.exists() and not seed_file.exists():
        raise ValueError("orphaned panel without its private draw")
    if not seed_file.exists():
        with seed_file.open("x", encoding="ascii") as fp:
            fp.write(secrets.token_hex(32) + "\n")
    salt = seed_file.read_text(encoding="ascii").strip()
    prior = set()
    for source in lock["train_root_sources"]:
        roots = set(root_values(read(Path(source))))
        if not roots:
            raise ValueError("empty reasoning training root manifest")
        prior.update(roots)
    old = archive_rows(Path(lock["pack"]))
    archived_lineages = {t["lineage"] for examples in old.values() for t, _ in examples}
    archived_graphs = {graph_hash(t) for examples in old.values() for t, _ in examples}
    rows = []
    unique = set(prior)
    programs = prior_repair_programs()
    prior_programs = len(programs)
    for family, words in HELD_WORDS.items():
        for word in words:
            rng = random.Random(_seed(salt, family, word))
            collected = 0
            attempts = 0
            while collected < ORBIT_COUNT:
                attempts += 1
                if attempts > 50000:
                    raise ValueError("cannot fill a distinct held program template")
                group = _orbit(word, rng, family)
                root = group[0].root
                orbit_programs = {
                    identity([r.prompt[3], list(zip(r.prompt[4:-1:2], r.prompt[5:-1:2], strict=True))]) for r in group
                }
                if len(orbit_programs) != 4:
                    raise ValueError("counterfactuals duplicated a program")
                if root in unique or orbit_programs & programs:
                    continue
                programs.update(orbit_programs)
                unique.add(root)
                for row in group:
                    rows.append(
                        checked_row(
                            row, "repair", family, word, str(row.order), "answer", row.prompt[1] != outcome(row)
                        )
                    )
                collected += 1
    seen_graphs = set(archived_graphs)
    seen_lineages = set(archived_lineages) | prior
    for regime in GRAPH_REGIMES:
        examples = list(revision.episodes(_seed(salt, regime), regime, GRAPH_COUNT))
        new_lineages = {t["lineage"] for t, _ in examples}
        new_graphs = {graph_hash(t) for t, _ in examples}
        if (
            len(new_lineages) != GRAPH_COUNT
            or len(new_graphs) != GRAPH_COUNT
            or new_lineages & seen_lineages
            or new_graphs & seen_graphs
        ):
            raise ValueError("fresh generator overlapped previous graph/lineage identities")
        seen_graphs.update(new_graphs)
        seen_lineages.update(new_lineages)
        rows.extend(graph_entries(examples, "revision", regime))
        if any(t["lineage"] in prior for t, _ in old[regime]):
            raise ValueError("historical OOD graph entered training")
        rows.extend(graph_entries(old[regime], "archived_revision", regime))
    save(target, rows)
    save(
        out / "panel.json",
        {
            "schema": "aleph-confirm-independent-reason-v1",
            "freeze_sha256": sha(out / "freeze.json"),
            "panel_sha256": sha(target),
            "private_seed_sha256": sha(seed_file),
            "train_hashes_checked": len(prior),
            "archived_public_repair_subprograms_checked": prior_programs,
            "archive_graphs_checked": len(archived_graphs),
            "archive_lineages_checked": len(archived_lineages),
            "rows_by_task": dict(Counter(r["task"] for r in rows)),
        },
    )
    print("prepared joint reasoning panel", dict(Counter(r["task"] for r in rows)), flush=True)


def key(row):
    return row["lane"], row["root"], row["group"], row["order"]


def units(panel, scores):
    lookup = {key(r): r for r in scores}
    grouped = defaultdict(list)
    for item in panel:
        row = item["row"]
        scored = lookup.pop(key(row))
        if not scored["known"]:
            raise ValueError("missing checked outcome")
        hits = {"model": int(scored["hit"]), **item["cue_hits"]}
        grouped[item["task"], item["family"], item["template"], row["root"]].append((item, hits))
    if lookup:
        raise ValueError("unexpected scored reasoning case")
    result = []
    for (task, family, template, root), items in sorted(grouped.items()):
        methods = items[0][1].keys()
        if len(items) != 4:
            raise ValueError("incomplete orbit or graph lineage")
        if task == "repair":
            values = {m: sum(h[m] for _, h in items) / 4 for m in methods}
            wrong = [h["model"] for x, h in items if x["changed"]]
            if len(wrong) != 3:
                raise ValueError("invalid wrong-proposal count")
            diagnostic = {
                "all_orbit_exact": int(all(h["model"] for _, h in items)),
                "wrong_proposal_accuracy": sum(wrong) / 3,
            }
        else:
            episodes = defaultdict(list)
            for x, h in items:
                episodes[x["variant"]].append((x, h))
            if len(episodes) != 2 or any({i["turn"] for i, _ in e} != {"before", "after"} for e in episodes.values()):
                raise ValueError("incomplete rule-revision turns")
            values = {m: sum(all(h[m] for _, h in e) for e in episodes.values()) / 2 for m in methods}
            diagnostic = {"all_variants_exact": int(all(h["model"] for _, h in items))}
            for e in episodes.values():
                kind = "changed" if e[0][0]["changed"] else "noop"
                diagnostic[kind + "_both_turns_exact"] = int(all(h["model"] for _, h in e))
                for x, h in e:
                    diagnostic[kind + "_" + x["turn"]] = h["model"]
        result.append(
            {
                "task": task,
                "family": family,
                "template": template,
                "root": root,
                "scores": values,
                "diagnostics": diagnostic,
            }
        )
    return result


def score_cell(out, arm, seed):
    import torch
    from confirm_text import document_scores, load_rows
    from study import dense, score

    lock = verify_freeze(out)
    panel_meta = read(out / "panel.json")
    data = out / "private-panel.json"
    if sha(data) != panel_meta["panel_sha256"] or sha(out / "freeze.json") != panel_meta["freeze_sha256"]:
        raise ValueError("private evaluation panel or freeze changed")
    cell = lock["fits"][f"{arm}-{seed}"]
    path = out / f"score-{arm}-{seed}.json"
    stamp = {
        "checkpoint_sha256": cell["sha256"],
        "freeze_sha256": panel_meta["freeze_sha256"],
        "reason_panel_sha256": panel_meta["panel_sha256"],
        "text_panel_sha256": lock["text_manifest_sha256"],
        "arm": arm,
        "seed": seed,
        "step": cell["step"],
    }
    if path.exists():
        if read(path)["stamp"] != stamp:
            raise ValueError("stale terminal score")
        return
    panel = read(data)
    rows = [Row(**{**e["row"], "prompt": tuple(e["row"]["prompt"]), "good": tuple(e["row"]["good"])}) for e in panel]
    text = load_rows(Path(lock["text_dir"]), lock["text_manifest_sha256"])
    model = dense(256)
    state = torch.load(cell["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    if sum(p.numel() for p in model.parameters()) != 14179584:
        raise ValueError("wrong registered model geometry")
    output, _ = score(model, [*rows, *text], arm, 2, 1)
    by_doc = document_scores(output, Path(lock["text_dir"]), lock["text_manifest_sha256"])
    clusters = units(panel, [x for x in output if x["lane"] != "text"])
    save(
        path,
        {
            "stamp": stamp,
            "text": by_doc,
            "clusters": clusters,
            "private_text_windows": [x for x in output if x["lane"] == "text"],
        },
    )
    print("scored terminal", arm, seed, len(clusters), "reasoning clusters", flush=True)


def reduce(out, speed):
    from confirm_stats import compare
    from confirm_text import baselines, pins

    lock = verify_freeze(out)
    frame = read(out / "panel.json")
    scores = {}
    for arm in ARMS:
        for seed in SEEDS:
            label = f"{arm}-{seed}"
            item = read(out / f"score-{label}.json")
            stamp = item["stamp"]
            if (
                stamp["checkpoint_sha256"] != lock["fits"][label]["sha256"]
                or stamp["freeze_sha256"] != sha(out / "freeze.json")
                or stamp["reason_panel_sha256"] != frame["panel_sha256"]
                or stamp["text_panel_sha256"] != lock["text_manifest_sha256"]
            ):
                raise ValueError("mismatched evaluation capsule")
            scores[label] = item
    text_baselines = baselines(Path(lock["pack"]), Path(lock["text_dir"]), lock["text_manifest_sha256"])
    rules = {d["id"]: d["stratum"] for d in pins()}
    speed_row = read(speed)
    cpu_ratio = speed_row["median_ratios"]["cpu"]
    if not math.isfinite(cpu_ratio) or (cpu_ratio >= 0.9) != speed_row["passed"]:
        raise ValueError("controlled CPU throughput gate inconsistent")
    result = compare(scores, rules, text_baselines)
    result["development"] = {
        a: lock[k]
        for a, k in zip(ARMS, ("reference_development_competent", "candidate_development_competent"), strict=True)
    }
    result["training_throughput"] = {
        "controlled_cpu_ratio": cpu_ratio,
        "pass": cpu_ratio >= 0.9,
        "speed_receipt_sha256": sha(speed),
    }
    result["frozen"] = {
        "freeze_sha256": sha(out / "freeze.json"),
        "held_text_sha256": lock["text_manifest_sha256"],
        "reason_panel_sha256": frame["panel_sha256"],
        "step": next(iter(lock["fits"].values()))["step"],
    }
    result["promote_B"] = bool(
        all(result["development"].values()) and result["held_gates_passed"] and result["training_throughput"]["pass"]
    )
    save(out / "decision.json", result)
    print("completed paired confirmation", result["promote_B"], flush=True)


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("freeze")
    for k in ("fit-manifest", "text-dir", "pack"):
        f.add_argument("--" + k, type=Path, required=True)
    f.add_argument("--train-roots", type=Path, action="append", default=[])
    sub.add_parser("prepare")
    s = sub.add_parser("score")
    s.add_argument("--arm", choices=ARMS, required=True)
    s.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("reduce").add_argument("--speed", type=Path, required=True)
    for parser in sub.choices.values():
        parser.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if HERE.parents[1] in (args.out.resolve(), *args.out.resolve().parents):
        p.error("keep private outputs outside the source repository")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.mode == "freeze":
        freeze(args.fit_manifest, args.text_dir, args.pack, args.out, args.train_roots)
    elif args.mode == "prepare":
        prepare(args.out)
    elif args.mode == "score":
        score_cell(args.out, args.arm, args.seed)
    else:
        reduce(args.out, args.speed)


if __name__ == "__main__":
    main()
