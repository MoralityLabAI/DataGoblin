"""Validate native mixed fits, reconcile independent shards and release matched paired results."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from statistics import mean

import torch

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / "aleph_rlt_placement"
sys.path.insert(0, str(LEGACY))
import corpus  # noqa: E402
import gym  # noqa: E402
import review  # noqa: E402
import study  # noqa: E402

SEEDS = (137, 163, 191)
ARMS = ("direct", "global_full_damp", "prelude_reread", "site_damp")
BASE_REV = "c9d88e390dd7b3435573e87a573f03b77525b931"
DATA_SHA = "a0d2dfbe5a93e7630c60e1c182dccabc03d60efea376eb65503b62e812c66383"
SOURCES = ("study.py", "corpus.py", "gym.py")


def original(name):
    return subprocess.check_output(["git", "show", f"{BASE_REV}:training/aleph_rlt_placement/{name}"], cwd=HERE)


def hashes():
    return (
        {n: hashlib.sha256(original(n)).hexdigest() for n in SOURCES},
        {n: gym.file_sha(LEGACY / n) for n in SOURCES},
    )


def check_controls(base, panel):
    """Check all named parameter gradients on a tiny fixture and on an archived trained model."""
    old_module = types.ModuleType("aleph_previous_gym")
    exec(compile(original("gym.py"), "previous_gym.py", "exec"), vars(old_module))
    for seed in SEEDS:
        torch.manual_seed(seed)
        old_init = old_module.Aleph(width=80, vocab=512, heads=4, kv=1, core_ff=350).state_dict()
        torch.manual_seed(seed)
        new_init = gym.Aleph(width=80, vocab=512, heads=4, kv=1, core_ff=350).state_dict()
        if old_init.keys() != new_init.keys() or any(not torch.equal(old_init[k], new_init[k]) for k in old_init):
            raise ValueError(f"archived width-80 initialization changed on seed {seed}")
    torch.manual_seed(37)
    rows = []
    native_tokens, _ = study.tensor([next(x for x in panel if x.lane == lane) for lane in ("text", "repair")])
    for width, vocab, core, tokens in ((16, 32, None, torch.randint(0, 32, (2, 5))), (80, 512, 350, native_tokens)):
        old = old_module.Aleph(width=width, vocab=vocab, heads=4, kv=1, core_ff=core)
        new = gym.Aleph(width=width, vocab=vocab, heads=4, kv=1, core_ff=core)
        if width == 80:
            state = torch.load(base / "w80-r2-s137-direct/checkpoint.pt", map_location="cpu", weights_only=True)
            old.load_state_dict(state["model"], strict=True)
        new.load_state_dict(old.state_dict(), strict=True)
        for arm in ARMS[:2]:
            old.zero_grad()
            new.zero_grad()
            a, b = old(tokens, arm), new(tokens, arm)
            if not torch.equal(a, b):
                raise ValueError(f"changed archived logits at {width}/{arm}")
            a.square().mean().backward()
            b.square().mean().backward()
            for (name, p), (other, q) in zip(old.named_parameters(), new.named_parameters(), strict=True):
                if name != other or not torch.equal(p.grad, q.grad):
                    raise ValueError(f"changed archived gradient at {width}/{arm}/{name}")
            rows.append(
                dict(
                    arm=arm,
                    width=width,
                    shape=list(a.shape),
                    equal_logits=True,
                    verified_width80_matched_init_seeds=list(SEEDS),
                    equal_named_parameter_gradients=len(list(old.parameters())),
                )
            )
    return rows


def expected(pack):
    train, dev, source = corpus.load_corpus(pack)
    if source["dataset_sha256"] != DATA_SHA:
        raise ValueError("mixed replay requires its frozen native corpus")
    for name, label in (("lean_pos", 1), ("lean_neg", 0)):
        train[name] = [r for r in train["lean"] if r.good == (label,)]
    plans = {}
    for seed in SEEDS:
        plan = dict(
            plan_hash="",
            input_positions=0,
            loss_positions=0,
            padded_positions=0,
            lane_examples={k: 0 for k in study.COUNTS},
        )
        for i in range(192):
            rows = study.choose(train, seed, i, 1)
            plan["plan_hash"] = corpus.identity(
                [plan["plan_hash"], [(r.root, r.group, r.order, r.prompt) for r in rows]]
            )
            plan["input_positions"] += sum(len(r.prompt) for r in rows)
            plan["loss_positions"] += sum(len(r.good) if r.lane == "text" else 1 for r in rows)
            plan["padded_positions"] += len(rows) * max(len(r.prompt) for r in rows)
            for row in rows:
                plan["lane_examples"][row.lane] += 1
        plans[seed] = plan
    panels = {}
    for depth in (6, 48):
        rows = study.select_panel(dev, depth)
        panels[depth] = [
            (r.lane, r.root, r.group, r.order)
            for r in sorted(rows, key=lambda r: (r.lane, len(r.prompt), r.root, r.order))
        ]
    return source, plans, panels, study.select_panel(dev, 48)


def corpus_members(pack):
    manifest = json.loads((pack / "dataset.json").read_bytes())
    expected = {"dataset.json": DATA_SHA, **manifest["files"]}
    revision = json.loads((pack / "revision/manifest.json").read_bytes())
    for split, row in revision["splits"].items():
        for kind in ("tasks", "targets"):
            expected[f"revision/{split}.{kind}.jsonl"] = row[f"{kind}_sha256"]
    for name, checksum in expected.items():
        path = (pack / name).resolve()
        if not path.is_relative_to(pack.resolve()) or gym.file_sha(path) != checksum:
            raise ValueError(f"changed frozen corpus member {name}")
    return expected


def root_order(rows):
    return [(r["lane"], r["root"], r["group"], r["order"]) for r in rows]


def reread_mechanism(row, arm):
    expected_source = "prelude" if arm == "prelude_reread" else "pass_input"
    sites = {}
    for sample in row["trace"]:
        visited = [v for v in sample["visits"] if v.get("reread")]
        if len(visited) != 1:
            raise ValueError("unexpected number of rereads on a native trace")
        visit = visited[0]
        read = visit["reread"]
        if (visit["visit"], visit["physical"], read["beta"], read["source"]) != (16, 6, 0.25, expected_source):
            raise ValueError("incorrect source, coefficient or physical reread site")
        sites[sample["lane"]] = {k: read[k] for k in ("displacement_rms", "source_rms")}
    if set(sites) != set(study.COUNTS):
        raise ValueError("missing native lane trace")
    return sites


def validate(folder, seed, arm, source, plans, panels, code):
    row = json.loads((folder / "receipt.json").read_bytes())
    cfg, ledger = row["config"], row["ledger"]
    fixed = dict(
        schema=2,
        width=80,
        r=2,
        s=1,
        seed=seed,
        arm=arm,
        steps=192,
        lr=0.0007,
        multiplier=1,
        roots=48,
        probe_roots=6,
        counts=study.COUNTS,
        weights=study.WEIGHTS,
        torch=str(torch.__version__),
        source=source,
    )
    if any(cfg[k] != v for k, v in fixed.items()) or row["params"] != 1414480:
        raise ValueError(f"{folder}: configuration or corpus mismatch")
    if cfg["code"] not in (code if arm in ARMS[:2] else (code[1],)):
        raise ValueError(f"{folder}: source mismatch")
    if (
        any(ledger[k] != v for k, v in plans[seed].items())
        or row["train_block_positions"] != 24 * ledger["padded_positions"]
    ):
        raise ValueError(f"{folder}: position, mixture or training plan mismatch")
    if gym.file_sha(folder / "checkpoint.pt") != row["checkpoint_sha256"]:
        raise ValueError(f"{folder}: checkpoint changed")
    saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True, mmap=True)
    if saved["step"] != 192 or saved["cfg"] != cfg or saved["ledger"] != ledger:
        raise ValueError(f"{folder}: checkpoint step, plan or optimizer cursor differs from receipt")
    if any(int(v["step"]) != 192 for v in saved["opt"]["state"].values()):
        raise ValueError(f"{folder}: Adam update clock differs from training ledger")
    if root_order(row["rows"]) != panels[48] or study.metrics(row["rows"]) != row["native"]:
        raise ValueError(f"{folder}: native panel or metrics mismatch")
    live = cfg["code"] == code[1]
    required = set(gym.ARMS) if live else set(gym.ARMS) - set(ARMS[2:])
    required = {f"{a}@2" for a in required} | {f"{arm}@{depth}" for depth in (1, 2, 3, 4, 6)}
    if set(row["sweep"]) != required:
        raise ValueError(f"{folder}: missing native control or depth sweep")
    for key, metrics in row["sweep"].items():
        sub = json.loads((folder / f"sweep-{key}.json").read_bytes())
        if (
            sub["cfg"] != corpus.identity(cfg)
            or root_order(sub["rows"]) != panels[6]
            or study.metrics(sub["rows"]) != metrics
        ):
            raise ValueError(f"{folder}: invalid probe {key}")
    return row


def choose_cell(out, shards, name):
    """Only a final scheduler receipt makes a candidate admissible."""
    choices = [p / name for p in (out, out / "completed-shards", *shards)]
    done = [p for p in choices if (p / "receipt.json").is_file() and (p / "checkpoint.pt").is_file()]
    return min(done, key=lambda p: (p.joinpath("receipt.json").stat().st_mtime_ns, choices.index(p))) if done else None


def admit(out, chosen):
    if chosen.is_relative_to(out):
        return chosen
    receipt_sha = gym.file_sha(chosen / "receipt.json")
    source_receipt = json.loads((chosen / "receipt.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix=".incoming-", dir=out) as temp:
        staged = Path(temp) / chosen.name
        staged.mkdir()
        # A finished fit comprises precisely the verified checkpoint, receipt
        # and its named probes. Never copy temporary or unrelated job files.
        for name in ("checkpoint.pt", "receipt.json", *(f"sweep-{k}.json" for k in source_receipt.get("sweep", {}))):
            shutil.copy2(chosen / name, staged / name)
        rec = json.loads((staged / "receipt.json").read_bytes())
        if (
            gym.file_sha(staged / "receipt.json") != receipt_sha
            or gym.file_sha(staged / "checkpoint.pt") != rec["checkpoint_sha256"]
        ):
            raise ValueError(f"shard changed during copy: {chosen}")
        # If the sequential worker already owns a partial directory, preserve it and
        # give the release its own completed copy. Windows rename refuses an existing
        # destination. On POSIX the worker can create an empty cell between the checks,
        # so always place external shards in the separate stable receipts namespace.
        stable = out / "completed-shards" / chosen.name
        stable.parent.mkdir(exist_ok=True)
        if not stable.exists():
            try:
                staged.rename(stable)
            except OSError:
                if not stable.exists():
                    raise
        if gym.file_sha(stable / "receipt.json") != receipt_sha:
            raise ValueError(f"different completed shard already admitted for {chosen.name}")
        return stable


def summarize(out, base, pack, shards=(), mode="release"):
    out, base = out.resolve(), base.resolve()
    out.mkdir(parents=True, exist_ok=True)
    shards = [x.resolve() for x in shards]
    code = hashes()
    source, plans, panels, native = expected(pack)
    args = (source, plans, panels, code)
    cells = {a: {} for a in ARMS}
    atlas, locations = [], {}
    for s in SEEDS:
        for arm in ARMS:
            name = f"w80-r2-s{s}-{arm}"
            folder = base / name if arm in ARMS[:2] else choose_cell(out, shards, name)
            if folder is None:
                if mode == "release":
                    raise ValueError(f"missing completed cell {name}")
                continue
            row = validate(folder, s, arm, *args)
            if arm in ARMS[2:]:
                folder = admit(out, folder)
                locations[name] = str(folder.relative_to(out).as_posix())
                atlas.append(
                    dict(
                        key=name,
                        checkpoint=locations[name] + "/checkpoint.pt",
                        sha256=row["checkpoint_sha256"],
                        architecture="Aleph-" + arm,
                        params=row["params"],
                        passes=2,
                        seed=s,
                        steps=192,
                        metrics=row["native"],
                        mechanism=dict(
                            single_native_reread_by_lane=reread_mechanism(row, arm),
                            gradient_preclip=[x["preclip_l2"] for x in row["ledger"]["grad"]],
                            training_stream=row["ledger"]["plan_hash"],
                            positions=row["ledger"]["input_positions"],
                        ),
                        failures="Previously visible mixed panel; limited generator and candidate-pool coverage.",
                    )
                )
            row["clusters"], _, _ = review.units(row["rows"])
            cells[arm][s] = row
    if mode == "admit":
        print("verified complete new cells", len(atlas), locations, flush=True)
        return
    audit = dict(
        corpus=source,
        corpus_members_sha256=corpus_members(pack),
        plans=plans,
        matching_sources=code,
        control_replays=check_controls(base, native),
        native_panel_digest=corpus.identity(panels[48]),
        probe_panel_digest=corpus.identity(panels[6]),
    )
    gym.atomic_json(out / "preflight.json", audit)
    if mode == "audit":
        print("pinned source, all plans and archived controls verified", flush=True)
        return
    aggregate = {
        arm: [
            {
                **r["native"],
                "checked_macro": review.checked_macro(r["native"]),
                "seed": s,
                "positions": r["ledger"]["input_positions"],
                "seconds": r["ledger"]["train_s"],
            }
            for s, r in by_seed.items()
        ]
        for arm, by_seed in cells.items()
    }
    pairs = {}
    for ref in ("direct", "global_full_damp", "site_damp"):
        group = [(cells["prelude_reread"][s], cells[ref][s]) for s in SEEDS]
        shifts = [
            {k: a["native"][k] - b["native"][k] for k in a["native"]}
            | {"checked_macro": review.checked_macro(a["native"]) - review.checked_macro(b["native"])}
            for a, b in group
        ]
        pairs[ref] = dict(
            per_seed=dict(zip(SEEDS, shifts, strict=True)),
            intervals=review.paired_intervals(group, 1000),
            positive_nll_reductions=sum(d["text_nll"] < 0 for d in shifts),
        )
    summary = dict(
        verified_existing_control_gradients=True,
        audit=audit,
        paths=locations,
        admitted_new_files={
            locations[f"w80-r2-s{s}-{arm}"]: [
                "checkpoint.pt",
                "receipt.json",
                *(f"sweep-{key}.json" for key in r["sweep"]),
            ]
            for arm in ARMS[2:]
            for s, r in cells[arm].items()
        },
        by_arm=aggregate,
        means={k: {m: mean(row[m] for row in rows) for m in rows[0] if m != "seed"} for k, rows in aggregate.items()},
        prelude_minus_control=pairs,
        probe_by_training_arm={
            arm: {
                runtime: {
                    "text_nll": mean(r["sweep"][runtime]["text_nll"] for r in by_seed.values()),
                    "checked_macro": mean(review.checked_macro(r["sweep"][runtime]) for r in by_seed.values()),
                }
                for runtime in next(iter(by_seed.values()))["sweep"]
            }
            for arm, by_seed in cells.items()
        },
        uncertainty="Public developmental clusters and fixed paired training seeds; 1000 unadjusted draws.",
    )
    if not all(math.isfinite(r["metrics"]["text_nll"]) for r in atlas):
        raise ValueError("nonfinite native language metric")
    gym.atomic_json(out / "summary.json", summary)
    gym.atomic_json(out / "atlas-input.json", atlas)
    print("released", len(atlas), "new cells and six archived same-stream controls", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--new", type=Path, required=True)
    cli.add_argument("--controls", type=Path, required=True)
    cli.add_argument("--pack", type=Path, default=Path("runs/placement/mixed"))
    cli.add_argument("--shard", type=Path, action="append", default=[])
    cli.add_argument("--mode", choices=("audit", "admit", "release"), default="release")
    a = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summarize(a.new, a.controls, a.pack, a.shard, a.mode)
