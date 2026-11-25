"""Build or independently replay the first-party research-decision training shard."""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import curation_tasks as identity
import milestone_research_check as checked
import milestone_research_tasks as tasks
from curation import INTERNAL, PackedStream, append, bind_registry, dump, member, tokens
from production import TOKENIZER_SHA256, digest, load_tokenizer

HERE = Path(__file__).resolve().parent
SCHEMA = "datagoblin.aleph.research-decisions.train.v1"
TARGET = 2_092_917
BASE = Path("../aleph-candidate01")
OUT = Path("../aleph-10m/extra/custom_research")
GEN = "milestone_research_tasks.py"
CHECK = "milestone_research_check.py"
GYMS = dict(
    zip(
        tasks.FAMILIES,
        (
            "../state01/state07.py",
            "../state01/state03.py",
            "../state01/state13.py",
            "curation_check.py",
            "../../../src/datagoblin/training_gym/rlt_gym/tasks.py",
        ),
        strict=True,
    )
)
MEMBERS = ("manifest.json", "train.jsonl", "records.jsonl", "receipts.jsonl", "registry.sqlite3", "tokenizer.json")
RESERVED = {v for group in identity.RESERVED.values() for v in group} | {
    "independent_sets",
    "subset_sum",
    "linear_extensions",
    "perfect_matchings",
    "proper_colorings",
    "domino_tilings",
}
INDEX_KEYS = ("roots", "lineages", "denotations", "public", "prompts")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code_hashes():
    files = (
        GEN,
        CHECK,
        "milestone_research.py",
        "curation.py",
        "production.py",
        "curation_tasks.py",
        "../../aleph_rlt_placement/packing.py",
        *GYMS.values(),
    )
    return {f: sha((HERE / f).read_bytes().replace(b"\r\n", b"\n")) for f in sorted(set(files))}


def read(path):
    return json.loads(path.read_bytes())


def index():
    return {k: set() for k in INDEX_KEYS}


def public_sha(prompt):
    try:
        payload = json.loads(prompt.rsplit("\n", 1)[-1])
    except json.JSONDecodeError:
        return None
    return identity.digest(payload)


def identities(record):
    return dict(
        roots=record["semantic_key"],
        lineages=record["lineage"],
        denotations=record.get("mathematical_denotation_sha256"),
        public=public_sha(record["prompt"]),
        prompts=identity.digest(record["prompt"]),
    )


def overlaps(previous, keys):
    return [k for k, v in keys.items() if v is not None and v in previous[k]]


def remember(previous, keys):
    for k, v in keys.items():
        if v is not None:
            previous[k].add(v)


def old_training(base):
    frozen = read(HERE / "SUPPLY.json")["artifacts_sha256"]
    out = index()
    sources = {}
    for name in ("verified", "tournament-train"):
        folder = base / name
        manifest = folder / "admissions.json"
        if digest(manifest) != frozen[f"{name}/admissions.json"]:
            raise ValueError("prior train admission changed")
        members = read(manifest)["members"]
        names = ("records.jsonl", "receipts.jsonl", "train.jsonl", "manifest.json")
        for n in names:
            if digest(folder / n) != members[n]:
                raise ValueError(f"prior train source changed: {name}/{n}")
        total = 0
        with (
            (folder / names[0]).open("rb") as records,
            (folder / names[1]).open("rb") as receipts,
            (folder / names[2]).open("rb") as shard,
        ):
            for r, c, v in zip(records, receipts, shard, strict=True):
                rec, receipt, item = map(json.loads, (r, c, v))
                if (
                    any(z.get("split") != "train" for z in (rec, receipt, item))
                    or item["root"] != rec["semantic_key"]
                    or receipt["root"] != rec["semantic_key"]
                    or sha(c.rstrip(b"\r\n")) != item["receipt_sha256"]
                    or sha(r.rstrip(b"\r\n")) != receipt["record_sha256"]
                ):
                    raise ValueError("prior checked train row lost custody")
                remember(out, identities(rec))
                total += 1
        sources[name] = dict(rows=total, admissions_sha256=digest(manifest), members={n: members[n] for n in names})
    return out, dict(sources=sources, unique_identities={k: len(v) for k, v in out.items()})


def make(serial, tokenizer, hashes):
    name, q, proposed = tasks.candidate(serial)
    if name in RESERVED or name not in GYMS:
        raise ValueError("unexpected train family")
    label, semantic, lineage, witness = checked.SOLVE[name](q)
    if label != proposed:
        raise ValueError(f"independent witness or decision failed: {name}/{serial}")
    family = "allocation/" + name
    prompt = tasks.INSTRUCTIONS[name] + "\n" + identity.canon(q)
    root = identity.digest(["allocation", semantic])
    lineage_sha = identity.digest(lineage)
    target = identity.canon(label)
    record = dict(
        split="train",
        skill="allocation",
        family=family,
        semantic_key=root,
        lineage=lineage_sha,
        mathematical_denotation_sha256=identity.digest(semantic),
        prompt=prompt,
        target=target,
        delimiter=identity.DELIMITER,
        rights=INTERNAL,
        source_generator_sha256_lf=hashes[GEN],
        difficulty=witness,
        origin=dict(
            kind="fresh-first-party",
            serial=serial,
            seed=tasks.SEED,
            source_gym=GYMS[name],
            source_gym_sha256_lf=hashes[GYMS[name]],
        ),
    )
    receipt = dict(
        schema=SCHEMA,
        split="train",
        root=root,
        lineage=lineage_sha,
        accepted=True,
        check="independent-public-math",
        record_sha256=identity.digest(record),
        generator_sha256_lf=hashes[GEN],
        verifier_sha256_lf=hashes[CHECK],
        tokenizer_sha256=TOKENIZER_SHA256,
        witness=witness,
    )
    ids, mask, start = tokens(tokenizer, prompt, target)
    item = dict(
        root=root,
        family=family,
        split="train",
        presentation=1,
        lane="allocation",
        objective="checked_response",
        token_ids=ids,
        target_mask=mask,
        response_start=start,
        receipt_sha256=identity.digest(receipt),
    )
    return record, receipt, item


def select(target, tokenizer, hashes, old, emit):
    fresh = index()
    exclusions = Counter()
    pending = []
    history = [1]
    remaining = target
    last = None
    tail_start = None
    # Only complete checked rows enter the cost-constrained tail subset.
    for serial in range(target // 80 + 4096):
        candidate = make(serial, tokenizer, hashes)
        keys = identities(candidate[0])
        old_hit = overlaps(old, keys)
        fresh_hit = overlaps(fresh, keys)
        if old_hit or fresh_hit:
            exclusions[("prior:" if old_hit else "candidate:") + ",".join(old_hit or fresh_hit)] += 1
            continue
        remember(fresh, keys)
        length = len(candidate[2]["token_ids"])
        if remaining > 8000:
            emit(candidate)
            remaining -= length
            continue
        if tail_start is None:
            tail_start = serial
        pending.append(candidate)
        history.append((history[-1] | (history[-1] << length)) & ((1 << (remaining + 1)) - 1))
        if (history[-1] >> remaining) & 1:
            last = serial
            break
        if len(pending) >= 512:
            raise ValueError("no exact whole-example tail within bound")
    if last is None:
        raise ValueError("research candidate bound exhausted")
    balance = remaining
    kept = []
    for i in range(len(pending) - 1, -1, -1):
        if not ((history[i] >> balance) & 1):
            kept.append(i)
            balance -= len(pending[i][2]["token_ids"])
    if balance:
        raise ValueError("incorrect whole-example subset")
    for i in reversed(kept):
        emit(pending[i])
    return dict(
        seed=tasks.SEED,
        serials_considered=last + 1,
        duplicate_rejections=dict(exclusions),
        tail_first_serial=tail_start,
        tail_last_serial=last,
        tail_remaining_positions=remaining,
        tail_candidates=len(pending),
        tail_selected=len(kept),
        tail_excluded=len(pending) - len(kept),
        tail_rule="Earliest reachable exact sum in round-robin order; backtrack preferring earlier complete examples.",
    )


def summary_counter():
    return defaultdict(Counter)


def count(stats, record, item):
    fam = record["family"]
    bucket = stats[fam]
    bucket["rows"] += 1
    bucket["positions"] += len(item["token_ids"])
    bucket["targets"] += sum(item["target_mask"])
    bucket["decision/" + record["difficulty"]["choice"]] += 1
    for key in ("merged", "equivalent"):
        if key in record["difficulty"]:
            bucket[f"{key}/{record['difficulty'][key]}"] += 1


def totals(stats):
    return {k: sum(t[k] for t in stats.values()) for k in ("rows", "positions", "targets")}


def verify(out, base):
    start = time.monotonic()
    admitted = read(out / "admissions.json")
    if set(admitted["members"]) != set(MEMBERS) or any(digest(out / n) != h for n, h in admitted["members"].items()):
        raise ValueError("admitted bytes changed")
    h = code_hashes()
    if h != admitted["implementation_sha256_lf"] or admitted["rights"] != INTERNAL or set(tasks.FAMILIES) & RESERVED:
        raise ValueError("implementation, rights or family changed")
    old, prior = old_training(base)
    if prior != admitted["prior_train"]:
        raise ValueError("prior train sources changed")
    seen = index()
    tok = load_tokenizer(out / "tokenizer.json")
    stats = summary_counter()
    last_serial = -1
    with ExitStack() as stack:
        sources = [
            stack.enter_context((out / n).open("rb")) for n in ("records.jsonl", "receipts.jsonl", "train.jsonl")
        ]
        for raw_rec, raw_cert, raw_item in zip(*sources, strict=True):
            rec, cert, item = map(json.loads, (raw_rec, raw_cert, raw_item))
            serial = rec["origin"]["serial"]
            if serial <= last_serial:
                raise ValueError("candidate repeated or order changed")
            last_serial = serial
            rebuilt = make(serial, tok, h)
            if any(
                identity.canon(x).encode() != b.rstrip(b"\r\n")
                for x, b in zip(rebuilt, (raw_rec, raw_cert, raw_item), strict=True)
            ):
                raise ValueError("row, independent receipt or tokenization failed replay")
            prompt, q = rec["prompt"].split("\n", 1)
            family = rec["family"].removeprefix("allocation/")
            verified, _, _, witness = checked.SOLVE[family](json.loads(q))
            if (
                prompt != tasks.INSTRUCTIONS[family]
                or json.loads(rec["target"]) != verified
                or cert["witness"] != witness
                or sha(raw_cert.rstrip(b"\r\n")) != item["receipt_sha256"]
                or sha(raw_rec.rstrip(b"\r\n")) != cert["record_sha256"]
            ):
                raise ValueError("native receipt, parsed public tariff or certificate changed")
            keys = identities(rec)
            if overlaps(old, keys) or overlaps(seen, keys):
                raise ValueError("prior or internal duplicate admitted")
            remember(seen, keys)
            count(stats, rec, item)
    total = totals(stats)
    if (
        stats != admitted["families"]
        or total != admitted["totals"]
        or total["positions"] != admitted["requested_positions"]
    ):
        raise ValueError("incorrect admitted counters")
    m = out / "manifest.json"
    with PackedStream(m, digest(m), "train") as stream:
        while stream.next_pack(total["positions"] + 1) is not None:
            pass
        packed = stream.state
    if (
        packed["processed"] != total["positions"]
        or packed["started_views"] != total["rows"]
        or packed["positions_by_lane"] != {"allocation": total["positions"]}
    ):
        raise ValueError("native stream lost or recycled real positions")
    if not 0 < packed["targets"] <= total["targets"] or packed["canonical_view_positions"] != total["positions"]:
        raise ValueError("checked-response native accounting failed")
    return dict(
        schema=SCHEMA,
        checks_passed=True,
        admissions_sha256=digest(out / "admissions.json"),
        members=admitted["members"],
        totals=total,
        families=stats,
        native_stream=packed,
        checks=[
            "prior train artifact chains",
            "independent public equations, tariffs and witnesses on all rows",
            "generator, serialized prompt and complete answer replay",
            "canonical identity, lineage, premises and prompt dedup",
            "receipt, response mask and tokenizer round trip",
            "all production PackedStream views and shard/index bindings",
        ],
        train_only=True,
        prior_train=prior,
        prior_train_overlap=0,
        internal_overlap=0,
        distinct_denotations=len(seen["denotations"]),
        distinct_lineages=len(seen["lineages"]),
        wall_seconds=round(time.monotonic() - start, 3),
    )


def build(out, base, target):
    if out.exists() or target < 1:
        raise ValueError("use a fresh output directory and a positive target")
    started = time.monotonic()
    source = base / "tokenizer.json"
    tok = load_tokenizer(source)
    old, prior = old_training(base)
    h = code_hashes()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        folder = Path(scratch) / "admission"
        folder.mkdir()
        shutil.copyfile(source, folder / "tokenizer.json")
        stats = summary_counter()
        registry = []
        with ExitStack() as stack:
            files = [
                stack.enter_context((folder / n).open("wb")) for n in ("records.jsonl", "receipts.jsonl", "train.jsonl")
            ]

            def emit(candidate):
                rec, _, item = candidate
                for handle, row in zip(files, candidate, strict=True):
                    append(handle, row)
                registry.append((item["root"], item["family"], identity.digest(item), item["receipt_sha256"]))
                count(stats, rec, item)

            selection = select(target, tok, h, old, emit)
        total = totals(stats)
        if total["positions"] != target or len(stats) != len(tasks.FAMILIES):
            raise ValueError("research mix failed coverage or did not meet requested positions")
        bind_registry(folder / "registry.sqlite3", registry, digest(folder / "train.jsonl"))
        dump(
            folder / "manifest.json",
            dict(
                schema="aleph.packed-stream.v1",
                context=2048,
                vocab_size=49152,
                pad_id=0,
                tokenizer=member(folder / "tokenizer.json"),
                registry=member(folder / "registry.sqlite3"),
                shards=[dict(**member(folder / "train.jsonl"), records=total["rows"], split="train")],
            ),
        )
        dump(
            folder / "admissions.json",
            dict(
                schema=SCHEMA,
                requested_positions=target,
                selection=selection,
                totals=total,
                positions=target,
                rows=total["rows"],
                target_positions=total["targets"],
                positions_by_lane={"allocation": target},
                targets_by_lane={"allocation": total["targets"]},
                rows_by_lane={"allocation": total["rows"]},
                families=stats,
                prior_train=prior,
                implementation_sha256_lf=h,
                rights=INTERNAL,
                sources=dict(
                    generator=GEN,
                    independent_checker=CHECK,
                    conceptual_gym_sources=GYMS,
                    kind="first-party generated TRAIN examples from public mathematical specifications",
                    imported_evaluation_rows=0,
                    imported_old_train_rows=0,
                    template_namespace=tasks.FAMILIES,
                    executed_gym_controls={
                        "native-experiment-plan-audit": [
                            "rlt_gym.tasks._world(experiment_plan, ood=False)",
                            "rlt_gym.tasks.reference_answer",
                            "rlt_gym.tasks.verify on reference and proposed plan",
                        ],
                        "polarity-cube-witness-cost": [
                            "curation_check.machine_witness on the pair of fixed-length observation languages"
                        ],
                    },
                    sensor_separation=(
                        "Prime fields 13/17/19, 2D affine trajectories, at most one noninvertible map, explicit paid "
                        "sensor selection. Reset and operator primitives recur across gyms. CONFIRM14 scalar "
                        "four-state SET/ADD reasoning and double-partial erasure have different signatures and "
                        "are excluded."
                    ),
                ),
                registered_held_exclusions=sorted(RESERVED),
                members={n: digest(folder / n) for n in MEMBERS},
            ),
        )
        audit = verify(folder, base)
        audit["build_including_replay_seconds"] = round(time.monotonic() - started, 3)
        dump(folder / "audit.json", audit)
        folder.rename(out)
    return audit


def self_test():
    choices = defaultdict(set)
    for serial in range(400):
        family, q, label = tasks.candidate(serial)
        replay, _, _, telemetry = checked.SOLVE[family](json.loads(identity.canon(q)))
        if replay != label:
            raise AssertionError((serial, family, q, label, replay))
        choices[family].add(telemetry["choice"])
    if any(len(options) < 2 for options in choices.values()) or set(choices) != set(tasks.FAMILIES):
        raise AssertionError(choices)
    # Small, manually costed fixtures guard shared implementation misunderstandings.
    fixtures = [
        (
            "gcd-solver-tariff",
            {"pair": [8, 12], "prices": [3, 1, 2]},
            {"gcd": 4, "counts": [3, 8, 2], "bills": [9, 12], "winner": "euclid"},
        ),
        (
            "minimax-probe-purchase",
            {"catalog": [0, 1, 2, 3], "observed": [[1, 0]], "fees": [1, 2], "verify_fee": 3},
            {"survivors": [0, 1], "probe": 0, "branches": [1, 1], "bill": 4},
        ),
        (
            "counterfactual-sensor-choice",
            {
                "p": 13,
                "baseline": [0, 1],
                "substitution": [2, 1],
                "maps": [[0, 0, 0, 0, 2, 3]],
                "sensors": [[1, 1, 0, 1], [0, 1, 0, 3]],
            },
            {"terminal": [[2, 3], [2, 3]], "first_equal": 1, "sensor": 1, "readings": [0, 2], "fee": 3},
        ),
        (
            "polarity-cube-witness-cost",
            {
                "bits": 4,
                "reference": [[1, 0]],
                "proposal": [[3, 0], [1, 2]],
                "scan_fee": 1,
                "emit_fee": 2,
                "scan_support_fee": 3,
            },
            {"witness": None, "truth": None, "charges": [48, 8], "winner": "algebra"},
        ),
    ]
    for family, q, expected in fixtures:
        got = checked.SOLVE[family](q)[0]
        if got != expected:
            raise AssertionError((family, got, expected))
    return dict(checked=400, manual_fixtures=len(fixtures), choices={k: sorted(v) for k, v in choices.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify", "self-test"))
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--positions", type=int, default=TARGET)
    args = p.parse_args()
    if args.action == "self-test":
        print(json.dumps(self_test()))
        return
    report = build(args.out, args.base, args.positions) if args.action == "build" else verify(args.out, args.base)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("checks_passed", "admissions_sha256", "totals", "families", "native_stream", "wall_seconds")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
