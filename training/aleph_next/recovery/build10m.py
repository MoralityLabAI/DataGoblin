"""Build a new, independently identified 10M-position train-only pilot.

Never overwrite the missing historical milestone. Recover public sources,
generate fresh checked tasks, preserve provenance, and replay the frozen
production packer. No model or paid accelerator is started.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter, defaultdict, deque
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "candidate01"
sys.path.insert(0, str(CANDIDATE))

import curation as c
import milestone_research as mr
import milestone_text as mt
import text_intake as ti
from confirm_text import body_bytes, fetch as fetch_books, native_bytes, pins
from curation_check import ALLOWED_AXIOMS, AX
from milestone_mix import replay, require_train
from milestone_text_overlap import Overlap, symbols
from production import TOKENIZER_SHA256, digest, load_tokenizer, read
from recover_sources import LBF, LRM, LT, LT_PIN, recover_looped, recover_native
from snapshot import RECIPE, next_allocation

VERSION = "aleph-direct-A-10m-replacement-v2"
SEED = 202611260915
RESEARCH_SEED = 202611260916
RECIPE_SHA = "58fd8c2805f04ed918b4547d5a177624f839648c6be98dc749426ba39cd32e87"
OLD_INDEX = "be25522ded491e0c0c3f81d87d786034b46733e1ccbe2b9cea2dafc47b48abf4"
TOKEN_URL = (
    "https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/"
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2/tokenizer.json"
)
TEXT_URL = (
    "https://huggingface.co/datasets/common-pile/libretexts_filtered/resolve/"
    "204d01e40af72b3b6d3f926e1f6148ee9ed3970a/libretexts-0000.json.gz"
)
RIGHTS = (
    "Original project-generated exercise, or an explicitly identified project training-source example. "
    "This pilot retains source attribution and verifier scope."
)
DENIED = {x for vs in c.tasks.RESERVED.values() for x in vs} | mr.RESERVED
MAX_DRAWS = 200000


def need(ok, why):
    if not ok:
        raise ValueError(why)


def download(url, path, stamp=None, limit=2_000_000_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=180) as response, path.open("wb") as f:
            n = 0
            while chunk := response.read(1024 * 1024):
                n += len(chunk)
                need(n <= limit, "download exceeds limit")
                f.write(chunk)
    if stamp is not None:
        need(digest(path) == stamp, f"source hash changed: {path.name}")


def quotas(cfg):
    goals = cfg["stream"]["lane_positions"]
    counts = dict.fromkeys(goals, 0)
    step, cap = cfg["stream"]["batch_nonpadding"], cfg["stream"]["checkpoints"][0]
    need(cap == 10_000_000, "10M milestone differs")
    for start in range(0, cap, step):
        stop = min(start + step, cap)
        while plan := next_allocation(counts, goals, stop):
            lane, n = plan
            counts[lane] += n
    return counts


def mk_checked(tok, skill, family, root, prompt, target, origin, witness):
    record = dict(
        split="train", skill=skill, family=family, semantic_key=root,
        prompt=prompt, target=target, delimiter=c.tasks.DELIMITER,
        rights=RIGHTS, origin=origin, witness=witness,
    )
    receipt = dict(
        schema=VERSION, split="train", root=root, accepted=True,
        tokenizer_sha256=TOKENIZER_SHA256, record_sha256=c.tasks.digest(record),
        check=origin["kind"], witness=witness,
    )
    ids, mask, start = c.tokens(tok, prompt, target)
    item = dict(
        root=root, family=family, split="train", lane=skill,
        presentation=1, objective="checked_response", token_ids=ids,
        target_mask=mask, response_start=start, receipt_sha256=c.tasks.digest(receipt),
    )
    return record, receipt, item


class Writer:
    def __init__(self, out, lane, token_path, cfg, global_roots, global_ids):
        self.folder = out / lane
        self.folder.mkdir()
        self.lane = lane
        self.cfg = cfg
        self.roots = global_roots
        self.sequences = global_ids
        self.stack = ExitStack()
        self.files = [
            self.stack.enter_context((self.folder / n).open("wb"))
            for n in ("records.jsonl", "receipts.jsonl", "rows.jsonl")
        ]
        self.registry = []
        self.positions = 0
        self.labels = 0
        self.counts = Counter()
        self.reject = Counter()
        shutil.copyfile(token_path, self.folder / "tokenizer.json")

    def add(self, triple):
        record, receipt, item = triple
        root = item["root"]
        seq = c.tasks.digest(item["token_ids"])
        if root in self.roots or seq in self.sequences:
            self.reject["duplicate_root_or_token_sequence"] += 1
            return False
        require_train(item, self.lane, self.cfg["stream"]["subtasks"], DENIED)
        need(record["split"] == receipt["split"] == "train", "non-train evidence")
        need(receipt["record_sha256"] == c.tasks.digest(record), "record receipt differs")
        if self.lane != "general":
            need(item["receipt_sha256"] == c.tasks.digest(receipt), "detached receipt differs")
        self.roots.add(root)
        self.sequences.add(seq)
        for f, row in zip(self.files, triple, strict=True):
            c.append(f, row)
        self.registry.append(
            (root, item["family"], c.tasks.digest(item), item.get("receipt_sha256"))
        )
        self.positions += len(item["token_ids"])
        self.labels += sum(item["target_mask"])
        self.counts[item["family"]] += 1
        return True

    def finish(self, target):
        self.stack.close()
        need(self.positions >= target, f"{self.lane} short: {self.positions} / {target}")
        folder = self.folder
        c.bind_registry(folder / "registry.sqlite3", self.registry, digest(folder / "rows.jsonl"))
        c.dump(folder / "manifest.json", dict(
            schema="aleph.packed-stream.v1", context=2048, vocab_size=49152, pad_id=0,
            tokenizer=c.member(folder / "tokenizer.json"), registry=c.member(folder / "registry.sqlite3"),
            shards=[dict(split="train", records=len(self.registry), **c.member(folder / "rows.jsonl"))],
        ))
        return {
            "path": f"{self.lane}/manifest.json",
            "manifest_sha256": digest(folder / "manifest.json"),
            "positions": self.positions,
            "raw_labels": self.labels,
            "rows": len(self.registry),
            "families": dict(self.counts),
            "rejections": dict(self.reject),
        }


def math_candidate(tok, skill, ordinal):
    row = getattr(c.tasks, skill)(random.Random(f"{VERSION}/{SEED}/{skill}/{ordinal}"), ordinal)
    if row is None:
        return None
    root, detail = c.inspect_candidate(row, {}, {})
    prompt, target = c.render(row)
    return mk_checked(
        tok, skill, skill + "/" + row["family"], root, prompt, target,
        dict(kind="independent-public-math", seed=SEED, ordinal=ordinal,
             generator_sha256=digest(CANDIDATE / "curation_tasks.py"),
             verifier_sha256=digest(CANDIDATE / "curation_check.py")),
        detail,
    )


def research_candidate(tok, ordinal, hashes):
    # Use the original separately implemented proposal and verification functions
    # with a new, explicitly recorded train seed.
    mr.tasks.SEED = RESEARCH_SEED
    return mr.make(ordinal, tok, hashes)


def fill_research(writer, tok, target):
    hashes = mr.code_hashes()
    seen = mr.index()
    for serial in range(MAX_DRAWS):
        triple = research_candidate(tok, serial, hashes)
        keys = mr.identities(triple[0])
        if mr.overlaps(seen, keys):
            writer.reject["semantic_or_lineage_duplicate"] += 1
            continue
        mr.remember(seen, keys)
        writer.add(triple)
        if writer.positions >= target:
            return
    raise ValueError("research generator bound exhausted")


def fill_math(writer, tok, target, skills):
    for draw in range(MAX_DRAWS):
        ordinal, slot = divmod(draw, len(skills))
        triple = math_candidate(tok, skills[slot], ordinal)
        if triple is None:
            writer.reject["no_reachable_fault"] += 1
            continue
        writer.add(triple)
        if writer.positions >= target:
            return
    raise ValueError(f"mathematical generator bound exhausted: {writer.lane}")


def normalize(weights):
    s = sum(weights, Fraction(0))
    return [v/s for v in weights]


def lrm_observations(row):
    old, slots, new = row["old"], row["correction_slots"], row["replacements"]
    need(len(old) == 12 and len(slots) == len(new) == 6, "LRM geometry")
    need(len(set(slots)) == 6 and all(0 <= x < 12 for x in slots), "LRM slots")
    need(all(type(x) is int and x in (0, 1) for x in old+new), "LRM bits")
    patch = dict(zip(slots, new, strict=True))
    shape = sorted((b, patch.get(i, -1)) for i, b in enumerate(old))
    return old, slots, new, shape


def lrm_target(row):
    old, slots, new, _ = lrm_observations(row)
    prior = [Fraction(9, 20), Fraction(7, 20), Fraction(1, 5)]
    prob = [Fraction(19, 20), Fraction(13, 20), Fraction(3, 20)]
    k0 = sum(old)
    k1 = sum(b for i, b in enumerate(old) if i not in slots) + sum(new)
    # Initial likelihood marginalizes unknown correction positions. Full
    # likelihood retains the valid old bits and the accepted replacements.
    weights = [
        pi * sum(
            Fraction(math.comb(6, k) * math.comb(6, k0-k), 64) * p**k * (1-p)**(6-k)
            for k in range(7) if 0 <= k0-k <= 6
        )
        for pi, p in zip(prior, prob, strict=True)
    ]
    first = normalize(weights)
    final = normalize([pi*p**k1*(1-p)**(12-k1) for pi, p in zip(prior, prob, strict=True)])
    return first, final


def lrm_rows(recovered, work, writer, tok):
    directory = recovered / "looped_train_candidates"
    manifest = read(directory / LRM / "lrm_rev02_v1_publication_manifest.json")
    specs = {r["path"]: r for r in manifest["files"]}
    denied = set()
    # Only holdout public-input canonical identities are used for exclusion.
    # No development payload or labels are copied into the release.
    for root in range(4):
        name = f"{LRM}/lrm_rev02_v1/root_{root}/development_outcomes.jsonl.gz"
        pin = specs[name]
        p = work / f"lrm-dev-{root}.gz"
        download(f"https://raw.githubusercontent.com/{LT}/{LT_PIN}/{name}", p, pin["sha256"], 5_000_000)
        with gzip.open(p, "rt") as f:
            for line in f:
                row = json.loads(line)
                denied.add(c.tasks.digest(lrm_observations(row)[3]))
    seen = set()
    report = Counter()
    for model in range(4):
        name = f"{LRM}/lrm_rev02_v1/root_{model}/training_episodes.jsonl.gz"
        path = directory / name
        with gzip.open(path, "rt") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                source = hashlib.sha256(json.dumps(
                    ["lrm-rev02/public-cpu-pilot/v1", "episode-source", model, "train", i],
                    separators=(",", ":"), ensure_ascii=True,
                ).encode("ascii")).hexdigest()
                need(row["source_id"] == source and row["episode_index"] == i, "LRM source identity")
                first, last = lrm_target(row)
                for expected, supplied in ((first, row["target_initial"]), (last, row["target_full"])):
                    need(
                        len(supplied) == 3 and all(
                            abs(float(a)-b) <= 1e-12 for a, b in zip(expected, supplied, strict=True)
                        ), "LRM archived Bayes target differs",
                    )
                report["recomputed"] += 1
                old, slots, new, shape = lrm_observations(row)
                key = c.tasks.digest(shape)
                if key in denied or key in seen:
                    report["held_or_semantic_duplicate"] += 1
                    continue
                seen.add(key)
                # The archived source's hidden true regime never becomes model input.
                public = dict(old=old, corrections=list(zip(slots, new, strict=True)),
                              prior=["9/20", "7/20", "1/5"], match=["19/20", "13/20", "3/20"])
                prompt = (
                    "Twelve old bits contain six uniformly located fair placeholder bits. The remaining bits "
                    "and their six replacements are independent Bernoulli observations given the hidden regime. "
                    "Compute the exact posterior before the correction schedule is revealed and after applying "
                    "the corrections. Return reduced fraction strings in regime order.\n" + c.tasks.canon(public)
                )
                target = c.tasks.canon(dict(initial=[str(x) for x in first], final=[str(x) for x in last]))
                triple = mk_checked(
                    tok, "state", "state/lrm-rev02-bayes", c.tasks.digest(["lrm", shape]),
                    prompt, target, dict(kind="lrm-exact-rational-replay", source_id=source,
                    repository=LT, revision=LT_PIN, path=name, shard_sha256=digest(path)),
                    {"accepted_matches": sum(b for i,b in enumerate(old) if i not in slots)+sum(new)},
                )
                report["admitted"] += int(writer.add(triple))
    return dict(report, denied_semantic_shapes=len(denied))


def lean_candidates(tok, mixed, recovered):
    for row in c.tasks.native_lean(mixed):
        prompt, target = c.render(row)
        root = c.tasks.digest(["lean", c.checked.lean_template(row["public"]["statement"])])
        yield (root, "lean/" + row["family"], " : " + row["public"]["statement"], prompt, target,
               {"kind": "lean-kernel", "source": row["origin"]})
    directory = recovered / "looped_train_candidates" / LBF
    for sub in ("references", "counterfactuals"):
        p = directory / sub / "Proofs.lean"
        manifest = read(p.parent / "manifest.json")
        src = p.read_text()
        blocks = re.findall(
            r"(?ms)^(?:theorem|def) (\w+)(.*?) := (by\n.*?)(?=^(?:theorem |def |#print |end )|\Z)", src
        )
        names = {a for a, _, _ in blocks}
        need(names == set(manifest["declarations"]), "LBF train declaration inventory changed")
        for name, header, proof in blocks:
            header, proof = header.rstrip(), proof.rstrip()
            root = c.tasks.digest(["lbf18", c.checked.lean_template(header)])
            prompt = "Complete this Lean 4 declaration in an environment importing Std.\n" + (
                "theorem candidate" + header + " :="
            )
            try:
                c.tokens(tok, prompt, proof)
            except ValueError:
                # Never cut a proof to fit the production context.
                continue
            yield root, "lean/lbf18-train-" + sub, header, prompt, proof, {
                "kind": "lean-kernel", "repository": LT, "revision": LT_PIN,
                "path": f"{LBF}/{sub}/Proofs.lean", "source_declaration": name,
                "source_sha256": manifest["source_sha256"],
            }


def compile_proofs(candidates, lean, directory):
    directory.mkdir()
    seq = 0
    out = []
    version = subprocess.check_output([str(lean), "--version"], text=True).strip()
    need("4.28.0" in version, "unexpected Lean compiler")
    compiler_sha = digest(lean)

    def invoke(text):
        nonlocal seq
        seq += 1
        src = directory / f"query-{seq:03}.lean"
        log = src.with_suffix(".log")
        src.write_bytes(text.encode())
        r = subprocess.run([str(lean), "-j", "1", str(src)], capture_output=True, timeout=120)
        log.write_bytes(r.stdout + r.stderr)
        return r.returncode, log.read_text(errors="replace"), src, log

    bad, log, _, _ = invoke("import Std\ntheorem negative_control : False := by exact True.intro\n")
    need(bad != 0 and "type mismatch" in log.lower(), "negative control did not fail")

    def check(group):
        text = "import Std\nset_option autoImplicit false\n"
        for root, _, header, _, proof, _ in group:
            need(not re.search(r"\b(sorry|admit|unsafe|axiom|native_decide)\b", proof), "unsafe proof")
            text += f"theorem candidate_{root}{header} := {proof}\n#print axioms candidate_{root}\n"
        code, log, src, path = invoke(text)
        if code and len(group) > 1:
            half = len(group)//2
            check(group[:half])
            check(group[half:])
            return
        audits = {name: sorted(v.strip() for v in vals.split(",") if v.strip()) for name, vals in AX.findall(log)}
        for entry in group:
            root = entry[0]
            axioms = audits.get("candidate_" + root)
            if code == 0 and axioms is not None and set(axioms) <= ALLOWED_AXIOMS:
                out.append((entry, dict(
                    version=version, compiler_sha256=compiler_sha, axioms=axioms,
                    source=src.name, source_sha256=digest(src), log=path.name, log_sha256=digest(path),
                )))
    unique = {}
    for row in candidates:
        unique.setdefault(row[0], row)
    values = list(unique.values())
    for i in range(0, len(values), 8):
        check(values[i:i+8])
    return out, {"candidates": len(values), "accepted": len(out), "rejected": len(values)-len(out),
                 "compiler": version, "negative_control": "type mismatch"}


def general_rows(source, mixed, work, writer, tok, target):
    # The original text eval used math-only LibreTexts roles and twelve named
    # Gutenberg books. The entire math source family universe is held back,
    # since the original private role manifests are missing.
    groups = mt.Groups()
    blocked_keys = set()
    eligible = []
    reject = Counter()
    historical = Overlap()
    cache = work / "gutenberg"
    fetch_books(cache)
    for doc in pins():
        historical.add(body_bytes(doc, cache)[0].decode())
        surname, given = doc["author"].split(",", 1)
        person = mt.canonical(re.sub(r"\([^)]*\)", "", given) + " " + surname)
        blocked_keys.update(("by:" + person, "person:" + person, "title:" + mt.canonical(doc["title"])))
    prior = defaultdict(list)
    for values in native_bytes(mixed).values():
        for v in values:
            for b in (v, ti.folded(v)):
                if len(b) >= 12:
                    prior[b[:8]].append(b)
    print("Reading and conservatively screening pinned LibreTexts source", flush=True)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            m = row.get("metadata") or {}
            subject = mt.domain(row)
            if not mt.metadata_ok(row):
                reject["missing_provenance"] += 1
                continue
            ks = mt.keys(row)
            groups.connect(ks)
            if subject in ("math", "stats"):
                blocked_keys.update(ks)
                reject["historical_math_family"] += 1
                # Protect text reused in another domain against the whole
                # historical math candidate universe, not the missing panel.
                if ti.eligibility(row) is None:
                    historical.add(row["text"])
                continue
            if m.get("license") != ti.LICENSE or mt.MIXED_RIGHTS.search(row["text"]):
                reject["rights_or_exception"] += 1
                continue
            if m["author"].strip().lower() in {"unknown", "anonymous", "various", "libretexts", "multiple authors"}:
                reject["unspecified_author"] += 1
                continue
            if not subject or not 700 <= len(row["text"].encode()) <= 65536:
                reject["length_or_domain"] += 1
                continue
            eligible.append((subject, row, min(ks)))
    blocked = {groups.find(k) for k in blocked_keys}
    queues = defaultdict(list)
    for subject, row, key in eligible:
        family = groups.find(key)
        if family in blocked:
            reject["connected_historical_group"] += 1
            continue
        queues[subject, row["metadata"]["author"]].append((row, family))
    queues = {k: deque(sorted(v, key=lambda x: c.tasks.digest(x[0]["id"]))) for k,v in queues.items()}
    del eligible
    author, book, domain, seen = Counter(), Counter(), Counter(), set()
    while queues and writer.positions < target:
        key = min(queues, key=lambda x: (domain[x[0]], author[x[1]], x))
        subject, a = key
        row, family = queues[key].popleft()
        if not queues[key]:
            del queues[key]
        m = row["metadata"]
        b = m["book_url"]
        if row["id"] in seen or author[a] >= 200_000 or book[b] >= 120_000:
            reject["source_duplicate_or_cap"] += 1
            continue
        original = row["text"]
        # One complete contiguous passage per original source. A long section
        # may be divided only at sentence boundaries, then one span is selected.
        ids_full = tok.encode(original, add_special_tokens=False)
        if len(ids_full.ids) <= 2048:
            text = original
        else:
            end = ids_full.offsets[2047][1]
            stops = [v.end() for v in mt.SENTENCE_END.finditer(original[:end])]
            text = original[:stops[-1]] if stops else ""
        if len(text) < 500 or ti.historical_overlap(text.encode(), prior):
            reject["short_or_native_window"] += 1
            continue
        norm, sketch = ti.duplicate_signature(text)
        syms = symbols(text)
        if why := historical.check(norm, sketch, syms):
            reject[why] += 1
            continue
        enc = tok.encode(text, add_special_tokens=False)
        ids = enc.ids
        if len(ids) > 2048 or len(ids) < 128 or tok.decode(ids, skip_special_tokens=False) != text:
            reject["length_or_roundtrip"] += 1
            continue
        if author[a]+len(ids)>200_000 or book[b]+len(ids)>120_000:
            reject["source_cap"] += 1
            continue
        root = "libretexts/" + ti.sha(row["id"].encode())
        family = "libretexts/replacement-v2/" + family
        record = dict(split="train", semantic_key=root, family=family, text=text, rights=ti.LICENSE,
                      attribution=dict(author=a, title=m["title"], original_url=row["id"], book_url=b,
                      upstream_line=m["provenance"], revision=ti.REV, source_sha256=ti.SOURCE,
                      original_text_sha256=ti.sha(original.encode()), source_start_byte=0,
                      source_end_byte=len(text.encode()), original_text_bytes=len(original.encode())))
        receipt = dict(schema=VERSION, split="train", root=root, accepted=True,
                       check="source-rights-and-contiguous-passage", record_sha256=c.tasks.digest(record),
                       tokenizer_sha256=TOKENIZER_SHA256)
        item = dict(root=root, family=family, split="train", lane="general", presentation=1,
                    objective="next_token", token_ids=ids, target_mask=[False]+[True]*(len(ids)-1))
        if writer.add((record, receipt, item)):
            seen.add(row["id"])
            historical.add_prepared(norm, sketch, syms)
            author[a] += len(ids)
            book[b] += len(ids)
            domain[subject] += len(ids)
    need(writer.positions >= target, f"general text short: {writer.positions} / {target}")
    return dict(rejected=dict(reject), authors=len(author), books=len(book), subject_positions=dict(domain),
                historical_books=len(pins()), blocked_historical_components=len(blocked),
                source_revision=ti.REV, source_sha256=ti.SOURCE, rights=ti.LICENSE)


def verify_release(base, expected, lean=None):
    ix = read(base / "index.json")
    need(digest(base / "index.json") == expected and ix["schema"] == VERSION, "snapshot identity")
    tok = load_tokenizer(base / "general/tokenizer.json")
    counts = Counter()
    global_roots = set()
    global_ids = set()
    mr.tasks.SEED = RESEARCH_SEED
    hashes = mr.code_hashes()
    for lane, spec in ix["lanes"].items():
        folder = base / lane
        for f in ("records.jsonl", "receipts.jsonl", "rows.jsonl"):
            need(digest(folder / f) == spec["source_members"][f], "source member changed")
        with ExitStack() as stack:
            fs = [stack.enter_context((folder / n).open("rb")) for n in (
                "records.jsonl", "receipts.jsonl", "rows.jsonl",
            )]
            for lines in zip(*fs, strict=True):
                rec, receipt, item = (json.loads(v) for v in lines)
                require_train(item, lane, read(RECIPE)["stream"]["subtasks"], DENIED)
                root = item["root"]
                seq = c.tasks.digest(item["token_ids"])
                need(root not in global_roots and seq not in global_ids, "duplicate exported row")
                global_roots.add(root)
                global_ids.add(seq)
                need(receipt["record_sha256"] == c.tasks.digest(rec), "receipt record changed")
                if lane == "general":
                    ids = tok.encode(rec["text"], add_special_tokens=False).ids
                    need(rec["rights"] == ti.LICENSE and ids == item["token_ids"], "text tokens or rights")
                    need(item["target_mask"] == [False]+[True]*(len(ids)-1), "text mask")
                else:
                    need(item["receipt_sha256"] == c.tasks.digest(receipt), "receipt hash changed")
                    ids, mask, start = c.tokens(tok, rec["prompt"], rec["target"])
                    need((ids, mask, start) == (item["token_ids"], item["target_mask"], item["response_start"]),
                         "checked tokens changed")
                    kind = rec["origin"]["kind"]
                    if lane == "custom_research":
                        rebuilt = research_candidate(tok, rec["origin"]["serial"], hashes)
                        need(rebuilt == (rec, receipt, item), "research replay differs")
                    elif kind == "independent-public-math":
                        rebuilt = math_candidate(tok, rec["skill"], rec["origin"]["ordinal"])
                        need(rebuilt == (rec, receipt, item), "mathematical replay differs")
                    elif kind == "lean-kernel":
                        wit = rec["witness"]
                        for label in ("source", "log"):
                            need(digest(base / "kernel" / wit[label]) == wit[label+"_sha256"], "kernel receipt")
                        need(set(wit["axioms"]) <= ALLOWED_AXIOMS, "Lean axiom set")
                    elif kind == "lrm-exact-rational-replay":
                        q = json.loads(rec["prompt"].rsplit("\n", 1)[-1])
                        slots, new = map(list, zip(*q["corrections"], strict=True))
                        a, b = lrm_target(dict(old=q["old"], correction_slots=slots, replacements=new))
                        need(rec["target"] == c.tasks.canon(dict(initial=list(map(str,a)), final=list(map(str,b)))),
                             "Bayesian target replay differs")
                    else:
                        raise ValueError(f"unrecognized verification scope {kind}")
                counts[lane] += 1
        need(counts[lane] == spec["rows"], "row count")
    result = replay(base, expected)
    result["verified_source_rows"] = dict(counts)
    result["production_launch_authorized"] = False
    return result


def build(out, lean):
    need(digest(RECIPE) == RECIPE_SHA, "frozen recipe changed")
    cfg = read(RECIPE)
    target = quotas(cfg)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as folder:
        work = Path(folder)
        recovered = work / "recovered"
        recovered.mkdir()
        native = recover_native(work, recovered)
        looped = recover_looped(recovered)
        mixed = work / "mixed"
        tok_path = work / "tokenizer.json"
        download(TOKEN_URL, tok_path, TOKENIZER_SHA256, 20_000_000)
        tok = load_tokenizer(tok_path)
        source = work / "libretexts.json.gz"
        download(TEXT_URL, source, ti.SOURCE)
        stage = work / "release"
        stage.mkdir()
        roots, seqs = set(), set()
        lanes = {}
        reports = {}
        for lane in ("custom_research", "verified", "procedural", "failure_directed", "general"):
            w = Writer(stage, lane, tok_path, cfg, roots, seqs)
            if lane == "custom_research":
                fill_research(w, tok, target[lane])
            elif lane == "verified":
                accepted, reports["kernel"] = compile_proofs(
                    lean_candidates(tok, mixed, recovered), lean, stage / "kernel"
                )
                for (root, fam, _, prompt, proof, origin), witness in accepted:
                    w.add(mk_checked(tok, "lean", fam, root, prompt, proof, origin, witness))
                fill_math(w, tok, target[lane], ("composition",))
            elif lane == "procedural":
                reports["lrm"] = lrm_rows(recovered, work, w, tok)
                fill_math(w, tok, target[lane], ("state", "operator"))
            elif lane == "failure_directed":
                fill_math(w, tok, target[lane], ("repair",))
            else:
                reports["general"] = general_rows(source, mixed, work, w, tok, target[lane])
            lanes[lane] = w.finish(target[lane])
            lanes[lane]["source_members"] = {
                n: digest(w.folder / n) for n in ("records.jsonl", "receipts.jsonl", "rows.jsonl")
            }
            print(json.dumps({"lane": lane, **lanes[lane]}, sort_keys=True), flush=True)
        index = dict(
            schema=VERSION, recipe_sha256=RECIPE_SHA, supersedes_missing_index=OLD_INDEX,
            target_by_lane=cfg["stream"]["lane_positions"], first_checkpoint=target,
            lanes=lanes, total_positions=sum(x["positions"] for x in lanes.values()),
            train_roots=len(roots), seed=SEED, research_seed=RESEARCH_SEED,
        )
        c.dump(stage / "index.json", index)
        expected = digest(stage / "index.json")
        c.dump(stage / "ADMISSION.json", dict(
            schema=VERSION, recipe_sha256=RECIPE_SHA, index_sha256=expected,
            source_recovery={"native": native, "looped": looped}, reports=reports,
            known_excluded_families=sorted(DENIED), train_only=True,
            original_10m_recovered=False, production_launch_authorized=False,
            limitations=[
                "Historical private panels are absent: exclude entire old math domain and pinned Gutenberg books; "
                "lexical and connected-source screens are not a proof of arbitrary semantic disjointness.",
                "This new pilot requires new prospective held evaluation custody before quality decisions.",
                "Some recovered candidate datasets remain staged for later admission; no quota credit for those.",
                "Source rights rely on CC BY 4.0 per-section metadata and visible exception screening.",
            ],
        ))
        first = verify_release(stage, expected)
        c.dump(stage / "REPLAY.json", first)
        # Attribution is inside general/records.jsonl; attach the complete
        # standard license text and the pinned tokenizer notice as well.
        # License texts are committed, so a late network error cannot discard a completed replay.
        for name, blob in (
            ("CC-BY-4.0.txt", "13ca539f377dc705af32b8d2ce89262298ea2f06"),
            ("Apache-2.0.txt", "137069b823873b8bcf42979bcf8e9371052d26a2"),
        ):
            license_file = HERE / "licenses" / name
            raw_license = license_file.read_bytes()
            actual = hashlib.sha1(f"blob {len(raw_license)}\0".encode() + raw_license).hexdigest()
            need(actual == blob, f"pinned license text changed: {name}")
            shutil.copyfile(license_file, stage / name)
        (stage / "TOKENIZER-NOTICE.md").write_text(
            f"SmolLM2 tokenizer, Apache-2.0 declaration at the pinned upstream model revision.\n{TOKEN_URL}\n"
            f"Tokenizer SHA-256: {TOKENIZER_SHA256}\n", encoding="utf8",
        )
        c.dump(stage / "MEMBERS.json", {
            p.relative_to(stage).as_posix(): dict(sha256=digest(p), bytes=p.stat().st_size)
            for p in sorted(stage.rglob("*")) if p.is_file()
        })
        stage.rename(out)
    return first


def pack(base, output):
    need(not output.exists(), "output exists")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(base.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(base).as_posix())
    need(output.stat().st_size < 95_000_000, "archive exceeds Git publication limit")
    with tempfile.TemporaryDirectory() as folder:
        dest = Path(folder)
        with zipfile.ZipFile(output) as z:
            need(z.testzip() is None, "archive CRC failure")
            for name in z.namelist():
                need((dest / name).resolve().is_relative_to(dest), "unsafe member")
            z.extractall(dest)
        members = read(dest / "MEMBERS.json")
        need(set(members) | {"MEMBERS.json"} == {
            p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()
        }, "archive inventory")
        for n, spec in members.items():
            need(digest(dest / n) == spec["sha256"] and (dest / n).stat().st_size == spec["bytes"],
                 "fresh extraction differs")
        again = verify_release(dest, digest(dest / "index.json"))
        before = read(dest / "REPLAY.json")
        for key in ("steps", "saved_cursor", "terminal_cursor", "total_targets", "full_positions", "lanes"):
            need(again[key] == before[key], "fresh replay differs: " + key)
        return dict(
            schema=VERSION, archive=output.name, archive_sha256=digest(output), archive_bytes=output.stat().st_size,
            index_sha256=digest(dest / "index.json"), replay_sha256=digest(dest / "REPLAY.json"),
            recipe_sha256=RECIPE_SHA, full_positions=again["full_positions"],
            supervised_labels=again["total_targets"], update_groups=again["update_groups"],
            restart_position=again["restart_position"], terminal_cursor=again["terminal_cursor"]["sha256"],
            original_10m_recovered=False, fresh_replay_pass=True,
            train_only=True, production_launch_authorized=False, accelerator_credits=0,
            rows=again["verified_source_rows"], lane_positions={k:v["processed"] for k,v in again["lanes"].items()},
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "verify", "pack"))
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--lean", type=Path)
    p.add_argument("--archive", type=Path)
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    if a.action == "build":
        if a.lean is None:
            p.error("build requires pinned Lean")
        result = build(a.base.resolve(), a.lean.resolve())
        print(json.dumps({k:v for k,v in result.items() if k not in {"steps","saved_cursor","terminal_cursor"}}))
    elif a.action == "verify":
        result = verify_release(a.base.resolve(), digest(a.base / "index.json"))
        print(json.dumps({"positions":result["full_positions"],"labels":result["total_targets"]}))
    else:
        if a.archive is None or a.release is None:
            p.error("pack requires --archive and --release")
        release = pack(a.base.resolve(), a.archive.resolve())
        c.dump(a.release, release)
        print(json.dumps(release, sort_keys=True))


if __name__ == "__main__":
    main()
