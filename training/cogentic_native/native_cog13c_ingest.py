"""Audit the frozen public COG-13C native artifact; export TRAIN-only rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE_SHA = "d99bc6a95d4ea31a132b408cd653644e9680a167fa87aca3b722522af0ab5aea"
SOURCE_COMMIT = "a50e53f2a527d8e9464538dbc0ae2b2603c90bee"
SOURCE_RUN = 45852214025
SOURCE_ARTIFACT = 15136527625
AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
ARMS = {"native_rr1", "native_rr2", "native_block", "dense_block", "dense_adaptive", "rank4_adaptive", "rank4q_adaptive"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def key(value):
    return sha(canonical(value).encode())


def dot(a, b):
    need(len(a) == len(b), "inconsistent factor shape")
    return sum(x * y for x, y in zip(a, b, strict=True))


def packet_audit(event, model, factors, mode):
    packet = event["transport"]
    if mode == "native":
        need(packet is None, "native policy carried a learned packet")
        return 0
    data = bytes(packet["packet"])
    version, count, dims, bits = data[:4]
    expected = (32, 32) if mode == "dense" else (4, 4 if mode == "rank4q" else 32)
    need(version == 1 and (dims, bits) == expected and count == len(event["options"]), "packet header")
    need(len(data) == 4 + (count * dims * bits + 7) // 8, "packet length")
    need(packet["bytes"] == len(data) + 1, "transport charge")
    if bits == 32:
        values = struct.unpack(f"<{count*dims}f", data[4:])
    else:
        codes = [(data[4 + j // 2] >> (4 * (j % 2))) & 15 for j in range(count * dims)]
        need(all(x < 15 for x in codes), "quantizer reserved code")
        values = [(x - 7) * factors["scale"][j % dims] for j, x in enumerate(codes)]
    scores = []
    for i in range(count):
        x = values[i * dims:(i + 1) * dims]
        pre = ([dot(w, x) for w in model["W"]] if mode == "dense" else
               [sum(u[j] * x[k] for k, u in enumerate(factors["U"])) for j in range(len(model["b"]))])
        h = [0.0] * len(model["b"])
        for _ in range(3):
            h = [math.tanh(pre[j] + dot(r, h) + model["b"][j]) for j, r in enumerate(model["R"])]
        scores.append(dot(model["v"], h) + model["o"])
    need(len(scores) == len(packet["scores"]) and all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
         for a, b in zip(scores, packet["scores"], strict=True)), "decoded scores differ")
    order = sorted(range(count), key=lambda i: (-packet["scores"][i], i))
    need(order == packet["order"] and order[0] == packet["reply"], "ranked reply")
    need(event["options"][order[0]] == event["action"], "action binding")
    return len(data) + 1


def audit_archive(archive):
    need(sha(archive.read_bytes()) == ARCHIVE_SHA, "unregistered archive SHA-256")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        need(len(names) == len(set(names)), "duplicate archive members")
        def read(name):
            info = z.getinfo(name)
            need(info.file_size <= 64 * 1024 * 1024, "oversized member")
            return z.read(info)
        def obj(name):
            return json.loads(read(name))
        def lines(name):
            return [json.loads(line) for line in read(name).splitlines() if line]
        manifest = obj("export_manifest.json")
        expected_files = {"result.json", "task_registry.json", "tactic_train.jsonl", "allocator_train.jsonl"}
        need(set(manifest["hashes"]) == expected_files, "manifest file set")
        for name, expected in manifest["hashes"].items():
            need(sha(read(name)) == expected, "manifest hash mismatch: " + name)
        result = obj("result.json")
        cases = result["cases"]
        need(cases == obj("task_registry.json") and len(cases) == 24, "task registry")
        registry = {c["id"]: c for c in cases}
        need(len(registry) == 24 and len({c["statement"] for c in cases}) == 24, "duplicate task")
        train = {c["id"] for c in cases if c["split"] == "train"}
        need(len(train) == 8 and Counter(c["split"] for c in cases) == {"train": 8, "ordinary": 8, "deeper": 8},
             "split registry")
        raw = defaultdict(Counter)
        initial = {}
        cert_logs = {}
        for proc in obj("processes.json"):
            folder = Path(proc["folder"]).name
            need(re.fullmatch(r"proc_\d{5}", folder) is not None, "unsafe process identity")
            need(sha(read(folder + "/stdout.log")) == proc["logSha256"], "process log hash")
            if proc["kind"] != "probe_server":
                cert_logs[folder] = proc
                continue
            startup = folder + "/response-0.json"
            if startup not in names:
                continue
            start = obj(startup)["state"]["goals"]
            rid = proc["root"]
            need(rid in registry and (rid not in initial or initial[rid] == start), "root goal changed")
            initial[rid] = start
            for path in names:
                if not re.fullmatch(re.escape(folder) + r"/response-[1-9]\d*\.json", path):
                    continue
                reply = obj(path)
                request = obj(f'{folder}/request-{reply["id"]}.json')
                need(request["id"] == reply["id"] and request["stop"] is False, "mailbox request binding")
                raw[(proc["phase"], rid)][key([request["actions"], reply["result"]])] += 1
        tactic_rows = []
        reached = {}
        for row in result["teacher"]["rows"]:
            rid = row["root"]
            need(rid in train and row["split"] == "train", "heldout teacher row")
            response = row["receipt"]
            probe = key([row["prefix"] + [row["action"]], response])
            pool = raw[("teacher_" + rid, rid)]
            need(pool[probe] > 0, "teacher reply absent in raw mailbox")
            pool[probe] -= 1
            progress = int(response["accepted"] and response["goals"] != row["goals"])
            closed = bool(response["accepted"] and response["goals"] == [])
            need(row["target"] == progress and row["closed"] == closed, "teacher label arithmetic")
            expected_goals = initial[rid] if not row["prefix"] else reached.get((rid, tuple(row["prefix"])))
            need(row["goals"] == expected_goals, "teacher pre-action goal custody")
            if response["accepted"]:
                reached[(rid, tuple(row["prefix"] + [row["action"]]))] = response["goals"]
            tactic_rows.append({"root": rid, "split": "train", "input": {"goals": row["goals"],
                               "candidate": row["action"]}, "target": {"progress": progress, "closed": closed},
                               "prefix": row["prefix"], "source": "native_lean_probe"})
        need(tactic_rows == lines("tactic_train.jsonl") and len(tactic_rows) == manifest["tacticRows"] == 224,
             "tactic export mismatch")
        need(sum(r["target"]["progress"] for r in tactic_rows) == result["fit"]["positives"] == 82,
             "positive tally")
        for pool in raw.values():
            need(all(v >= 0 for v in pool.values()), "duplicate raw response credit")
        summary = defaultdict(lambda: defaultdict(lambda: Counter()))
        certified = set()
        packets = probes = cert_count = 0
        allocator_rows = []
        all_runs = result["training"] + result["runs"]
        need(len(result["training"]) == 2 and len(result["runs"]) == 28, "portfolio registry")
        tags = set()
        for run in all_runs:
            arm, tag = run["arm"], run["tag"]
            need(tag not in tags, "duplicate portfolio")
            tags.add(tag)
            is_train = tag.startswith("train")
            cohort = run["roots"]
            need(len(cohort) == 4 and len({r["c"]["id"] for r in cohort}) == 4, "portfolio roots")
            split = cohort[0]["c"]["split"]
            need(is_train == (split == "train"), "portfolio split")
            need(arm["name"] in (ARMS if not is_train else {"train_dense"}), "arm registry")
            events = [e for e in run["events"] if e["kind"] == "probe"]
            need(len(events) == run["calls"] <= arm["probes"] and [e["portfolioCall"] for e in events] ==
                 list(range(1, len(events) + 1)), "probe budget")
            root_count = byte_count = 0
            for root in cohort:
                c, rid = root["c"], root["c"]["id"]
                need(c == registry[rid] and c["split"] == split, "root identity")
                own = [e for e in events if e["root"] == rid]
                need(own == root["events"] and len(own) == root["attempts"], "root probe sequence")
                root_bytes = 0
                for event in own:
                    probes += 1
                    need(event["prefix"][-1] == event["action"] and event["action"] in event["options"], "action")
                    need(not event["reply"].get("unknown"), "unverified native response")
                    pool = raw[(tag, rid)]
                    probe = key([event["prefix"], event["reply"]])
                    need(pool[probe] > 0, "evaluation reply absent in raw mailbox")
                    pool[probe] -= 1
                    n = packet_audit(event, result["fit"]["model"], result["factors"], arm["router"])
                    packets += int(n > 0)
                    root_bytes += n
                need(root_bytes == root["bytes"], "root bytes")
                byte_count += root_bytes
                certs = root["certificates"]
                if root["solved"]:
                    need(len(certs) == 2 and own and own[-1]["reply"].get("goals") == [] and
                         all(x["accepted"] for x in certs) and certs[0]["sourceHash"] == certs[1]["sourceHash"],
                         "uncertified solved root")
                    credit_events = [e for e in run["events"] if e["kind"] == "certificate" and e["root"] == rid]
                    need(len(credit_events) == 2 and credit_events[-1]["elapsedMs"] <= arm["wallMs"], "late certificate")
                    root_count += 1
                    certified.add((tag, rid))
                for cert in certs:
                    cert_count += 1
                    folder = Path(cert["sourceFile"]).parent.name
                    proc = cert_logs.pop(folder)
                    src, log = read(folder + "/Certificate.lean"), read(folder + "/stdout.log")
                    need(proc["root"] == rid and proc["phase"] == tag and sha(src) == cert["sourceHash"] and
                         sha(log) == cert["logSha256"] and b"import ProbeServer" not in src, "certificate custody")
                    expected = ("import Mathlib\nset_option maxHeartbeats 1000000\n" + f'theorem {rid} {c["statement"]} := by\n' +
                                "".join("  " + s + "\n" for s in own[-1]["prefix"]) + f"#print axioms {rid}\n")
                    need(src.decode() == expected, "certificate not closed tactic script")
                    if cert["accepted"]:
                        need(proc["exitCode"] == cert["exitCode"] == 0 and set(cert["axioms"]) <= AXIOMS and
                             not re.search(rb"sorryAx|error(?:\([^\n]*\))?:|declaration uses 'sorry'", log), "bad proof")
                if is_train and root["solved"]:
                    examples = root["examples"]
                    need(len(examples) == len(own), "allocator trajectory length")
                    previous = Counter()
                    for i, (x, event) in enumerate(zip(examples, own, strict=True)):
                        prefix = tuple(event["prefix"][:-1])
                        goals = event["before"]
                        char_count = len(" ".join(goals).encode("utf-16-le")) // 2
                        expected = [1, min(1, len(goals) / 8), min(1, char_count / 512),
                                    min(1, len(prefix) / 8), min(1, previous[prefix] / 16)]
                        need(x == expected and all(math.isfinite(v) for v in x), "allocator feature custody")
                        previous[prefix] += 1
                        allocator_rows.append({"root": rid, "split": "train", "features": x,
                                               "remaining": len(examples) - i, "weight": 1 / len(examples)})
            need(root_count == run["solved"] and byte_count == run["onlineBytes"], "portfolio totals")
            if not is_train:
                a = summary[split][arm["name"]]
                a.update({"roots": len(cohort), "solved": root_count, "bytes": byte_count, "calls": len(events)})
        need(not cert_logs and cert_count == 128, "certificate count")
        need(allocator_rows == lines("allocator_train.jsonl") == result["allocator"]["rows"] and
             len(allocator_rows) == manifest["allocatorRows"] == 44, "allocator export mismatch")
        need(all(not any(v for v in pool.values()) for pool in raw.values()), "unaccounted native replies")
        for split, arms in summary.items():
            need(set(arms) == ARMS, "missing arms")
            for name, a in arms.items():
                b = result["summary"][split][name]
                need((a["roots"], a["solved"], a["bytes"], a["calls"]) ==
                     (b["roots"], b["solved"], b["onlineBytes"], b["calls"]) and a["roots"] == 8, "readout summary")
        return {"tactic": tactic_rows, "allocator": allocator_rows, "registry": registry,
                "audit": {"passed": True, "probeRows": probes, "decodedPackets": packets,
                          "certificateProcesses": cert_count, "certifiedEpisodes": len(certified),
                          "tacticTrainRows": len(tactic_rows), "allocatorTrainRows": len(allocator_rows),
                          "summary": summary, "archiveSha256": ARCHIVE_SHA}}


def export(archive, out):
    audited = audit_archive(archive)
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for kind in ("tactic", "allocator"):
        rows = []
        for i, r in enumerate(audited[kind]):
            c = audited["registry"][r["root"]]
            inp = r["input"] if kind == "tactic" else {"features": r["features"]}
            target = r["target"] if kind == "tactic" else {"residualVerifierCalls": r["remaining"]}
            row = {"id": key([ARCHIVE_SHA, kind, i, r]), "root": key([SOURCE_COMMIT, c["id"], c["statement"]]),
                   "split": "train", "family": c["family"], "input": inp, "target": target,
                   "source": "cog13c_native_lean", "source_root": c["id"], "source_commit": SOURCE_COMMIT,
                   "source_archive_sha256": ARCHIVE_SHA, "source_run": SOURCE_RUN,
                   "source_artifact": SOURCE_ARTIFACT, "loss": {"boundary": "response_only", "field": "target", "eligible": True},
                   "verification_kind": "archived_native_probe_progress" if kind == "tactic" else "certified_trajectory_residual"}
            if kind == "allocator":
                row["weight"] = r["weight"]
            rows.append(row)
        data = ("\n".join(canonical(row) for row in rows) + "\n").encode()
        name = "tactic_outcomes.jsonl" if kind == "tactic" else "allocator_residuals.jsonl"
        (out / name).write_bytes(data)
        hashes[name] = sha(data)
    manifest = {"schema": "cog13c.datagoblin_import.v1", "source": "MoralityLabAI/HybridTRMLDT",
                "source_commit": SOURCE_COMMIT, "archive_sha256": ARCHIVE_SHA, "audit": audited["audit"], "hashes": hashes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.archive, args.out), indent=2))
