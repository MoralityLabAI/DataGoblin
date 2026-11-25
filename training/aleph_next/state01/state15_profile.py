"""Measure cold-process singleton check costs on public STATE-15 archived proof anchors."""

import argparse
import json
import statistics
from pathlib import Path

from state15_kernel import compile_query, digest, proof_axioms, toolchain
from state15_replay import EXPECTED, HEADER, NAME


def declarations(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"prepared source changed: {path}")
    text = "\n".join(line for line in data.decode().splitlines() if not line.startswith("#print axioms "))
    starts = list(NAME.finditer(text))
    return [
        (m[1], text[m.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)].rstrip() + "\n")
        for i, m in enumerate(starts)
    ]


def measure(prepared, out, executable):
    chain = toolchain(executable)
    data = json.loads((prepared / "PREPARED.json").read_bytes())
    out.mkdir(parents=True, exist_ok=True)
    runs = []

    def check(stem, source, kind, stage, expected=()):
        path = out / (stem + ".lean")
        path.write_bytes(source.encode())
        cost, log = compile_query(chain["executable"], path)
        valid_axioms, axioms = proof_axioms(log, expected)
        if kind == "rejected_transport":
            passed = not cost["timed_out"] and cost["exit_code"] != 0 and "type mismatch" in log.lower()
        else:
            passed = not cost["timed_out"] and cost["exit_code"] == 0 and valid_axioms
        runs.append(dict(query=stem, stage=stage, kind=kind, passed=passed, axioms=axioms, **cost))

    # Histograms retain every attempt. The first invocation starts a fresh process, like each
    # later invocation; OS/DLL page-cache state can still differ and is explicitly unmeasured.
    for i in range(3):
        check(f"empty_{i}", HEADER, "empty_cli", None)
    for stage in EXPECTED:
        terms = declarations(prepared / f"STATE{stage}_REPLAY.lean", data["prepared"][str(stage)]["source_sha256"])
        terms.sort(key=lambda t: (len(t[1].encode()), t[0]))
        for q in (0.1, 0.5, 0.9):
            name, term = terms[int((len(terms) - 1) * q)]
            check(
                f"s{stage}_q{int(q * 100)}_{name}",
                HEADER + term + "\n#print axioms " + name + "\n",
                "accepted_proof",
                stage,
                (name,),
            )
    negative = json.loads((prepared / "NEGATIVES.json").read_bytes())
    for stage in (10, 11):
        cases = [r for r in negative if r["stage"] == stage]
        for row in (cases[0], cases[-1]):
            prefix = HEADER + row["context"]
            name = f"s{stage}_{row['id']}"
            check(name + "_context", prefix, "valid_context", stage)
            check(name + "_invalid", prefix + "\n" + row["invalid"], "rejected_transport", stage)
    summary = {}
    for kind in sorted({r["kind"] for r in runs}):
        rs = [r for r in runs if r["kind"] == kind]
        summary[kind] = {"queries": len(rs), "passed": sum(r["passed"] for r in rs)}
        for unit in ("wall_seconds", "cpu_seconds"):
            values = [r[unit] for r in rs]
            summary[kind][unit] = {
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.mean(values),
                "max": max(values),
                "sum": sum(values),
            }
    result = {
        "schema": "datagoblin.state15.singleton-profile.v1",
        "toolchain": chain,
        "sample": (
            "three archived positive declarations per stage at empirical 10/50/90 percentiles by UTF-8 source bytes, "
            "plus first and last registered controls per negative-bearing stage and three empty invocations"
        ),
        "runs": runs,
        "summary": summary,
        "all_expected_outcomes": all(r["passed"] for r in runs),
    }
    (out / "SINGLETON_RECEIPTS.json").write_bytes((json.dumps(result, indent=2) + "\n").encode())
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--lean", type=Path)
    args = p.parse_args()
    result = measure(args.prepared, args.out, args.lean)
    print(json.dumps(result["summary"], indent=2))
    raise SystemExit(0 if result["all_expected_outcomes"] else 1)
