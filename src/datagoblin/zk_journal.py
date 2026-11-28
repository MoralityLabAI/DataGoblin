"""Proof-bound public output journal for the frontier circuit.

The journal packs bounded output digits injectively, not as a hiding hash.
Claims must be verified against a relying-party-owned plan before admission.
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datagoblin import zk_frontier as frontier
from datagoblin.zk_fabric import hex_digest, merkle_root
from datagoblin.zk_replay import Plan, canonical, digest, require
from datagoblin.zk_schedule import equal, next_state, product, times, total
from datagoblin.zk_sigma import Circuit, const

LIMB_BYTES = 16
LIMB_BOUND = 1 << (8 * LIMB_BYTES)
MAX_PROOF_BYTES = 64_000_000


def limb_count(plan: Plan) -> int:
    frontier.layout(plan)
    n, w = len(plan.jobs), len(plan.jobs[0].source)
    return (n*w + LIMB_BYTES-1) // LIMB_BYTES


def initial(plan: Plan) -> tuple[int, ...]:
    return (*frontier.initial(plan), *((0,) * limb_count(plan)))


def unpack(plan: Plan, words: tuple[int, ...], count: int) -> tuple[tuple[int, ...], ...]:
    """Decode exactly count worker outputs and require zero padding."""
    w = len(plan.jobs[0].source)
    require(type(count) is int and 0 <= count <= len(plan.jobs), "journal count")
    require(len(words) == limb_count(plan), "journal length")
    require(all(type(v) is int and 0 <= v < LIMB_BOUND for v in words), "journal limb range")
    raw = b"".join(v.to_bytes(LIMB_BYTES, "little") for v in words)
    require(all(v <= 254 for v in raw[:count*w]), "invalid output digit")
    require(not any(raw[count*w:]), "nonzero journal padding")
    return tuple(tuple(v-127 for v in raw[j*w:(j+1)*w]) for j in range(count))


def compile_step(plan: Plan, state: tuple[int, ...]) -> dict[str, Any]:
    """Append the constrained output, using the already-constrained worker selector."""
    limbs = limb_count(plan)
    old, words = state[:-limbs], state[-limbs:]
    base = frontier.compile_step(plan, old)
    unpack(plan, words, old[0])
    c = Circuit(values=list(base["witness"]))
    incoming = [c.alloc(v, True) for v in words]
    sels, h = base["worker_selectors"], base["worker_output"]
    w = len(h)
    outgoing = []
    for limb in range(limbs):
        contributions = []
        for i, value in enumerate(h):
            # A position is selected only by the original graph's constrained one-hot PC.
            coefficient = total([
                times(s, 1 << (8 * ((j*w+i) % LIMB_BYTES)))
                for j, s in enumerate(sels) if (j*w+i) // LIMB_BYTES == limb
            ])
            contributions.append(product(c, coefficient, value + const(127)))
        new = incoming[limb] + total(contributions)
        out = c.alloc(c.value(new), True)
        equal(c, out, new)
        outgoing.append(out)
    return {
        "format": "zk-dg-frontier-journal-r1cs-v1",
        "state_in": [*base["state_in"], *(e[0][0] for e in incoming)],
        "state_out": [*base["state_out"], *(e[0][0] for e in outgoing)],
        "constraints": [*base["constraints"], *((e, const(1), ()) for e in c.linear), *c.products],
        "witness": c.values,
    }


def claimed_outputs(plan: Plan, state: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Structural decoding only. A syntactically valid claim is not a proof."""
    require(all(type(v) is int for v in state), "state integer encoding")
    limbs = limb_count(plan)
    output = frontier.terminal(plan, state[:-limbs])
    journal = unpack(plan, state[-limbs:], len(plan.jobs))
    require(journal[-1] == output, "journal sink mismatch")
    return journal


def schedule(plan: Plan) -> list[dict[str, Any]]:
    state = initial(plan)
    rows = []
    for _ in plan.jobs:
        row = compile_step(plan, state)
        rows.append(row)
        state = next_state(row)
    claimed_outputs(plan, state)
    return rows


def authorized_statement(plan: Plan, claim: tuple[int, ...]) -> dict[str, Any]:
    """Rebuild the relation from an authorized plan and a structurally valid claim.

    Only the first step is synthesized to obtain the fixed shape. The claimed
    output is not independently computed or treated as true by this function.
    """
    claimed_outputs(plan, claim)
    z0 = initial(plan)
    shape = compile_step(plan, z0)
    return {
        "format": "zk-dg-nova-statement-v1",
        "steps": len(plan.jobs),
        "wires": len(shape["witness"]),
        "state_in": shape["state_in"],
        "state_out": shape["state_out"],
        "constraints": [[[[i, str(a)] for i, a in expr] for expr in row] for row in shape["constraints"]],
        "z0": [str(v) for v in z0],
        "zn": [str(v) for v in claim],
    }


def native_bundle(plan: Plan) -> tuple[dict[str, Any], list[list[str]]]:
    rows = schedule(plan)
    shape = rows[0]
    require(all(row["constraints"] == shape["constraints"] and row["state_in"] == shape["state_in"]
                and row["state_out"] == shape["state_out"] for row in rows), "journal shape mismatch")
    statement = authorized_statement(plan, next_state(rows[-1]))
    return statement, [[str(v) for v in row["witness"]] for row in rows]


def claim_manifest(plan: Plan, claim: tuple[int, ...], authorized_roots: dict[str, str],
                   train_roots: set[str], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Check claimed payloads against a claim, without granting proof status."""
    outputs = dict(zip((job.name for job in plan.jobs), claimed_outputs(plan, claim), strict=True))
    selected: dict[str, dict[str, Any]] = {}
    for item in items:
        require(set(item) == {"job", "root", "split", "payload"}, "corpus fields")
        name, root = item["job"], item["root"]
        require(name in outputs and name not in selected and hex_digest(root), "job or root")
        require(item["split"] == "TRAIN" and root in train_roots and authorized_roots.get(name) == root,
                "unauthorized TRAIN")
        require(canonical(item["payload"]) == canonical(outputs[name]), "payload mismatch")
        selected[name] = item
    plan_id = digest("plan", asdict(plan))
    state_id = digest("journal-state-v1", claim)
    leaves = [digest("journal-corpus-item-v1", [plan_id, state_id, selected[job.name]])
              for job in plan.jobs if job.name in selected]
    return {"plan": plan_id, "state": state_id,
            "admission_policy": digest("corpus-policy", [authorized_roots, sorted(train_roots)]),
            "count": len(leaves), "root": merkle_root(leaves), "leaves": leaves}


def verify_and_certify(plan: Plan, claim: tuple[int, ...], proof: bytes, executable: Path,
                       executable_sha256: str, authorized_roots: dict[str, str],
                       train_roots: set[str], items: list[dict[str, Any]],
                       timeout: float = 180.0) -> dict[str, Any]:
    """Check actual Nova proof via the operator-pinned native verification program.

    Deployment must pin the runtime and dependencies in an immutable sandbox.
    A mock executable proves transport behavior only. This function does not
    enforce persistent anti-replay, source truth, or completeness of selection.
    """
    manifest = claim_manifest(plan, claim, authorized_roots, train_roots, items)
    statement = authorized_statement(plan, claim)
    require(type(proof) is bytes and 0 < len(proof) <= MAX_PROOF_BYTES, "proof size")
    require(hex_digest(executable_sha256), "executable digest")
    exe = executable.resolve(strict=True)
    require(exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest() == executable_sha256, "verifier pin")
    raw = canonical(statement)
    pin = hashlib.sha256(raw).hexdigest()
    with tempfile.TemporaryDirectory(prefix="zk-dg-journal-") as tmp:
        s, p = Path(tmp) / "statement.json", Path(tmp) / "proof.bin"
        s.write_bytes(raw)
        p.write_bytes(proof)
        result = subprocess.run([str(exe), "verify", str(s), pin, str(p)],
                                capture_output=True, timeout=timeout, check=False)
        require(result.returncode == 0 and re.fullmatch(rb"accepted verify_ms=[0-9]+\s*", result.stdout) is not None,
                "native verifier rejected claim")
    return {"backend": "nova-native-journal-dev-v1", "statement_sha256": pin,
            "proof_sha256": hashlib.sha256(proof).hexdigest(), "verifier_sha256": executable_sha256,
            "manifest": manifest}
