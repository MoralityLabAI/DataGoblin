"""ZK-DG-02..07 development integration. Explicit backend and trust boundaries."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datagoblin import zk_sigma
from datagoblin.zk_replay import Job, Plan, canonical, clamp, digest, require, step, validate


@dataclass(frozen=True)
class CheckerPolicy:
    backend: str
    relation: str
    key_sha256: str
    executable_sha256: str


def hex_digest(value: str) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def node_context(plan: Plan, job: Job, parents: list[str]) -> str:
    return digest("sigma-graph-node", [digest("plan", asdict(plan)), job.name, parents])


def joined_input(job: Job, outputs: dict[str, tuple[int, ...]]) -> tuple[int, ...]:
    return tuple(clamp(v + sum(outputs[p][i] for p in job.parents)) for i, v in enumerate(job.source))


def prove_graph(plan: Plan) -> list[dict[str, Any]]:
    require(validate(plan) <= 2 and len(plan.jobs) <= 32, "sigma graph limits")
    outputs: dict[str, tuple[int, ...]] = {}
    ids: dict[str, str] = {}
    receipts = []
    for job in plan.jobs:
        x = joined_input(job, outputs)
        y = x
        for _ in range(job.steps):
            y = step(job, x, y)
        context = node_context(plan, job, [ids[p] for p in job.parents])
        receipt = {"job": job.name, "output": y, "proof": zk_sigma.prove(job, x, y, context)}
        outputs[job.name] = y
        ids[job.name] = digest("sigma-receipt", receipt)
        receipts.append(receipt)
    return receipts


def verify_graph(plan: Plan, receipts: list[dict[str, Any]]) -> str:
    """Verify every local proof and ordered join; this is linear, not recursive aggregation."""
    require(validate(plan) <= 2 and len(plan.jobs) <= 32, "sigma graph limits")
    require(len(receipts) == len(plan.jobs), "incomplete graph")
    outputs: dict[str, tuple[int, ...]] = {}
    ids: dict[str, str] = {}
    for job, receipt in zip(plan.jobs, receipts, strict=True):
        require(set(receipt) == {"job", "output", "proof"} and receipt["job"] == job.name, "job mismatch")
        require(type(receipt["output"]) in (list, tuple), "output encoding")
        y = tuple(receipt["output"])
        x = joined_input(job, outputs)
        context = node_context(plan, job, [ids[p] for p in job.parents])
        require(zk_sigma.verify(job, x, y, context, receipt["proof"]), "invalid node proof")
        outputs[job.name] = y
        ids[job.name] = digest("sigma-receipt", receipt)
    return ids[plan.jobs[-1].name]


def lean_statement(theorem: str, environment: str, checker: str, proof_term: str,
                   dependencies: str, allowed_axioms: tuple[str, ...], context: str) -> dict[str, Any]:
    require(all(hex_digest(v) for v in (theorem, environment, checker, proof_term, dependencies, context)),
            "Lean statement digests")
    require(tuple(sorted(set(allowed_axioms))) == allowed_axioms and "sorryAx" not in allowed_axioms,
            "invalid axiom policy")
    return {"relation": "lean-kernel-acceptance-v1", "theorem": theorem, "environment": environment,
            "checker": checker, "proof_term": proof_term, "dependencies": dependencies,
            "allowed_axioms": allowed_axioms, "context": context}


def external_verify(policy: CheckerPolicy, executable: Path, key: bytes, statement: dict[str, Any],
                    proof: bytes, timeout: float = 30.0) -> bool:
    """Operator-pinned verifier protocol. The executable must verify the entire statement."""
    try:
        require(policy.backend in {"zkpi", "lean-zkvm", "nova-compressed", "lucid-supported"}, "backend")
        require(all(hex_digest(v) for v in (policy.key_sha256, policy.executable_sha256)), "pin encoding")
        require(statement.get("relation") == policy.relation, "relation mismatch")
        if policy.backend in {"zkpi", "lean-zkvm"}:
            require(policy.relation == "lean-kernel-acceptance-v1", "Lean relation")
            require(set(statement) == {"relation", "theorem", "environment", "checker", "proof_term",
                                      "dependencies", "allowed_axioms", "context"}, "Lean fields")
            require(tuple(statement["allowed_axioms"]) == tuple(sorted(set(statement["allowed_axioms"])))
                    and "sorryAx" not in statement["allowed_axioms"], "Lean axiom policy")
            require(all(hex_digest(statement[k]) for k in ("theorem", "environment", "checker", "proof_term",
                                                          "dependencies", "context")), "Lean digests")
        require(0 < len(proof) <= 16_000_000 and 0 < len(key) <= 16_000_000, "proof/key size")
        exe = executable.resolve(strict=True)
        require(exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest() == policy.executable_sha256,
                "verifier pin")
        require(hashlib.sha256(key).hexdigest() == policy.key_sha256, "key pin")
        request = {"policy": asdict(policy), "statement": statement,
                   "key": base64.b64encode(key).decode(), "proof": base64.b64encode(proof).decode()}
        raw = canonical(request)
        require(len(raw) <= 48_000_000, "request size")
        # Deployment must use an immutable sandbox and enforce output/resource limits.
        result = subprocess.run([str(exe)], input=raw, capture_output=True, timeout=timeout, check=False)
        require(result.returncode == 0 and len(result.stdout) <= 1024, "checker failure")
        expected = {"accepted": True, "request_sha256": hashlib.sha256(raw).hexdigest()}
        return canonical(json.loads(result.stdout)) == canonical(expected)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return False


@dataclass(frozen=True)
class CircuitRegistration:
    circuit_id: str
    circuit_name: str
    version: str
    proof_system: str
    curve: str
    key_sha256: str
    num_public_inputs: int
    setup_hash: str | None = None


def lucid_registration(reg: CircuitRegistration, key: bytes) -> dict[str, Any]:
    """Build registry metadata only; live API and independent verification are separate."""
    require(reg.proof_system in {"groth16", "plonk", "fflonk", "stark"}, "unsupported Lucid format")
    require(0 < len(key) <= 16_000_000 and hashlib.sha256(key).hexdigest() == reg.key_sha256, "key pin")
    require(type(reg.num_public_inputs) is int and reg.num_public_inputs >= 0, "input count")
    require(reg.proof_system != "groth16" or hex_digest(reg.setup_hash), "setup pin required")
    data = asdict(reg)
    del data["key_sha256"]
    data["verification_key"] = base64.b64encode(key).decode()
    return data


@dataclass(frozen=True)
class Route:
    name: str
    fixed_cost: int
    per_step_cost: int
    proof_class: str


def choose_route(routes: tuple[Route, ...], steps: int, budget: int, required_class: str) -> str | None:
    """A deterministic cost-policy oracle. Availability must be independently measured."""
    require(type(steps) is int and steps > 0 and type(budget) is int and budget >= 0, "invalid budget")
    require(all(type(r.fixed_cost) is int and type(r.per_step_cost) is int and r.fixed_cost >= 0
                and r.per_step_cost >= 0 for r in routes), "invalid costs")
    viable = [(r.fixed_cost + r.per_step_cost * steps, r.name) for r in routes if r.proof_class == required_class
              and r.fixed_cost + r.per_step_cost * steps <= budget]
    return min(viable)[1] if viable else None


def fit_route_cost(observations: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Fit a nonnegative conservative affine upper envelope on train observations."""
    require(len(observations) >= 2 and all(type(x) is int and type(y) is int and x > 0 and y >= 0
                                         for x, y in observations), "invalid cost observations")
    # Integer slope candidates from observed breakpoints plus zero; train-only fit.
    slopes = {0}
    for x, y in observations:
        for u, v in observations:
            if x > u and y >= v:
                slopes.add((y - v + x - u - 1) // (x - u))
    pairs = [(max(0, max(y - b*x for x, y in observations)), b) for b in slopes]
    return min(pairs, key=lambda ab: (sum(ab[0] + ab[1]*x - y for x, y in observations), ab))


def merkle_root(leaves: list[str]) -> str:
    require(bool(leaves) and all(hex_digest(v) for v in leaves), "empty or invalid leaves")
    layer = [digest("corpus-leaf", x) for x in leaves]
    count = len(layer)
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [digest("corpus-branch", layer[i:i+2]) for i in range(0, len(layer), 2)]
    return digest("corpus-root", [count, layer[0]])


def certify_corpus(plan: Plan, receipts: list[dict[str, Any]], authorized_roots: dict[str, str],
                   train_roots: set[str], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Certify only exact graph outputs under an independently authorized TRAIN allowlist."""
    terminal = verify_graph(plan, receipts)
    by_job = {r["job"]: r for r in receipts}
    seen: set[str] = set()
    leaves = []
    for item in items:
        require(set(item) == {"job", "root", "split", "payload"}, "corpus fields")
        job, root = item["job"], item["root"]
        require(job in by_job and job not in seen and hex_digest(root), "job or source root")
        require(item["split"] == "TRAIN" and root in train_roots and authorized_roots.get(job) == root,
                "not authorized TRAIN")
        require(canonical(item["payload"]) == canonical(by_job[job]["output"]), "payload not proved output")
        seen.add(job)
        leaves.append(digest("corpus-item", [item, digest("sigma-receipt", by_job[job])]))
    return {"backend": "sigma-verified-public-corpus-v1", "plan": digest("plan", asdict(plan)),
            "terminal": terminal, "admission_policy": digest("corpus-policy", [authorized_roots, sorted(train_roots)]),
            "count": len(leaves), "root": merkle_root(leaves), "leaves": leaves}
