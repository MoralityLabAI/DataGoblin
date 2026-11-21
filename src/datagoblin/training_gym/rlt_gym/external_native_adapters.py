"""Live-callback Lean kernel and MCP state-effect adapters for external exit replay.

The caller owns pinned runtime and private oracle. These adapters reject
unauthorized candidate IDs and never trust a candidate's claimed success.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .external_checked_exit import digest

_ALLOWED_AXIOMS = frozenset({"propext", "Quot.sound", "Classical.choice"})


def _receipt(
    task: dict[str, Any], candidate: Any, verifier_sha: str, *,
    passed: bool, code: str, evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "passed": passed, "code": code, "task_digest": digest(task),
        "candidate_digest": digest(candidate), "verifier_sha256": verifier_sha,
        "evidence": evidence,
    }


class LeanKernelChecker:
    """Run menu-ID tactics from a pinned state, then demand a Lean kernel proof."""

    def __init__(self, *, lean: Any, states: Mapping[str, dict[str, Any]], verifier_sha: str):
        if not states or len(verifier_sha) != 64:
            raise ValueError("pinned Lean source states and verifier SHA required")
        self.lean, self.states, self.verifier_sha = lean, states, verifier_sha

    def __call__(self, task: dict[str, Any], candidate: Any) -> dict[str, Any]:
        token = digest(task)
        if token not in self.states:
            raise ValueError("unknown Lean source goal")
        entry = self.states[token]
        menu = entry.get("menu")
        if not isinstance(menu, dict) or not isinstance(entry.get("root_sid"), int):
            raise ValueError("Lean source state/menu missing")
        identifier = candidate.get("id") if isinstance(candidate, dict) else None
        if type(identifier) is not int or identifier not in menu:
            return _receipt(
                task, candidate, self.verifier_sha, passed=False,
                code="candidate_id_rejected", evidence={"kernel_invoked": False},
            )
        tactic = menu[identifier]
        if not isinstance(tactic, str) or not tactic or any(
            x in tactic.lower() for x in ("sorry", "admit", "run_tac", "run_meta", "#eval", "unsafe", "extern")
        ):
            raise ValueError("untrusted tactic in frozen whitelist")
        result = self.lean.run(entry["root_sid"], tactic)
        if not isinstance(result, dict):
            raise ValueError("invalid Lean response")
        state = result.get("state")
        closed = bool(
            result.get("ok") is True and isinstance(state, dict)
            and state.get("closed") is True and state.get("goals") == []
        )
        evidence = {"kernel_invoked": False}
        if not closed:
            return _receipt(
                task, candidate, self.verifier_sha, passed=False, code="open_goals", evidence=evidence,
            )
        sid = state.get("sid")
        if type(sid) is not int:
            raise ValueError("closed Lean state lacks kernel certificate SID")
        cert = self.lean.kernel(sid)
        if not isinstance(cert, dict) or type(cert.get("ok")) is not bool:
            raise ValueError("invalid Lean kernel certificate")
        axioms = cert.get("axioms")
        if not isinstance(axioms, list) or any(type(a) is not str for a in axioms):
            raise ValueError("Lean kernel missing explicit axiom list")
        accepted = cert["ok"] and set(axioms) <= _ALLOWED_AXIOMS
        evidence = {
            "kernel_invoked": True, "kernel_ok": cert["ok"],
            "axioms": sorted(axioms), "kernel_session_sid": sid,
        }
        return _receipt(
            task, candidate, self.verifier_sha, passed=bool(accepted),
            code="kernel_certified" if accepted else "kernel_rejected",
            evidence=evidence,
        )


class MCPStateChecker:
    """Reset and reexecute frozen menu actions, then judge actual state effects."""

    def __init__(
        self, *, cases: Mapping[str, dict[str, Any]], verifier_sha: str,
        execute_fresh: Callable[[str, int], dict[str, Any]],
        judge: Callable[[str, dict[str, Any]], tuple[bool, str]],
    ):
        if not cases or len(verifier_sha) != 64 or not callable(execute_fresh) or not callable(judge):
            raise ValueError("pinned cases, executor and independent judge required")
        self.cases, self.verifier_sha = cases, verifier_sha
        self.execute_fresh, self.judge = execute_fresh, judge

    def __call__(self, task: dict[str, Any], candidate: Any) -> dict[str, Any]:
        token = digest(task)
        if token not in self.cases:
            raise ValueError("unknown MCP source case")
        spec = self.cases[token]
        menu = spec.get("menu")
        if not isinstance(menu, dict) or not isinstance(spec.get("snapshot_sha"), str):
            raise ValueError("MCP case needs frozen menu and environment snapshot")
        choice = candidate.get("id") if isinstance(candidate, dict) else None
        if type(choice) is not int or choice not in menu:
            return _receipt(
                task, candidate, self.verifier_sha, passed=False,
                code="candidate_id_rejected", evidence={"tools_executed": False},
            )
        result = self.execute_fresh(token, choice)
        if not isinstance(result, dict) or result.get("snapshot_sha") != spec["snapshot_sha"]:
            raise ValueError("MCP runner did not reset to registered state")
        if result.get("action_id") != choice:
            raise ValueError("MCP runner executed another action")
        ok, reason = self.judge(token, result)
        if type(ok) is not bool or not isinstance(reason, str):
            raise ValueError("invalid independent MCP checker result")
        return _receipt(
            task, candidate, self.verifier_sha, passed=ok, code=reason,
            evidence={
                "tools_executed": True, "snapshot_sha": spec["snapshot_sha"],
                "effect_sha256": digest(result.get("effect")),
            },
        )
