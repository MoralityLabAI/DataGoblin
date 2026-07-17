from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from datagoblin.domain.recipes import DataRecipe
from datagoblin.serialization import canonical_json


class VerdictDecision(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    AUDIT = "AUDIT"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"


ROLE_CONDITIONS = (
    "direct",
    "synth,direct",
    "synth,cot",
    "textual_scout",
    "textual_alchemist",
    "textual_rat",
)


@dataclass(frozen=True)
class Verdict:
    decision: VerdictDecision
    violations: tuple[str, ...]
    confidence: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "violations": list(self.violations),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ParsedVerdict:
    parsed: bool
    verdict: Verdict | None
    error: str | None


_ROLE_TEXT = {
    "direct": "Judge the recipe directly.",
    "synth,direct": "Internally synthesize the relevant checks, then judge the recipe directly.",
    "synth,cot": "Reason internally through the checks; do not reveal private reasoning.",
    "textual_scout": "Act as SCOUT: emphasize unexplained gaps, competing hypotheses, and falsifiability.",
    "textual_alchemist": "Act as ALCHEMIST: emphasize recipe design, controls, splits, solvers, and verifiers.",
    "textual_rat": "Act as RAT: adversarially inspect leakage, confounds, self-certification, and permissions.",
}


def build_verdict_prompt(recipe: DataRecipe | Mapping[str, Any], *, condition: str = "direct") -> str:
    if condition not in ROLE_CONDITIONS:
        raise ValueError(f"unsupported role condition: {condition}")
    recipe_value = recipe.to_jsonable() if isinstance(recipe, DataRecipe) else dict(recipe)
    return "\n".join(
        (
            "You are evaluating a DataGoblin data recipe under its executable research-integrity constitution.",
            _ROLE_TEXT[condition],
            "Treat architectural interpretations and proposer evidence as hypotheses, not proof.",
            "Return exactly one JSON object with no markdown and exactly these keys:",
            '{"decision":"ACCEPT|REVISE|AUDIT|ABSTAIN|REJECT","violations":["rule_id"],"confidence":0.0}',
            "Confidence must lie in [0,1]. Use only public recipe content.",
            f"RECIPE_JSON={canonical_json(recipe_value)}",
        )
    )


def parse_verdict_output(output: str) -> ParsedVerdict:
    try:
        value = json.loads(output.strip())
    except json.JSONDecodeError as exc:
        return ParsedVerdict(False, None, f"invalid_json:{exc.msg}")
    if not isinstance(value, dict) or set(value) != {"decision", "violations", "confidence"}:
        return ParsedVerdict(False, None, "schema_keys_mismatch")
    if not isinstance(value["violations"], list) or not all(
        isinstance(item, str) for item in value["violations"]
    ):
        return ParsedVerdict(False, None, "violations_must_be_string_array")
    if isinstance(value["confidence"], bool) or not isinstance(value["confidence"], (int, float)):
        return ParsedVerdict(False, None, "confidence_must_be_number")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        return ParsedVerdict(False, None, "confidence_out_of_range")
    try:
        decision = VerdictDecision(value["decision"])
    except ValueError:
        return ParsedVerdict(False, None, "unknown_decision")
    return ParsedVerdict(
        True,
        Verdict(decision, tuple(dict.fromkeys(str(item) for item in value["violations"])), confidence),
        None,
    )


def schema_complexity(recipe: DataRecipe | Mapping[str, Any]) -> float:
    value = recipe.to_jsonable() if isinstance(recipe, DataRecipe) else recipe

    def count(item: Any, depth: int) -> float:
        if isinstance(item, Mapping):
            return 1.0 + 0.1 * depth + sum(count(child, depth + 1) for child in item.values())
        if isinstance(item, list):
            return 0.5 + sum(count(child, depth + 1) for child in item)
        return 0.25

    return count(value, 0)

