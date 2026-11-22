#!/usr/bin/env python3
"""Independent, restricted reference semantics for finite ground MeTTa queries.

Supports only ground relation facts, match &self with pattern variables,
structural projection, and a small declared Number/String type-error rule.
Every other operation is rejected as outside scope. This module never imports
or calls Hyperon and never reads the upstream author's expected answers.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

Atom = str | tuple[Any, ...]
MAX_TOKENS = 4096
MAX_DEPTH = 24
MAX_RESULTS = 256


class UnsupportedSemantics(ValueError):
    """A source construct is outside the independently specified subset."""


def tokenize(source: str) -> list[str]:
    """Lex symbols, quoted strings and balanced expression delimiters."""
    out: list[str] = []
    i = 0
    while i < len(source):
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if ch in "()[],":
            out.append(ch)
            i += 1
            continue
        if ch == '"':
            start = i
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
            if i > len(source) or source[i - 1] != '"':
                raise UnsupportedSemantics("Unterminated quoted string")
            out.append(source[start:i])
            continue
        start = i
        while i < len(source) and not (
            source[i].isspace() or source[i] in '();[],"'
        ):
            i += 1
        if i == start:
            raise UnsupportedSemantics("Invalid symbol boundary")
        out.append(source[start:i])
        if len(out) > MAX_TOKENS:
            raise UnsupportedSemantics("Token limit exceeded")
    if len(out) > MAX_TOKENS:
        raise UnsupportedSemantics("Token limit exceeded")
    return out


def parse_at(tokens: list[str], offset: int = 0, depth: int = 0) -> tuple[Atom, int]:
    if offset >= len(tokens) or depth > MAX_DEPTH:
        raise UnsupportedSemantics("Truncated or over-deep expression")
    symbol = tokens[offset]
    if symbol == "(":
        members: list[Atom] = []
        offset += 1
        while offset < len(tokens) and tokens[offset] != ")":
            element, offset = parse_at(tokens, offset, depth + 1)
            members.append(element)
        if offset >= len(tokens):
            raise UnsupportedSemantics("Unclosed expression")
        return tuple(members), offset + 1
    if symbol in {")", "[", "]", ","}:
        raise UnsupportedSemantics("Unexpected expression separator")
    if symbol == "!":
        raise UnsupportedSemantics("Top-level evaluation unsupported in facts")
    return symbol, offset + 1


def parse_one(source: str) -> Atom:
    tokens = tokenize(source)
    value, next_pos = parse_at(tokens)
    if next_pos != len(tokens):
        raise UnsupportedSemantics("Trailing tokens")
    return value


def parse_result_line(line: str) -> list[Atom]:
    """Parse native '[a, (b c)]' output without evaluating any expression."""
    tokens = tokenize(line)
    if len(tokens) < 2 or tokens[0] != "[":
        raise UnsupportedSemantics("Expected a native result list")
    output: list[Atom] = []
    pos = 1
    while pos < len(tokens) and tokens[pos] != "]":
        result, pos = parse_at(tokens, pos)
        output.append(result)
        if pos < len(tokens) and tokens[pos] == ",":
            pos += 1
        elif pos < len(tokens) and tokens[pos] != "]":
            raise UnsupportedSemantics("Missing native result separator")
        if len(output) > MAX_RESULTS:
            raise UnsupportedSemantics("Result limit exceeded")
    if pos != len(tokens) - 1 or tokens[pos] != "]":
        raise UnsupportedSemantics("Trailing native output or unclosed result list")
    return output


def is_ground(value: Atom) -> bool:
    if isinstance(value, str):
        return not value.startswith("$")
    return all(is_ground(item) for item in value)


def is_ground_relation(value: Atom) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) > 0
        and isinstance(value[0], str)
        and not value[0].startswith("$")
        and value[0] not in {"=", ":", "!", "match", "pragma!"}
        and is_ground(value)
    )


def match_ground(pattern: Atom, fact: Atom, bindings: dict[str, Atom]) -> bool:
    """Bind query variables against ground facts, including repeated variables."""
    if isinstance(pattern, str) and pattern.startswith("$"):
        if pattern in bindings:
            return bindings[pattern] == fact
        bindings[pattern] = fact
        return True
    if isinstance(pattern, str):
        return pattern == fact
    if not isinstance(fact, tuple) or len(pattern) != len(fact):
        return False
    return all(match_ground(x, y, bindings) for x, y in zip(pattern, fact, strict=True))


def project(template: Atom, bindings: dict[str, Atom]) -> Atom:
    if isinstance(template, str):
        if template.startswith("$"):
            if template not in bindings:
                raise UnsupportedSemantics("Unbound template variable")
            return bindings[template]
        return template
    return tuple(project(x, bindings) for x in template)


def finite_match(facts: list[Atom], query: Atom) -> list[Atom]:
    """Evaluate (match &self pattern projection) over ground, distinct facts."""
    if (
        not isinstance(query, tuple)
        or len(query) != 4
        or query[0] != "match"
        or query[1] != "&self"
    ):
        raise UnsupportedSemantics("Only (match &self pattern projection) is supported")
    if any(not is_ground_relation(fact) for fact in facts):
        raise UnsupportedSemantics("Source facts must be ground relations")
    if len(facts) != len(set(facts)):
        raise UnsupportedSemantics("Repeated source facts require separate semantics")
    answers: list[Atom] = []
    for fact in facts:
        bindings: dict[str, Atom] = {}
        if match_ground(query[2], fact, bindings):
            answers.append(project(query[3], bindings))
    if len(answers) > MAX_RESULTS:
        raise UnsupportedSemantics("Too many query results")
    return answers


def render(value: Atom) -> str:
    if isinstance(value, str):
        return value
    return "(" + " ".join(render(part) for part in value) + ")"


def alpha_signature(value: Atom) -> str:
    """Bounded alpha-renaming for a single returned expression."""
    seen: dict[str, str] = {}

    def norm(v: Atom) -> Atom:
        if isinstance(v, str):
            if v.startswith("$"):
                if v not in seen:
                    seen[v] = f"$alpha{len(seen)}"
                return seen[v]
            return v
        return tuple(norm(x) for x in v)

    return render(norm(value))


def result_signature(results: list[Atom]) -> dict[str, int]:
    """Treat branch order as irrelevant; preserve multiplicities conservatively."""
    counts = Counter(alpha_signature(result) for result in results)
    return dict(sorted(counts.items()))


def compare_native(reference: list[Atom], native_line: str) -> dict[str, Any]:
    observed = parse_result_line(native_line)
    expected_sig = result_signature(reference)
    actual_sig = result_signature(observed)
    return {
        "agreement": expected_sig == actual_sig,
        "reference_multiset": expected_sig,
        "native_multiset": actual_sig,
        "reference_results": len(reference),
        "native_results": len(observed),
        "normalization": "alpha-renamed_branch_unordered_multiplicity_preserved",
    }


def type_of_ground(value: Atom) -> str:
    if not isinstance(value, str):
        return "Expression"
    if value.startswith('"') and value.endswith('"'):
        return "String"
    if value.lstrip("-").isdigit():
        return "Number"
    return "Symbol"


def typed_plus_reference(expr: Atom) -> list[Atom]:
    """One explicit scoped Number + Number rule for type-check-auto probes."""
    if not isinstance(expr, tuple) or len(expr) != 3 or expr[0] != "+":
        raise UnsupportedSemantics("Only binary addition typing is supported")
    for argument_index in (1, 2):
        argument = expr[argument_index]
        kind = type_of_ground(argument)
        if kind != "Number":
            return [("Error", expr, ("BadArgType", str(argument_index), "Number", kind))]
    left, right = expr[1], expr[2]
    if not isinstance(left, str) or not isinstance(right, str):
        raise UnsupportedSemantics("Ground numeric operands required")
    return [str(int(left) + int(right))]
