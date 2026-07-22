from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from datagoblin.hashing import sha256_json

_WORD = re.compile(r"[a-z0-9_:+.-]+")


def _flatten_tokens(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in sorted(value.items(), key=lambda pair: str(pair[0])):
            yield f"key:{str(key).lower()}"
            yield from _flatten_tokens(nested)
    elif isinstance(value, list):
        yield "type:list"
        for nested in value:
            yield from _flatten_tokens(nested)
    elif isinstance(value, str):
        yield from _WORD.findall(value.lower())
    elif value is None:
        yield "value:null"
    elif isinstance(value, bool):
        yield f"value:{str(value).lower()}"
    else:
        yield f"value:{value}"


def semantic_fingerprint(
    game: str,
    values: Sequence[Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    """Return value-hiding exact and word-shingle hashes for allowlisted safe fields."""
    content = [
        {"field": item["field"], "path_sha256": item["path_sha256"], "value": item["value"]}
        for item in values
    ]
    exact = sha256_json({"game": game, "content": content})
    tokens = [game]
    for item in content:
        tokens.append(f"field:{item['field']}")
        tokens.extend(_flatten_tokens(item["value"]))
    if len(tokens) == 1:
        return exact, (exact,)
    width = min(3, len(tokens))
    shingles = {
        sha256_json(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    }
    return exact, tuple(sorted(shingles or {exact}))


def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0
