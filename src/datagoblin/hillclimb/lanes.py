"""Lane registry: hill-climb lanes, their frozen gates and allowlisted recipes.

`configs/hillclimb/lanes.json` is the single source. A recipe is a fixed argv template with typed
parameters; the model chooses parameter values, never the command. `call_reason` is the exact
legality check shared by the MCP service and the decision bench.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

SCHEMA = "datagoblin.hillclimb.lanes.v1"
TIERS = ("offline", "loopback", "gpu")
PARAM_TYPES = frozenset(
    {"int", "float", "enum", "str", "loopback_url", "path", "path_list", "int_list", "enum_list"}
)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
CALL_REASONS = (
    "lane_not_open",
    "recipe_not_in_lane",
    "tier_not_allowed",
    "unknown_param",
    "missing_param",
    "non_loopback_url",
    "bad_param",
    "ok",
)
_TOKEN = re.compile(r"^\{([a-z_][a-z0-9_]*)(\*|\?--[a-z0-9-]+)?\}$")
_INLINE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
DEFAULT_LANES = Path(__file__).resolve().parents[3] / "configs" / "hillclimb" / "lanes.json"


class HillclimbError(Exception):
    """A classified failure at the hill-climb boundary; `code` is stable for callers."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def tier_rank(tier: str) -> int:
    if tier not in TIERS:
        raise HillclimbError("bad_tier", tier)
    return TIERS.index(tier)


def is_loopback_url(value: str) -> bool:
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname in LOOPBACK_HOSTS
        and port is not None
        and not parts.username
        and not parts.password
        and not parts.query
        and not parts.fragment
        and parts.path.rstrip("/") == "/v1"
    )


@dataclass(frozen=True)
class Param:
    name: str
    type: str
    default: Any = None
    optional: bool = False
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] | None = None
    pattern: str | None = None
    max_items: int | None = None
    kind: str | None = None

    @property
    def required(self) -> bool:
        return self.default is None and not self.optional

    def _scalar_ok(self, value: Any, kind: str) -> bool:
        if kind == "int":
            if type(value) is not int:
                return False
        elif kind == "float":
            if type(value) not in (int, float) or not math.isfinite(value):
                return False
        elif not isinstance(value, str) or not value or len(value) > 500 or "\x00" in value:
            return False
        if self.choices is not None and value not in self.choices:
            return False
        if kind in ("int", "float") and (
            (self.min is not None and value < self.min) or (self.max is not None and value > self.max)
        ):
            return False
        return not (kind == "str" and self.pattern is not None and re.fullmatch(self.pattern, value) is None)

    def check(self, value: Any) -> str:
        """Return ok, non_loopback_url or bad_param for one supplied value."""
        if self.type == "loopback_url":
            if not isinstance(value, str) or not value:
                return "bad_param"
            return "ok" if is_loopback_url(value) else "non_loopback_url"
        if self.type in ("int_list", "enum_list", "path_list"):
            item_kind = {"int_list": "int", "enum_list": "enum", "path_list": "path"}[self.type]
            if not isinstance(value, list) or not value:
                return "bad_param"
            if self.max_items is not None and len(value) > self.max_items:
                return "bad_param"
            if len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
                return "bad_param"
            return "ok" if all(self._scalar_ok(item, item_kind) for item in value) else "bad_param"
        return "ok" if self._scalar_ok(value, self.type) else "bad_param"

    def public(self) -> dict[str, Any]:
        view: dict[str, Any] = {"type": self.type}
        for key in ("min", "max", "pattern", "max_items", "kind"):
            if getattr(self, key) is not None:
                view[key] = getattr(self, key)
        if self.choices is not None:
            view["choices"] = list(self.choices)
        if self.default is not None:
            view["default"] = self.default
        if self.optional:
            view["optional"] = True
        return view


@dataclass(frozen=True)
class Recipe:
    name: str
    tier: str
    description: str
    argv: tuple[str, ...]
    params: Mapping[str, Param]
    timeout_seconds: int
    produces: Mapping[str, Mapping[str, Any]]
    env: Mapping[str, str]
    cwd: str | None = None
    requires: tuple[str, ...] = ()
    mkdir_out: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "description": self.description,
            "params": {name: param.public() for name, param in self.params.items()},
            "produces": sorted(self.produces),
            "requires": list(self.requires),
        }


@dataclass(frozen=True)
class LaneMetric:
    name: str
    direction: str
    min_delta: float
    split: str

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "direction": self.direction, "min_delta": self.min_delta, "split": self.split}


@dataclass(frozen=True)
class Lane:
    id: str
    title: str
    climbs: str
    metric: LaneMetric | None
    patience: int
    max_moves: int
    recipes: tuple[str, ...]


def _param(name: str, raw: Mapping[str, Any]) -> Param:
    kind = raw.get("type")
    if kind not in PARAM_TYPES:
        raise HillclimbError("bad_registry", f"param {name}: unknown type {kind!r}")
    choices = raw.get("choices")
    return Param(
        name=name,
        type=kind,
        default=raw.get("default"),
        optional=bool(raw.get("optional", False)),
        min=raw.get("min"),
        max=raw.get("max"),
        choices=tuple(choices) if choices is not None else None,
        pattern=raw.get("pattern"),
        max_items=raw.get("max_items"),
        kind=raw.get("kind"),
    )


class Registry:
    """The parsed lanes.json, with its byte digest for ledger records."""

    def __init__(self, raw: Mapping[str, Any], sha256: str, path: Path | None = None) -> None:
        if raw.get("schema") != SCHEMA:
            raise HillclimbError("bad_registry", f"schema must be {SCHEMA}")
        self.sha256 = sha256
        self.path = path
        self.roots: dict[str, Mapping[str, Any]] = dict(raw.get("roots", {}))
        self.recipes: dict[str, Recipe] = {}
        for name, spec in raw["recipes"].items():
            tier_rank(spec["tier"])
            params = {pname: _param(pname, praw) for pname, praw in spec.get("params", {}).items()}
            for default_name, param in params.items():
                if param.default is not None and param.check(param.default) != "ok":
                    raise HillclimbError("bad_registry", f"{name}.{default_name}: default fails its own check")
            self.recipes[name] = Recipe(
                name=name,
                tier=spec["tier"],
                description=spec["description"],
                argv=tuple(spec["argv"]),
                params=params,
                timeout_seconds=int(spec.get("timeout_seconds", 600)),
                produces=dict(spec.get("produces", {})),
                env=dict(spec.get("env", {})),
                cwd=spec.get("cwd"),
                requires=tuple(spec.get("requires", ())),
                mkdir_out=bool(spec.get("mkdir_out", False)),
            )
            for root in self.recipes[name].requires:
                if root not in self.roots:
                    raise HillclimbError("bad_registry", f"{name} requires undeclared root {root}")
        self.lanes: dict[str, Lane] = {}
        for lane_id, spec in raw["lanes"].items():
            metric = spec.get("metric")
            lane_metric = None
            if metric is not None:
                if metric["direction"] not in ("up", "down"):
                    raise HillclimbError("bad_registry", f"{lane_id}: direction must be up or down")
                lane_metric = LaneMetric(
                    metric["name"], metric["direction"], float(metric["min_delta"]), metric["split"]
                )
            recipes = tuple(spec["recipes"])
            unknown = [name for name in recipes if name not in self.recipes]
            if unknown:
                raise HillclimbError("bad_registry", f"{lane_id}: unknown recipes {unknown}")
            if lane_metric is not None and not any(lane_metric.name in self.recipes[r].produces for r in recipes):
                raise HillclimbError("bad_registry", f"{lane_id}: no recipe produces {lane_metric.name}")
            self.lanes[lane_id] = Lane(
                id=lane_id,
                title=spec["title"],
                climbs=spec["climbs"],
                metric=lane_metric,
                patience=int(spec.get("patience", 0)),
                max_moves=int(spec["max_moves"]),
                recipes=recipes,
            )

    @classmethod
    def load(cls, path: Path | None = None) -> Registry:
        source = Path(path) if path is not None else DEFAULT_LANES
        data = source.read_bytes()
        return cls(json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest(), source)

    def lanes_using(self, recipe: str) -> list[str]:
        return [lane.id for lane in self.lanes.values() if recipe in lane.recipes]

    def resolve_roots(self, root: Path) -> dict[str, Path | None]:
        """External checkouts by environment variable, then sibling directory name."""
        found: dict[str, Path | None] = {}
        for name, spec in self.roots.items():
            marker = spec.get("marker", "")
            candidates: list[Path] = []
            env = spec.get("env")
            if env and os.environ.get(env):
                candidates.append(Path(os.environ[env]))
            candidates.extend(root.parent / sibling for sibling in spec.get("siblings", ()))
            found[name] = next((c.resolve() for c in candidates if (c / marker).is_file()), None)
        return found


def param_reason(recipe: Recipe, params: Mapping[str, Any]) -> str:
    if any(name not in recipe.params for name in params):
        return "unknown_param"
    if any(param.required and name not in params for name, param in recipe.params.items()):
        return "missing_param"
    for name, param in recipe.params.items():
        if name in params:
            reason = param.check(params[name])
            if reason != "ok":
                return reason
    return "ok"


def call_reason(
    *, lane_status: str, lane_recipes: Sequence[str], recipe: Recipe, max_tier: str, params: Mapping[str, Any]
) -> str:
    """The first failing legality check for one recipe call, or ok (order is part of the contract)."""
    if lane_status != "open":
        return "lane_not_open"
    if recipe.name not in lane_recipes:
        return "recipe_not_in_lane"
    if tier_rank(recipe.tier) > tier_rank(max_tier):
        return "tier_not_allowed"
    return param_reason(recipe, params)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        raise HillclimbError("bad_param", "booleans have no argv form")
    if isinstance(value, float):
        return repr(value)
    return str(value)


def render_text(template: str, values: Mapping[str, Any], context: Mapping[str, str]) -> str:
    """Substitute every {name} in a string from the parameter values, then the context."""

    def inline(found: re.Match[str]) -> str:
        name = found.group(1)
        value = values[name] if name in values else context.get(name)
        if value is None or isinstance(value, list):
            raise HillclimbError("missing_param", name)
        return _format(value)

    return _INLINE.sub(inline, template)


def render_argv(recipe: Recipe, values: Mapping[str, Any], context: Mapping[str, str]) -> list[str]:
    """Expand the recipe's argv template: {name}, {name*} (list) and {name?--flag} (optional)."""
    argv: list[str] = []
    for token in recipe.argv:
        match = _TOKEN.match(token)
        if not match:
            argv.append(render_text(token, values, context))
            continue
        name, modifier = match.group(1), match.group(2)
        value = values[name] if name in values else context.get(name)
        if modifier == "*":
            if not isinstance(value, list):
                raise HillclimbError("bad_param", f"{name} must be a list")
            argv.extend(_format(item) for item in value)
        elif modifier:
            if value is not None:
                argv.extend([modifier[1:], _format(value)])
        else:
            if value is None:
                raise HillclimbError("missing_param", name)
            argv.append(_format(value))
    return argv
