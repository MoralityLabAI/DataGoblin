"""GPU and RAM for hill-climb recipes, through the vramorama Python client.

A recipe may declare `resources` in `configs/hillclimb/lanes.json`:

- `{"ram": "3G"}`: a CPU gym. It waits for room in the shared RAM reservations, then runs under a
  hard RAM cap (a Windows Job Object memory limit over its whole process tree) at below-normal
  priority. The card is not involved, so a power-capped GPU never holds it back.
- `{"vram": "6G", "ram": "8G", ...}`: a job that needs the card. It queues as a `vramorama run` lease
  whose commit gate leaves room for its RAM, and runs under the same cap inside the lease.

The client is the `vramorama` package (vramorama's `python/` folder): installed, named by
DATAGOBLIN_VRAMORAMA_PY, or found in a sibling `vramorama` checkout. Without it, CPU recipes run
without a cap and the run record says so; recipes that need the card are refused.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from .lanes import HillclimbError

REPORT = "resources.json"
EXIT_NO_ROOM = 75


def find_client(root: Path) -> tuple[ModuleType | None, str | None]:
    """The vramorama module and the folder to put on a child's PYTHONPATH (None when installed)."""
    try:
        return importlib.import_module("vramorama"), None
    except ImportError:
        pass
    candidates = [os.environ.get("DATAGOBLIN_VRAMORAMA_PY"), str(root.parent / "vramorama" / "python")]
    for candidate in candidates:
        if candidate and (Path(candidate) / "vramorama" / "__init__.py").is_file():
            sys.path.insert(0, candidate)
            try:
                return importlib.import_module("vramorama"), candidate
            except ImportError:
                sys.path.remove(candidate)
    return None, None


class ResourceManager:
    def __init__(self, root: Path) -> None:
        self.module, self.path = find_client(root)
        self.client = self.module.Client() if self.module is not None else None

    @property
    def available(self) -> bool:
        return self.module is not None

    def status(self) -> dict[str, Any]:
        """The card, the lease queue, RAM and RAM reservations, or why they cannot be read."""
        if self.module is None or self.client is None:
            return {"client": None, "note": "vramorama Python client not found: CPU recipes run without RAM caps, "
                                            "GPU recipes are refused"}
        summary = self.client.status()
        summary["client"] = getattr(self.module, "__version__", "unknown")
        summary["rule"] = ("GPU work queues as vramorama leases (one queued job per session); CPU gyms wait for RAM "
                           "room and run under a hard cap")
        return summary

    def wrap(self, recipe: str, resources: Mapping[str, Any], argv: Sequence[str], *, run_dir: Path,
             owner: str, timeout: int) -> tuple[list[str], dict[str, Any]]:
        """The argv that runs this recipe with its declared resources, and what the run record should say."""
        if not resources:
            return list(argv), {"managed": False}
        if self.module is None:
            if resources.get("vram"):
                raise HillclimbError("resources_unavailable",
                                     f"{recipe} needs the GPU queue: install vramorama's Python client and binary")
            return list(argv), {"managed": False, "reason": "vramorama client not found; ran without a RAM cap"}
        default_wait = min(600, max(30, timeout // 2))
        spec = self.module.Resources(
            vram=resources.get("vram"), ram=resources.get("ram"), owner=owner,
            priority=resources.get("priority", "below_normal"), wait_secs=resources.get("wait_secs", default_wait),
            guard_power=resources.get("guard_power"), guard_idle=resources.get("guard_idle"),
            restart=resources.get("restart"),
        )
        try:
            plan = self.module.plan(list(argv), spec, client=self.client, report=run_dir / REPORT)
        except self.module.ResourceError as exc:
            raise HillclimbError("resources_unavailable", f"{recipe}: {exc}") from exc
        return plan.argv, {"managed": True, "mode": plan.mode, "ram_mib": plan.ram_mib, "vram_mib": plan.vram_mib,
                           "commit_gate_pct": plan.commit_gate_pct, "owner": owner}

    @staticmethod
    def report(run_dir: Path) -> dict[str, Any] | None:
        path = run_dir / REPORT
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            return None
        keys = ("admitted", "waited_seconds", "run_seconds", "limit_mib", "peak_mib", "limit_hit", "priority")
        return {key: data.get(key) for key in keys if key in data}
