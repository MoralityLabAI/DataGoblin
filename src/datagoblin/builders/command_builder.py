from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from datagoblin.domain.builders import BuilderTicket
from datagoblin.hashing import sha256_file, sha256_json

from .protocol import BuildResult


def _resolved_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


class CommandBuilderBackend:
    """Explicit opt-in process boundary; not a security sandbox."""

    def __init__(self, argv: Sequence[str], environment_allowlist: Sequence[str] = ()) -> None:
        if not argv:
            raise ValueError("command builder requires argv")
        self.argv = tuple(argv)
        self.environment_allowlist = tuple(environment_allowlist)

    def build(self, ticket: BuilderTicket, build_root: Path) -> BuildResult:
        if ticket.network_permission:
            raise ValueError("command builder network is denied in v0")
        root = build_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        env = {name: os.environ[name] for name in self.environment_allowlist if name in os.environ}
        completed = subprocess.run(
            self.argv,
            input=json.dumps(ticket.to_jsonable()),
            text=True,
            capture_output=True,
            timeout=ticket.resource_budget.wall_clock_seconds,
            cwd=root,
            env=env,
            shell=False,
            check=False,
        )
        cap = min(ticket.resource_budget.storage_mb * 1024 * 1024, 4 * 1024 * 1024)
        if len(completed.stdout.encode()) + len(completed.stderr.encode()) > cap:
            raise ValueError("builder output exceeded receipt cap")
        if completed.returncode != 0:
            raise RuntimeError(f"builder command failed: {completed.returncode}")
        outputs = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"builder output contains symlink: {path}")
            if path.is_file():
                if not _resolved_within(path, root):
                    raise ValueError(f"builder output escaped build root: {path}")
                outputs.append(path)
        receipt = {
            "ticket_id": ticket.ticket_id,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "outputs": {path.relative_to(root).as_posix(): sha256_file(path) for path in outputs},
            "cleanup": "working directory retained for audit",
        }
        receipt["receipt_hash"] = sha256_json(receipt)
        return BuildResult(True, root, tuple(outputs), receipt)
