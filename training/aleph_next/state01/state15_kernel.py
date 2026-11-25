"""Single-process Lean CLI accounting for STATE-15 and external proof-query experiments."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

PIN = "4.19.0"
COMMIT = "6caaee842e94"
AXIOM_LINE = re.compile(r"^'([^']+)' (?:does not depend on any axioms|depends on axioms: \[([^\]]*)\])$", re.M)
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def toolchain(lean=None):
    executable = lean or shutil.which("lean")
    if not executable:
        raise RuntimeError("Lean executable unavailable")
    path = Path(executable).resolve()
    version = subprocess.check_output([path, "--version"], encoding="utf-8").strip()
    if f"version {PIN}," not in version or COMMIT not in version:
        raise ValueError(f"expected Lean {PIN} at {COMMIT}, observed {version}")
    prefix = subprocess.check_output([path, "--print-prefix"], encoding="utf-8").strip()
    return {
        "executable": str(path),
        "version": version,
        "prefix": prefix,
        "executable_sha256": digest(path.read_bytes()),
    }


def _child_cpu(process, before):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        times = [wintypes.FILETIME() for _ in range(4)]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GetProcessTimes(wintypes.HANDLE(int(process._handle)), *(ctypes.byref(t) for t in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        return sum((t.dwHighDateTime << 32) + t.dwLowDateTime for t in times[2:]) / 10_000_000
    import resource

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime - before


def compile_query(lean, path, timeout=180):
    """Charge even a timeout or rejected elaboration. Call serially within a worker."""
    path = Path(path).resolve()
    source = path.read_bytes()
    before = 0.0
    if os.name != "nt":
        import resource

        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        before = usage.ru_utime + usage.ru_stime
    start = time.perf_counter()
    with subprocess.Popen([str(lean), "-j", "1", str(path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as child:
        try:
            data, _ = child.communicate(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            child.kill()
            data, _ = child.communicate()
            timed_out = True
        elapsed = time.perf_counter() - start
        cpu_seconds = _child_cpu(child, before)
        code = child.returncode
    path.with_suffix(".log").write_bytes(data)
    log = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    return {
        "exit_code": code,
        "timed_out": timed_out,
        "wall_seconds": elapsed,
        "cpu_seconds": cpu_seconds,
        "source_bytes": len(source),
        "source_sha256": digest(source),
        "log_sha256": digest(data),
    }, log


def proof_axioms(log, names):
    rows = AXIOM_LINE.findall(log)
    axioms = {name: sorted({item.strip() for item in raw.split(",") if item.strip()}) for name, raw in rows}
    valid = (
        len(rows) == len(names) and set(axioms) == set(names) and all(set(a) <= ALLOWED_AXIOMS for a in axioms.values())
    )
    return valid, axioms
