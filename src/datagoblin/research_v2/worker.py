"""Execute allowlisted NumPy operations inside a short-lived Linux process."""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    # Apply limits before importing NumPy. No generated code or shell enters this worker.
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    from .learners import evaluate, fit

    raw = sys.stdin.buffer.read(200001)
    if len(raw) > 200000:
        raise ValueError("worker request exceeds bound")
    request = json.loads(raw)
    if request.get("op") == "fit" and set(request) == {"op", "kind", "seed", "config", "calibration_seed"}:
        artifact = fit(request["kind"], request["seed"], request["config"])
        result = {"artifact": artifact, "calibration": evaluate(artifact, request["calibration_seed"], "calibration")}
    elif request.get("op") == "evaluate" and set(request) == {"op", "artifact", "seed", "split"}:
        result = evaluate(request["artifact"], request["seed"], request["split"])
    else:
        raise ValueError("unsupported worker request")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
