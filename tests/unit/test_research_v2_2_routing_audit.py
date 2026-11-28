"""Exhaustive routing support and native CPU experiment."""

import json
import os
import subprocess
import sys


def test_routing_support_audit(capsys):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    cmd = [sys.executable, "scripts/research_v2_2/routing_support_audit.py"]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=True)
    report = json.loads(result.stdout)
    assert report["support_size"] == 896
    assert len(report["runs"]) == 5
    assert abs(report["expected_seen_test_probability"] - .839372755691776) < 1e-10
    assert all(r["rule"]["unseen_accuracy"] == 1 for r in report["runs"])
    with capsys.disabled():
        print("RV22_H_RECEIPT " + json.dumps(report, sort_keys=True, separators=(",", ":")))
