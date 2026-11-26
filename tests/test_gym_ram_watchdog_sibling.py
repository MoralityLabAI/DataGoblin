import importlib.util
import json
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "gym_ram_watchdog.py"


class SiblingTimerTests(unittest.TestCase):
    def test_one_action_per_pass_preserves_sibling_idle_samples(self):
        spec = importlib.util.spec_from_file_location("watchdog_under_test", SCRIPT)
        watchdog = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(watchdog)

        elapsed = 0

        def wait(timeout=None):
            nonlocal elapsed
            if timeout is not None:
                elapsed += 1
            return 0

        workers = [
            SimpleNamespace(pid=pid, create_time=lambda: 1, children=lambda: [],
                            status=lambda: "sleeping", is_running=lambda: True,
                            cpu_times=lambda: SimpleNamespace(user=0, system=0),
                            memory_info=lambda: SimpleNamespace(rss=64 * 1024**2))
            for pid in (11, 12)
        ]
        child = SimpleNamespace(pid=10, wait=wait, poll=lambda: 0 if elapsed >= 8 else None)
        root = SimpleNamespace(children=lambda recursive: workers, is_running=lambda: True)
        psutil = SimpleNamespace(
            Process=lambda pid: root, Error=OSError, NoSuchProcess=ProcessLookupError,
            STATUS_STOPPED="stopped", STATUS_ZOMBIE="zombie",
            virtual_memory=lambda: SimpleNamespace(available=1),
        )
        argv = [
            str(SCRIPT), "--cpu-only", "--worker-mib", "48", "--idle-seconds", "3",
            "--sample-seconds", "1", "--report-only", "--", "fixture",
        ]
        output = StringIO()
        with (patch.object(watchdog.subprocess, "Popen", return_value=child),
              patch.object(watchdog.time, "monotonic", side_effect=lambda: elapsed),
              patch.dict(sys.modules, {"psutil": psutil}), patch.object(sys, "argv", argv),
              redirect_stderr(output)):
            self.assertEqual(watchdog.main(), 0)

        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([(e["pid"], e["idle_seconds"]) for e in events],
                         [(11, 3), (12, 4)])
