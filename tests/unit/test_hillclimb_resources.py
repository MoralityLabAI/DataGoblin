from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from datagoblin.hillclimb import bench
from datagoblin.hillclimb.lanes import HillclimbError, Registry
from datagoblin.hillclimb.resources import ResourceManager
from datagoblin.hillclimb.service import Service

ROOT = Path(__file__).resolve().parents[2]
HAS_CLIENT = importlib.util.find_spec("vramorama") is not None


def _without_client() -> ResourceManager:
    manager = ResourceManager.__new__(ResourceManager)
    manager.module, manager.path, manager.client = None, None, None
    return manager


def test_every_recipe_declares_ram_except_the_detaching_launcher() -> None:
    registry = Registry.load()
    uncapped = [name for name, recipe in registry.recipes.items() if not recipe.resources.get("ram")]
    # adaptergym_launch starts detached GPU jobs through vramorama; a kill-on-close cap would end them.
    assert uncapped == ["adaptergym_launch"]


def test_unknown_resource_keys_are_refused(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs" / "hillclimb" / "lanes.json").read_text(encoding="utf-8"))
    raw["recipes"]["curate"]["resources"]["gpus"] = 2
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(HillclimbError, match="unknown resources"):
        Registry.load(path)


def test_resource_declarations_leave_the_bench_unchanged(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "configs" / "hillclimb" / "lanes.json").read_text(encoding="utf-8"))
    for recipe in raw["recipes"].values():
        recipe.pop("resources", None)
    path = tmp_path / "lanes.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    stripped, _ = bench.make_tasks("dev", 0, 120, Registry.load(path))
    current, _ = bench.make_tasks("dev", 0, 120, Registry.load())
    assert [t["view_sha256"] for t in stripped] == [t["view_sha256"] for t in current]


def test_without_the_client_cpu_recipes_run_uncapped_and_gpu_recipes_are_refused(tmp_path: Path) -> None:
    manager = _without_client()
    argv, info = manager.wrap("rlt_run", {"ram": "3G"}, ["python", "x"], run_dir=tmp_path, owner="o", timeout=600)
    assert argv == ["python", "x"] and info == {"managed": False,
                                                 "reason": "vramorama client not found; ran without a RAM cap"}
    with pytest.raises(HillclimbError, match="resources_unavailable"):
        manager.wrap("serve", {"vram": "4G"}, ["x"], run_dir=tmp_path, owner="o", timeout=600)
    assert manager.status()["client"] is None


@pytest.mark.skipif(not HAS_CLIENT, reason="vramorama Python client not installed")
def test_a_recipe_runs_under_its_ram_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRAMORAMA_HOME", str(tmp_path / "vramorama-home"))
    service = Service(root=ROOT, state_dir=tmp_path / "state", max_tier="offline")
    status = service.call("suite_status", {})["resources"]
    assert status["client"] and "memory" in status and "ram_reservations" in status
    move = service.call("propose_move", {"lane": "research_loop", "plan": ["dg_validate"],
                                         "hypothesis": "The validator runs inside its RAM cap."})["move_id"]
    run = service.call("dg_validate", {"move_id": move})
    resources = run["resources"]
    assert run["status"] == "ok", run["stderr_tail"]
    assert resources["managed"] and resources["mode"] == "ramcap" and resources["limit_mib"] == 1024
    assert resources["admitted"] is True and resources["limit_hit"] in (False, None)
    assert "vramorama.ramcap" in service.ledger.read()[-1]["body"]["argv"]


def test_loopback_recipes_never_ask_for_gpu_memory() -> None:
    # They only send HTTP to a server something else already holds, often from inside that server's
    # vramorama lease (an adaptergym eval step); a vram request there would queue a nested lease.
    registry = Registry.load()
    loopback = {name: recipe.resources for name, recipe in registry.recipes.items() if recipe.tier == "loopback"}
    assert loopback and not any(resources.get("vram") for resources in loopback.values()), loopback
