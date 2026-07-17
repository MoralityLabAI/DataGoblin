from __future__ import annotations

from pathlib import Path

import pytest

from datagoblin.adapters.hybrid_trm_ldt import HybridTRMLDTAdapter
from datagoblin.adapters.upstream_registry import UpstreamEntry, UpstreamRegistry
from datagoblin.builders.command_builder import _resolved_within
from datagoblin.ledger.artifact_store import ArtifactStore
from datagoblin.ledger.database import ExperimentLedger


def test_artifact_hash_matches_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    reference = store.put_bytes(b"goblin", media_type="text/plain")
    assert store.read(reference) == b"goblin"
    assert reference.sha256 == "f59ddf918f384a1b7e1d1011c49c3f3fd38421fc3ed3d90dfaa9bb1633325478"


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    assert _resolved_within(root / "ok.json", root)
    assert not _resolved_within(root / ".." / "escape.json", root)


def test_symlink_escape_is_detectable(tmp_path: Path) -> None:
    root = tmp_path / "build"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")
    assert not _resolved_within(link / "file", root)


def test_evaluated_record_is_immutable_in_ledger(tmp_path: Path) -> None:
    with ExperimentLedger(tmp_path / "ledger.sqlite") as ledger:
        ledger.insert("experiments", "exp", {"status": "EVALUATED", "metric": 1.0})
        with pytest.raises(ValueError, match="immutable"):
            ledger.insert("experiments", "exp", {"status": "EVALUATED", "metric": 0.0})
        ledger.annotate_experiment("exp", {"note": "append-only"})


def test_upstream_absence_is_clean(tmp_path: Path) -> None:
    registry = UpstreamRegistry(
        (
            UpstreamEntry(
                "missing",
                tmp_path / "absent",
                None,
                None,
                ("README.md",),
                (),
                "hybrid_trm_ldt",
                True,
            ),
        )
    )
    assert registry.status()[0]["available"] is False
    receipt = registry.ingest(tmp_path / "context")
    assert receipt["upstreams"]["missing"]["available"] is False
    assert HybridTRMLDTAdapter().compatibility()["root_available"] is False
