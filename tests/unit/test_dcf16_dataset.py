from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import pytest

from datagoblin.datasets import dcf16

jsonschema = pytest.importorskip("jsonschema")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "dcf16"


@pytest.fixture
def splits() -> dict[str, list[dict]]:
    return {
        "val": dcf16.load_split(FIXTURES / "val.jsonl.gz", "val"),
        "heldout": dcf16.load_split(FIXTURES / "heldout.jsonl.gz", "heldout"),
    }


def _write(path: Path, records: list[dict]) -> None:
    raw = "".join(json.dumps(r) + "\n" for r in records).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(raw)


def test_schema_is_valid_draft_2020_12() -> None:
    schema = json.loads(dcf16.SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_validates_fixtures(splits: dict[str, list[dict]]) -> None:
    assert len(splits["val"]) >= 3
    assert len(splits["heldout"]) >= 1
    dcf16.validate_records(splits["val"] + splits["heldout"])


def test_schema_rejects_missing_target_and_unknown_field(splits: dict[str, list[dict]]) -> None:
    broken = copy.deepcopy(splits["val"][0])
    del broken["targets"]["log2_exact_cost"]
    with pytest.raises(ValueError, match="targets"):
        dcf16.validate_records([broken])
    extra = copy.deepcopy(splits["val"][0])
    extra["rogue"] = 1
    with pytest.raises(ValueError, match="rogue"):
        dcf16.validate_records([extra])


def test_loader_rejects_heldout_family_in_training_split(tmp_path: Path, splits: dict[str, list[dict]]) -> None:
    contaminated = copy.deepcopy(splits["heldout"][0])
    assert contaminated["family"] in dcf16.HELDOUT_ONLY_FAMILIES
    contaminated["split"] = "val"
    _write(tmp_path / "val.jsonl.gz", [*splits["val"], contaminated])
    with pytest.raises(dcf16.SplitHygieneError, match="held-out-only family"):
        dcf16.load_split(tmp_path / "val.jsonl.gz", "val")


def test_loader_rejects_oversize_record_in_training_split(tmp_path: Path, splits: dict[str, list[dict]]) -> None:
    oversize = copy.deepcopy(splits["val"][0])
    oversize["n"] = 48
    oversize["id"] = "0" * 20
    _write(tmp_path / "val.jsonl.gz", [oversize])
    with pytest.raises(dcf16.SplitHygieneError, match="training range"):
        dcf16.load_split(tmp_path / "val.jsonl.gz", "val")


def test_loader_rejects_split_tag_mismatch(tmp_path: Path, splits: dict[str, list[dict]]) -> None:
    _write(tmp_path / "train.jsonl.gz", splits["val"])
    with pytest.raises(dcf16.SplitHygieneError, match="tagged"):
        dcf16.load_split(tmp_path / "train.jsonl.gz", "train")


def test_loader_rejects_ids_repeated_across_splits(tmp_path: Path, splits: dict[str, list[dict]]) -> None:
    train = [{**copy.deepcopy(r), "split": "train"} for r in splits["val"]]
    _write(tmp_path / "train.jsonl.gz", train)
    _write(tmp_path / "val.jsonl.gz", splits["val"])
    with pytest.raises(dcf16.SplitHygieneError, match="appears in both"):
        dcf16.load_dataset(tmp_path, ("train", "val"))


def test_loader_rejects_repeated_id_inside_split(tmp_path: Path, splits: dict[str, list[dict]]) -> None:
    _write(tmp_path / "val.jsonl.gz", [splits["val"][0], splits["val"][0]])
    with pytest.raises(dcf16.SplitHygieneError, match="repeats"):
        dcf16.load_split(tmp_path / "val.jsonl.gz", "val")


def test_training_views_exclude_heldout(splits: dict[str, list[dict]]) -> None:
    views = dcf16.training_views(splits)
    assert set(views) == {"val"}
    heldout_ids = {r["id"] for r in splits["heldout"]}
    for rows in views["val"].values():
        assert {row["id"] for row in rows}.isdisjoint(heldout_ids)
        assert {row["split"] for row in rows} <= set(dcf16.TRAINING_SPLITS)


def test_supervision_view_refuses_heldout_without_flag(splits: dict[str, list[dict]]) -> None:
    with pytest.raises(dcf16.SplitHygieneError, match="refused"):
        list(dcf16.supervision_view(splits["heldout"], "algorithm"))
    rows = list(dcf16.supervision_view(splits["heldout"], "algorithm", allow_heldout=True))
    assert len(rows) == len(splits["heldout"])


def test_every_target_field_present(splits: dict[str, list[dict]]) -> None:
    expected = {
        "dependency_edges": {"edges"},
        "relevant_variables": {"relevant_variables"},
        "complexity": {"log2_exact_cost"},
        "algorithm": {"label", "operator"},
        "recurrent_depth": {"recurrent_depth"},
        "calibration": {"p_reference", "exact_action"},
        "abstention": {"abstain", "verified_by"},
    }
    assert set(expected) == set(dcf16.TARGETS)
    views = dcf16.training_views(splits)["val"]
    for target, fields in expected.items():
        assert len(views[target]) == len(splits["val"])
        for row in views[target]:
            assert set(row["target"]) == fields
            assert all(value is not None for value in row["target"].values())
            assert "targets" not in row["input"]
    for record, row in zip(splits["val"], views["abstention"], strict=True):
        assert row["target"]["abstain"] == (record["selected"]["option"] == "abstain")


def test_unknown_target_rejected(splits: dict[str, list[dict]]) -> None:
    with pytest.raises(KeyError):
        list(dcf16.supervision_view(splits["val"], "nonsense"))
