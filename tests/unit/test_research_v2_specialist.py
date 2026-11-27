"""Test intake isolation with synthetic policies and source bytes."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from datagoblin.research_v2 import specialist as subject


@pytest.fixture
def intake(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    policy_text = """(benchmark fixture)
(constitution fixture)
(critical-cap 0.2)
(dimension criterion 1.0)
(tag-score {prefix}_positive criterion 1.0)
(tag-score {prefix}_negative criterion -1.0)
(critical-tag {prefix}_negative)
"""
    manifest = {
        "source_id": subject.SOURCE_ID,
        "schema": "research-v2-specialist-sources-v1",
        "visibility": "private",
        "use": "internal_specialist_training",
        "public_redistribution": "unknown",
        "revision": "fixture-revision",
        "claim_scope": "Fixture policy emulation",
        "expected_rows": 5,
        "expected_splits": {"train": 3, "validation": 1, "test": 1},
        "source_row_schema": "two_frame_tagged_candidate_v1",
        "train_minimum": "0.7",
        "reject_maximum": "0.3",
        "files": {},
    }
    policies = {}
    for frame, prefix in (("jinn", "a"), ("beast", "b")):
        data = policy_text.format(prefix=prefix).encode()
        (source / f"{frame}.metta").write_bytes(data)
        manifest["files"][frame] = {
            "path": f"{frame}.metta", "bytes": len(data), "sha256": subject.digest(data),
        }
        policies[frame] = subject._policy(data)
    rows = []
    for split, tags in (
        ("train", ["a_positive"]),
        ("train", ["a_negative", "b_positive"]),
        ("train", ["a_positive", "b_negative"]),
        ("validation", ["b_positive"]),
        ("test", ["b_negative"]),
    ):
        labels = {name: subject._score(tags, policy, Fraction("0.3"), Fraction("0.7"))
                  for name, policy in policies.items()}
        rows.append({
            "row_id": "mb-" + subject.semantic_digest(tags)[:16],
            "tags": tags,
            "schema_version": manifest["source_row_schema"],
            "split": split,
            "labels": labels,
            "joint_label": {name: label["lane"] for name, label in labels.items()},
        })

    def repin_rows():
        data = "".join(subject.canonical(row) + "\n" for row in rows).encode()
        (source / "rows.jsonl").write_bytes(data)
        manifest["files"]["dataset"] = {"path": "rows.jsonl", "bytes": len(data), "sha256": subject.digest(data)}

    repin_rows()
    monkeypatch.setattr(subject, "_load_manifest", lambda path: manifest)
    registry = tmp_path / "held.json"
    registry_value = {
        "schema": subject.REGISTRY_SCHEMA,
        "held_root_ids": [],
        "held_semantic_sha256": [],
        "allow_source_ids": [subject.SOURCE_ID],
    }
    registry.write_text(json.dumps(registry_value))
    return source, tmp_path / "output", registry, rows, repin_rows, registry_value


def test_preserves_original_splits_and_binds_output(intake):
    source, out, registry, rows, _, _ = intake
    receipt = subject.import_specialist(source, out, registry)
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert receipt["training_targets"] == 6
    assert receipt["admitted_train_roots"] == 3
    assert receipt["all_source_labels_recomputed"]
    assert {row["split"] for row in records} == {"train"}
    assert {row["root_id"] for row in records} == {subject.root_id(row["tags"]) for row in rows[:3]}
    assert receipt["files"]["train.jsonl"]["sha256"] == subject.digest((out / "train.jsonl").read_bytes())
    held = json.loads((out / "held_roots.json").read_text())
    assert set(held["held_semantic_sha256"]) == {subject.semantic_digest(row["tags"]) for row in rows[3:]}
    assert {path.name for path in out.iterdir()} == {"train.jsonl", "held_roots.json", "RECEIPT.json"}


def test_fails_closed_on_changed_source_bytes(intake):
    source, out, registry, *_ = intake
    with (source / "rows.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Pinned source mismatch"):
        subject.import_specialist(source, out, registry)
    assert not out.exists()


def test_independent_label_check_rejects_even_repinned_bad_label(intake):
    source, out, registry, rows, repin, _ = intake
    rows[0]["labels"]["jinn"]["lane"] = "reject"
    repin()
    with pytest.raises(ValueError, match="Independent policy check"):
        subject.import_specialist(source, out, registry)
    assert not out.exists()


def test_prospective_held_root_excludes_all_frame_targets_and_variants(intake):
    source, out, registry, rows, _, registry_value = intake
    blocked = subject.root_id(rows[1]["tags"])
    registry_value["held_root_ids"] = [blocked]
    registry.write_text(json.dumps(registry_value))
    receipt = subject.import_specialist(source, out, registry, augment=True)
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert blocked not in {row["root_id"] for row in records}
    assert receipt["admitted_train_roots"] == 2
    assert receipt["excluded_views"]["original"] == 2


def test_derived_views_cannot_reintroduce_original_holdout(intake):
    source, out, registry, rows, *_ = intake
    receipt = subject.import_specialist(source, out, registry, augment=True)
    records = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    held = {subject.semantic_digest(row["tags"]) for row in rows[3:]}
    assert not held & {row["semantic_sha256"] for row in records}
    assert receipt["excluded_views"]["own_frame_view"] == 2
    assert {row["root_id"] for row in records} <= {subject.root_id(row["tags"]) for row in rows[:3]}
    assert len({(row["policy_frame"], row["semantic_sha256"]) for row in records}) == len(records)


def test_duplicate_tag_family_cannot_cross_splits(intake):
    source, out, registry, rows, repin, _ = intake
    rows[-1] = {**rows[0], "split": "test"}
    repin()
    with pytest.raises(ValueError, match="Duplicate tag family"):
        subject.import_specialist(source, out, registry)


def test_requires_explicit_source_allowance(intake):
    source, out, registry, _, _, registry_value = intake
    registry_value["allow_source_ids"] = []
    registry.write_text(json.dumps(registry_value))
    with pytest.raises(ValueError, match="does not allow"):
        subject.import_specialist(source, out, registry)


def test_refuses_existing_output(intake):
    source, out, registry, *_ = intake
    out.mkdir()
    (out / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="fresh directory"):
        subject.import_specialist(source, out, registry)
    assert (out / "precious.txt").read_text() == "keep"
