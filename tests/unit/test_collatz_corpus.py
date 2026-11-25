from __future__ import annotations

import copy
import json
from math import gcd

import pytest

from datagoblin.training_gym.collatz_corpus import (
    admitted_record,
    audit_directory,
    audit_records,
    generate,
    generate_records,
    make_certificate,
)
from datagoblin.training_gym.collatz_verifier import parameter_split, verify_certificate


def test_known_two_step_cycle_and_full_map_cost():
    cert = make_certificate(13, 5)
    assert cert["odd_states"] == [5, 7, 5]
    assert cert["edges"] == [{"halvings": 2, "triple": 1}, {"halvings": 2, "triple": 0}]
    assert cert["claim"] == {"cycle_minimum": 5, "odd_steps": 2, "halvings": 4,
                              "triple_branches": 1, "period": 6, "gcd_with_d": 1}
    receipt = verify_certificate(cert)
    assert receipt["checked_full_map_steps"] == 6
    assert receipt["scope"] == "finite_unaccelerated_replay"


def test_exhaustive_small_core_replays():
    records = []
    for d in range(1, 258, 4):
        if gcd(d, 6) != 1:
            continue
        for n in range(1, d + 1, 2):
            row = admitted_record(d, n)
            assert row["certificate"]["claim"]["gcd_with_d"] == gcd(n, d)
            assert row["admission"]["cycle_minimum"] >= 1
            records.append(row)
    assert len(records) == 2860
    # General replay includes nonunits. Audit per root because scaled traces
    # across arbitrary parameters can deliberately fall in different splits.
    for d in {row["certificate"]["d"] for row in records}:
        audit_records([row for row in records if row["certificate"]["d"] == d])


@pytest.mark.parametrize("mutation", [
    "next_state", "halvings", "branch", "minimum", "period", "cycle_start",
    "split", "root", "truncate", "duplicate_cycle", "bool_d", "bool_a",
])
def test_checker_rejects_corrupted_certificates(mutation):
    cert = make_certificate(13, 5)
    if mutation == "next_state":
        cert["odd_states"][1] = 9
    elif mutation == "halvings":
        cert["edges"][0]["halvings"] += 1
    elif mutation == "branch":
        cert["edges"][0]["triple"] = 0
    elif mutation == "minimum":
        cert["claim"]["cycle_minimum"] = 1
    elif mutation == "period":
        cert["claim"]["period"] = 7
    elif mutation == "cycle_start":
        cert["cycle_start"] = 1
    elif mutation == "split":
        cert["split"] = "fake"
    elif mutation == "root":
        cert["root_id"] = "d:5"
    elif mutation == "truncate":
        cert["odd_states"].pop()
    elif mutation == "duplicate_cycle":
        cert["odd_states"] = [5, 7, 5, 7, 5]
        cert["edges"] *= 2
    elif mutation == "bool_d":
        cert["d"] = True
    elif mutation == "bool_a":
        cert["edges"][0]["halvings"] = True
    with pytest.raises(ValueError):
        verify_certificate(cert)


def test_resource_bounds_and_out_of_core_seed():
    cert = make_certificate(13, 10_001)
    assert cert["cycle_start"] > 0
    verify_certificate(cert)
    with pytest.raises(ValueError):
        verify_certificate(cert, max_edges=1)
    with pytest.raises(ValueError):
        make_certificate(13, 5, max_edges=1)
    for d in [0, 9, 6, True, 5.0]:
        with pytest.raises(ValueError):
            parameter_split(d)


@pytest.mark.parametrize("mutation", ["target", "input", "receipt", "split", "id", "bool_input", "scope"])
def test_dataset_admission_is_replayed(mutation):
    row = admitted_record(1, 1)
    if mutation == "target":
        row["target"]["cycle_minimum"] = 7
    elif mutation == "input":
        row["input"]["n"] = 5
    elif mutation == "receipt":
        row["admission"]["status"] = "proposer_says_verified"
    elif mutation == "split":
        row["split"] = "other"
    elif mutation == "id":
        row["id"] = "forged"
    elif mutation == "bool_input":
        row["input"]["n"] = True
    elif mutation == "scope":
        row["supervision_scope"] = "all seeds and all parameters"
    with pytest.raises(ValueError):
        audit_records([row])


def test_duplicate_task_and_scaled_cross_split_rejected():
    one = admitted_record(13, 5)
    with pytest.raises(ValueError):
        audit_records([one, copy.deepcopy(one)])
    scale = next(g for g in range(5, 202, 4)
                 if gcd(g, 6) == 1 and parameter_split(13 * g) != parameter_split(13))
    scaled = admitted_record(13 * scale, 5 * scale)
    with pytest.raises(ValueError, match="scaled duplicate"):
        audit_records([one, scaled])


def test_generate_is_deterministic_disjoint_and_roundtrips(tmp_path):
    records, audit = generate_records(257, 4)
    again, audit_again = generate_records(257, 4)
    assert records == again
    assert audit == audit_again
    assert audit["holdout_roots_skipped"] > 0
    assert all(row["split"] != "holdout" for row in records)
    assert all(gcd(row["input"]["d"], row["input"]["n"]) == 1 for row in records)
    roots = {split: {row["root_id"] for row in records if row["split"] == split}
             for split in ("train", "dev")}
    assert roots["train"].isdisjoint(roots["dev"])
    generate(tmp_path, 257, 4)
    checked = audit_directory(tmp_path)
    assert checked["canonical_dataset_sha256"] == audit["canonical_dataset_sha256"]
    assert generate(tmp_path, 257, 4) == audit
    with pytest.raises(ValueError, match="existing evaluated export"):
        generate(tmp_path, 257, 5)


def test_manifest_tampering_rejected(tmp_path):
    generate(tmp_path, 257, 4)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["records"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        audit_directory(tmp_path)


def test_generator_bounds():
    for limit in [0, 4097, True]:
        with pytest.raises(ValueError):
            generate_records(limit)
    for cap in [0, 33, True]:
        with pytest.raises(ValueError):
            generate_records(257, cap)
