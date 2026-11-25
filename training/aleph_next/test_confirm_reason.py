"""Public fixtures only: no private text, held graph draw or trained checkpoint access."""

import json
import random

import pytest
from confirm_eval import freeze, graph_entries, repair_cues, units
from confirm_reason import TRAIN_WORDS, _orbit, revision, train_repair_rows, train_rule_rows
from confirm_stats import ARMS, SEEDS, compare


def test_primitive_training_orbits_and_bounded_repair():
    assert train_repair_rows(137, 12, 17) == train_repair_rows(137, 12, 17)
    assert len(train_rule_rows(137, 12, 9)) == 9
    rows = train_repair_rows(163, 73, 20)
    assert all("".join("S" if x == 16 else "A" for x in r.prompt[4:-1:2]) in TRAIN_WORDS for r in rows)
    for start in range(0, 20, 4):
        piece = rows[start : start + 4]
        assert len({r.root for r in piece}) == 1
        assert {r.good[0] for r in piece} == set(range(4))
        assert sum(r.prompt[1] == r.good[0] for r in piece) == 1
    public = _orbit("SAA", random.Random(4), "unit-test")
    assert all(repair_cues(r)["anchor_then_two_tail_adds"] == r.good[0] for r in public)


def test_revision_stale_versus_complete_checked_controls():
    panel = graph_entries(list(revision.episodes(2026, "train", 4)), "revision", "toy-train-only")
    scored = [{**e["row"], "known": True, "hit": True} for e in panel]
    values = units(panel, scored)
    assert len(values) == 4
    for item in values:
        assert item["scores"]["model"] == 1
        assert item["scores"]["stale_original_walk"] == 0.5
        assert item["scores"]["patch_with_one_tail_hop"] >= 0.5
    with pytest.raises(ValueError, match="incomplete"):
        units(panel[1:], scored[1:])


def test_failed_development_leaves_panel_unopened(tmp_path):
    manifest = tmp_path / "f.json"
    manifest.write_text(json.dumps({"reference_competent": False}), encoding="utf-8")
    out = tmp_path / "sealed"
    with pytest.raises(ValueError, match="leave fresh panel unopened"):
        freeze(manifest, tmp_path, tmp_path, out)
    assert not (out / "private.seed").exists()
    assert not (out / "freeze.json").exists()


def test_frozen_public_fixture_roster_checks_endpoint_lineage_and_receipt(tmp_path, monkeypatch):
    import confirm_eval as ev

    monkeypatch.setattr(ev, "sources", lambda: {"fixture": "public"})
    panel, pack, out = (tmp_path / n for n in ("text", "pack", "gate"))
    panel.mkdir()
    pack.mkdir()
    (panel / "manifest.json").write_text("{}")
    (pack / "dataset.json").write_text(json.dumps({"schema": "datagoblin.mixed-pretrain-pilot.v1", "files": {}}))
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            base = tmp_path / f"{arm}-{seed}"
            base.mkdir()
            weights = base / "h0064.pt"
            weights.write_bytes(f"public fixture {arm} {seed}".encode())
            roots = base / "h0064-roots.json"
            roots.write_text(json.dumps({"repair": ["c" * 64]}))
            meta = weights.with_suffix(".json")
            meta.write_text(
                json.dumps(
                    dict(
                        arm=arm,
                        seed=seed,
                        step=64,
                        checkpoint_sha256=ev.sha(weights),
                        params=14179584,
                        visits=24,
                        roots=roots.name,
                        roots_sha256=ev.sha(roots),
                        stream=str(seed),
                        tokens=9,
                        padded=10,
                        labels=7,
                        prefix_sha="test-parent",
                        prefix_stream="toy",
                        prefix_tokens=2,
                        prefix_padded=2,
                        recipe="toy",
                        corpus="toy",
                    )
                )
            )
            records.append(
                dict(
                    arm=arm,
                    seed=seed,
                    step=64,
                    checkpoint=str(weights),
                    sha256=ev.sha(weights),
                    receipt_sha256=ev.sha(meta),
                    parent_sha256="test-parent",
                )
            )
    manifest = tmp_path / "fits.json"
    manifest.write_text(
        json.dumps(
            dict(
                reference_competent=True,
                candidate_competent=False,
                development_competence={"fixture": True},
                fits=records,
            )
        )
    )
    ev.freeze(manifest, panel, pack, out)
    assert not (out / "private.seed").exists() and not ev.verify_freeze(out)["candidate_development_competent"]
    meta.write_text("{}")
    with pytest.raises(ValueError, match="frozen endpoint receipt changed"):
        ev.verify_freeze(out)
    with pytest.raises(ValueError, match="differs from the canonical fit roster"):
        ev.freeze(manifest, panel, pack, tmp_path / "second")


def synthetic_scores(gain):
    docs = {f"d{i}": str(i % 3) for i in range(12)}
    scores = {}
    baselines = {
        k: {"per_document": dict.fromkeys(docs, v)} for k, v in (("unigram", 3.4), ("interpolated_trigram", 3.1))
    }
    for arm in ARMS:
        for seed in SEEDS:
            model = []
            b = arm == ARMS[1]
            for lane in ("repair", "revision", "archived_revision"):
                for f in range(4 if lane == "repair" else 2):
                    for root in range(3):
                        model.append(
                            {
                                "task": lane,
                                "family": str(f),
                                "template": "x",
                                "root": f"fixture-{f}-{root}",
                                "scores": {"model": 0.7 + (gain if b else 0), "shallow_cue": 0.5, "naive_cue": 0.2},
                                "diagnostics": {"public_fixture": 1},
                            }
                        )
            scores[f"{arm}-{seed}"] = {
                "clusters": model,
                "text": {"per_document": dict.fromkeys(docs, 2.80 if b else 2.85), "panel_sha256": "fixture"},
            }
    return scores, docs, baselines


def test_paired_clusters_routes_and_genuine_noninferiority():
    scores, docs, baselines = synthetic_scores(0.05)
    r = compare(scores, docs, baselines, draws=200)
    assert r["held_gates_passed"] and all(r["benefit_routes"].values())
    assert r["intervals"]["primary"]["checked_gain"] == pytest.approx([0.05, 0.05])
    assert all(x["qualifying_checked_tasks"] == 2 for x in r["held_competence"].values())
    scores, docs, baselines = synthetic_scores(-0.05)
    r = compare(scores, docs, baselines, draws=200)
    assert r["benefit_routes"]["text"] and not r["safeguards"]["checked_within_2pp"]
    assert not r["held_gates_passed"]
    scores["direct-163"]["clusters"][0]["root"] = "substituted fixture"
    with pytest.raises(ValueError, match="unmatched"):
        compare(scores, docs, baselines, draws=200)
