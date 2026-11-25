"""Public synthetic fixtures for the private text confirmation reader and reducer."""

import math

import pytest

from training.aleph_next import confirm_text as text
from training.aleph_next import confirm_text_check as independent


@pytest.fixture
def fake_panel(tmp_path):
    rows = []
    for doc in text.pins():
        for block in range(text.BLOCKS):
            raw = bytes(range(block, block + text.WINDOW))
            offset = block * text.WINDOW
            rows.append(
                dict(
                    doc=doc["id"], block=block, offset=offset, hex=raw.hex(), root=text.root_id(doc["id"], offset, raw)
                )
            )
    content = b"".join(text.encode(r) for r in rows)
    manifest = text.encode(
        dict(
            schema=text.SCHEMA,
            source_pins_sha256=text.sha(text.encode(text.pins())),
            native_sha256=text.NATIVE,
            windows_sha256=text.sha(content),
        )
    )
    (tmp_path / "windows.jsonl").write_bytes(content)
    (tmp_path / "manifest.json").write_bytes(manifest)
    return tmp_path, text.sha(manifest)


def test_native_compatibility_and_completeness(fake_panel):
    panel, checksum = fake_panel
    rows = text.load_rows(panel, checksum)
    assert all(len(r.prompt) == len(r.good) == 32 and r.prompt[1:] == r.good[:-1] for r in rows)
    scores = [dict(lane="text", root=r.root, group=r.group, order=r.order, positions=32, nll=64.0) for r in rows]
    result = text.document_scores(reversed(scores), panel, checksum)
    assert (result["documents"], result["windows"], result["supervised_bytes"]) == (12, 384, 12288)
    assert set(result["per_document"].values()) == {2.0}
    with pytest.raises(ValueError):
        text.document_scores(scores[1:], panel)
    with pytest.raises(ValueError):
        text.document_scores([*scores[1:], scores[-1]], panel)
    with pytest.raises(ValueError):
        text.document_scores([*scores[:-1], dict(scores[-1], nll=math.nan)], panel)
    with pytest.raises(ValueError):
        text.load_rows(panel, "wrong")
    (panel / "windows.jsonl").write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        text.load_rows(panel, checksum)


def test_bootstrap_pairs_seed_then_document_and_preserves_strata():
    docs = text.pins()
    groups = {"fiction": 0.3, "observation": -0.1, "nonfiction": 0.1}

    def item(shift, effect):
        return dict(
            panel_sha256="fixture",
            per_document={d["id"]: 2 + shift + (groups[d["stratum"]] if effect else 0) for d in docs},
        )

    a = {1: item(0, False), 2: item(2, False)}
    b = {1: item(0.2, True), 2: item(1.6, True)}
    comparison = text.compare(a, b, draws=100)
    assert comparison["paired_B_minus_A_nats_per_byte"] == pytest.approx(0)
    assert comparison["paired_document_interval_95"] == pytest.approx([0, 0])
    assert comparison["per_seed"] == pytest.approx({1: 0.3, 2: -0.3})
    with pytest.raises(ValueError):
        text.compare(a, {1: b[1]})
    with pytest.raises(ValueError):
        text.compare(a, {1: b[1], 2: dict(b[2], panel_sha256="different")})


def test_normalized_overlap_and_train_only_baselines(fake_panel, monkeypatch):
    assert not text.overlap_keys(b"Some REPEATED, passage here.").isdisjoint(
        text.overlap_keys(b"some repeated -- passage here.")
    )
    panel, checksum = fake_panel
    training = [bytes(range(text.WINDOW))]

    def unique_train(_):
        return {"text-train.jsonl": training, "text-held.jsonl": [bytes([255]) * text.WINDOW]}

    monkeypatch.setattr(text, "native_bytes", unique_train)
    scores = text.baselines(None, panel, checksum)
    uni = scores["unigram"]["nats_per_byte"]
    expected = (
        -sum(math.log((int(t in range(1, 33)) + 1) / 288) for i in range(32) for t in range(i + 1, i + 33)) / 1024
    )
    assert uni == pytest.approx(expected)
    assert scores["interpolated_trigram"]["nats_per_byte"] < uni


def test_exact_author_bootstrap_reduces_to_binomial_when_one_author_changes():
    groups = {doc["id"]: doc["stratum"] for doc in text.pins()}
    base = dict.fromkeys(groups, 2.0)
    changed = dict(base)
    changed[next(iter(changed))] = 1.8
    out = independent.exact_document_bootstrap(base, changed, groups)
    # The number of draws of the sole changed author is exactly Binomial(4, 1/4).
    # Its 2.5% and 97.5% population quantiles are 0 and 3.
    assert out["equivalent_ordered_draws"] == 4**12 and out["count_vectors"] == 35**3
    assert out["relative_gain_interval_95"] == pytest.approx([0, 0.025])
    assert out["B_minus_A_nats_interval_95"] == pytest.approx([-0.05, 0])
