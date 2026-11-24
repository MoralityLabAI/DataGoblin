"""Conformance of the compact placement views, loss and provenance boundary."""
import hashlib
import importlib
import io
import json
import math
import random
import zipfile
from pathlib import Path

import pytest

LAB = Path(__file__).resolve().parents[2] / "training" / "aleph_rlt_placement"


@pytest.fixture
def corpus(monkeypatch):
    monkeypatch.syspath_prepend(str(LAB))
    return importlib.import_module("corpus")


@pytest.fixture
def study(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(LAB))
    return importlib.import_module("study")


def test_shifted_byte_windows_and_typed_actor_view(corpus):
    row, = corpus.text_rows([{"id": "w", "corpus": "fixture", "tokens": [0, 64, 255]}])
    assert row.public() == (0, 64)
    assert row.good == (64, 255)
    with pytest.raises(ValueError, match="byte"):
        list(corpus.text_rows([{"id": "w", "tokens": [0, -1]}]))


def test_agonist_semantics_and_lineage_holdout(corpus):
    rng = random.Random(103)
    modes, no_witness, multiple = set(), False, False
    for _ in range(200):
        repair = corpus.generate_repair(rng)
        prompt = repair.public()
        trace, state = prompt[2:-5], prompt[1]
        for i in range(0, len(trace), 3):
            opcode, arg, submitted = trace[i:i + 3]
            corrected = (state + arg) % 4 if opcode % 2 else arg
            if opcode >= 18:
                assert submitted != corrected
                assert repair.good == (corrected,)
                modes.add(opcode)
                break
            assert corrected == submitted
            state = corrected
        else:
            pytest.fail("repair has no marked first fault")
        row = corpus.generate_counter(rng)
        tag, pos_a, neg_a, sep1, pos_b, neg_b, sep2 = row.public()
        assert (tag, sep1, sep2) == (corpus.COUNTER, corpus.SEP, corpus.SEP)
        a = {k for k in range(32) if k & (pos_a | neg_a) == pos_a}
        b = {k for k in range(32) if k & (pos_b | neg_b) == pos_b}
        assert set(row.good) == (a ^ b or {corpus.NONE})
        no_witness |= a == b
        multiple |= len(a ^ b) > 1
    assert modes == {18, 19} and no_witness and multiple
    for generator in (corpus.generate_counter, corpus.generate_repair):
        train, held = corpus.agonist_rows(generator, 80, 30)
        assert len({row.root for row in [*train, *held]}) == 110
        assert not {row.prompt for row in train} & {row.prompt for row in held}


def test_frozen_rule_twin_inputs_and_independent_endpoints(corpus):
    examples = list(corpus.revision.episodes(71, "development", 1))
    rows = list(corpus.rule_rows(examples))
    assert len({row.root for row in rows}) == 1
    assert rows[0].prompt == rows[2].prompt and rows[0].good == rows[2].good
    assert rows[1].prompt != rows[3].prompt
    task, original = examples[0]
    target = {**original, "after": "bad-endpoint"}
    with pytest.raises(ValueError, match="replay"):
        list(corpus.rule_rows([(task, target)]))


def test_constructive_controls_match_enumerated_mask_and_repair_oracles(corpus):
    controls = importlib.import_module("control_audit")
    rng = random.Random(23)
    for _ in range(128):
        repair = corpus.generate_repair(rng)
        assert controls.repair_prediction(repair.public()) in repair.good
        counter = corpus.generate_counter(rng)
        cue, exact = controls.counter_predictions(counter.public())
        assert 0 <= cue <= corpus.NONE and exact in counter.good
    # Both canonical positive masks pass both rules. Adding a forbidden bit
    # separates the rules even though neither unmodified mask can witness it.
    prompt = (corpus.COUNTER, 1, 2, corpus.SEP, 1, 4, corpus.SEP)
    assert controls.counter_predictions(prompt) == (1, 3)



def test_lean_keeps_unknown_in_rank_budget_and_joins_epoch(corpus, study):
    candidates = [{"id": str(i), "tactic": f"apply t{i}"} for i in range(5)]
    order = [item["id"] for item in candidates]
    public = {"epoch": "e", "id": "goal", "lineage": "family", "statement": "x = x",
              "candidates": candidates, "orders": dict.fromkeys(("compose", "wider", "legacy"), order)}
    grades = {"id": "goal", "epoch": "e", "outcomes": {
        **{str(i): {"label": None if i == 0 else 0} for i in range(4)},
        "4": {"label": 1, "closed": True, "kernel_ok": True, "kernel_calls": 1, "axioms": []},
    }}
    rows = list(corpus.lean_rows([(public, grades)]))
    assert [row.good for row in rows] == [(), (0,), (0,), (0,), (1,)]
    result = study.metrics([{"lane": "lean", "root": row.root, "order": row.order,
                             "score": 0, "grade": row.good[0] if row.good else None} for row in rows])
    assert result == {"lean_model": 0, "lean_lexical": 0, "lean_pool": 1}
    with pytest.raises(ValueError, match="stale"):
        list(corpus.lean_rows([(public, {**grades, "epoch": "past"})]))
    grades["outcomes"]["4"]["kernel_ok"] = False
    with pytest.raises(ValueError, match="unverified"):
        list(corpus.lean_rows([(public, grades)]))


def test_weighted_loss_masks_padding_and_accepts_every_witness(corpus, study):
    import torch

    # Binary task has two valid witnesses out of four actions, one per tested row.
    rows = [corpus.Row("text", "t", (5, 7), (7, 9), 256),
            corpus.Row("counter", "c", (corpus.COUNTER,), (0, 3), 4)]
    logits = torch.zeros(2, 2, corpus.VOCAB, dtype=torch.float64, requires_grad=True)
    loss = study.objective(logits, rows)
    wt = study.WEIGHTS["text"]
    wc = study.WEIGHTS["counter"]
    assert loss.item() == pytest.approx((wt * math.log(corpus.VOCAB) + wc * math.log(2)) / (wt + wc))
    loss.backward()
    assert torch.count_nonzero(logits.grad[1, 1]) == 0
    assert torch.count_nonzero(logits.grad[1, 0, 4:]) == 0
    assert logits.grad[1, 0, 0] == logits.grad[1, 0, 3] < 0
    assert logits.grad[1, 0, 1] == logits.grad[1, 0, 2] > 0


def zip_bytes(files):
    blob = io.BytesIO()
    with zipfile.ZipFile(blob, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return blob.getvalue()


def test_verifies_outer_envelope_and_refuses_changed_members(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(LAB))
    module = importlib.import_module("unpack_evidence")
    members = {"mixed/dataset.json": json.dumps({"schema": "fixture", "files": {"example": "x"}}).encode(),
               "mixed/revision/manifest.json": b'{"splits": {}}', "mixed/example": b"original"}
    manifest = {"files": {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}}
    members["bundle_manifest.json"] = json.dumps(manifest).encode()
    raw = zip_bytes(members)
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(module, "BLOB", blob)
    archive = tmp_path / "source.zip"
    out = tmp_path / "out"
    archive.write_bytes(zip_bytes({module.MEMBER: raw}))
    assert module.unpack(archive, out)["git_blob"] == blob
    assert (out / "example").read_bytes() == b"original"
    (out / "example").write_bytes(b"modified")
    with pytest.raises(ValueError, match="overwrite"):
        module.unpack(archive, out)
    members["mixed/example"] = b"different"
    raw = zip_bytes(members)
    archive.write_bytes(raw)
    with pytest.raises(ValueError, match="pinned Git blob"):
        module.unpack(archive, out)
    monkeypatch.setattr(module, "BLOB", hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest())
    with pytest.raises(ValueError, match="digest changed"):
        module.unpack(archive, out)
