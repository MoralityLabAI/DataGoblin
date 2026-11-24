"""Focused regression checks for source custody, mask alignment and exact restart."""

import hashlib
import json
import sqlite3

import pytest
import torch
from gym import Aleph
from packing import IGNORE, INDEX_SCHEMA, PackedStream, reference_logits, sdpa_mask, token_loss_sum


def encoded(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def row(root, tokens, mask, objective="next_token", start=None, receipt=None):
    data = {
        "root": root,
        "family": "family-" + root,
        "split": "train",
        "presentation": 1,
        "lane": objective,
        "objective": objective,
        "token_ids": tokens,
        "target_mask": mask,
        "receipt_sha256": receipt,
    }
    if start is not None:
        data["response_start"] = start
    return data


def make_pack(tmp_path, *, bad_split=False, bad_prompt=False, missing_receipt=False):
    a = row("doc", [2, 3, 4], [False, True, True])
    b = row(
        "goal",
        [7, 8, 9, 10, 11, 12],
        [False, bad_prompt, True, True, False, True],
        "checked_response",
        2,
        None if missing_receipt else "c" * 64,
    )
    c = row("unknown", [5, 14], [False, False], "checked_response", 1, None)
    db = tmp_path / "registry.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(INDEX_SCHEMA)
    specs = []
    for n, records in enumerate(([a, b], [c])):
        path = tmp_path / f"rows-{n}.jsonl"
        blob = b"".join(encoded(it) + b"\n" for it in records)
        path.write_bytes(blob)
        digest = sha(blob)
        specs.append({"path": path.name, "sha256": digest, "records": len(records), "split": "train"})
        for i, rec in enumerate(records):
            conn.execute(
                "INSERT INTO families VALUES(?,?)", (rec["family"], "test" if bad_split and i == 1 else "train")
            )
            conn.execute("INSERT INTO roots VALUES(?,?)", (rec["root"], rec["family"]))
            conn.execute(
                "INSERT INTO views VALUES(?,?,?,?,?,?)",
                (rec["root"], 1, digest, i, sha(encoded(rec)), rec["receipt_sha256"]),
            )
    conn.commit()
    conn.close()
    tokens = tmp_path / "tokenizer.json"
    tokens.write_bytes(b"test-only token codec")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        encoded(
            {
                "schema": "aleph.packed-stream.v1",
                "vocab_size": 32,
                "context": 5,
                "pad_id": 0,
                "tokenizer": {"path": tokens.name, "sha256": sha(tokens.read_bytes())},
                "registry": {"path": db.name, "sha256": sha(db.read_bytes())},
                "shards": specs,
            }
        )
    )
    return manifest, sha(manifest.read_bytes())


def test_interrupted_stream_matches_at_each_registered_stop(tmp_path):
    path, digest = make_pack(tmp_path)
    with PackedStream(path, digest, "train") as stream:
        first = stream.next_pack(7)
        partial = stream.next_pack(7)
        assert first["segment_ids"] == [0, 0, 0, 1, 1]
        assert first["labels_at_query"] == [3, 4, IGNORE, IGNORE, IGNORE]
        assert partial["input_ids"] == [9, 10, 0, 0, 0]
        assert partial["labels_at_query"] == [10, IGNORE, IGNORE, IGNORE, IGNORE]
        assert partial["position_ids"] == [0, 1, 0, 0, 0]
        assert stream.next_pack(7) is None
        cursor = stream.checkpoint()
        assert cursor["payload"]["state"]["within"] == 4
        reference = stream.next_pack(11)
        final = stream.checkpoint()
    with PackedStream(path, digest, "train", cursor) as restored:
        assert restored.next_pack(11) == reference
        assert restored.checkpoint() == final
        assert restored.next_pack(12) is None
        assert restored.state["processed"] == 11
        assert restored.state["targets"] == 4
        assert restored.state["padding"] == 4
        assert restored.state["started_views"] == 3
        assert restored.state["canonical_view_positions"] == 11
        assert restored.state["positions_by_lane"] == {"next_token": 3, "checked_response": 8}
        assert restored.state["targets_by_lane"] == {"next_token": 2, "checked_response": 2}


def test_prefix_segment_and_padding_attention():
    mask = sdpa_mask([[0, 0, 1, 1, -1], [0, 1, 1, -1, -1]])
    assert mask.shape == (2, 1, 5, 5)
    assert mask[0, 0].int().tolist() == [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ]
    x = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)
    z = torch.zeros_like(x)
    before = torch.nn.functional.scaled_dot_product_attention(z, z, x, attn_mask=mask[:1])
    x[:, :, :2] += 50
    after = torch.nn.functional.scaled_dot_product_attention(z, z, x, attn_mask=mask[:1])
    assert torch.equal(before[:, :, 2:], after[:, :, 2:])


def test_segmentwise_logits_and_loss_do_not_cross_sources(tmp_path):
    path, digest = make_pack(tmp_path)
    with PackedStream(path, digest, "train") as stream:
        pack = stream.next_pack(5)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = Aleph(16, 32)
    actual = reference_logits(model, pack, r=1)
    left = model(torch.tensor([[2, 3, 4]]), r=1)
    right = model(torch.tensor([[7, 8]]), r=1)
    assert torch.equal(actual, torch.cat((left, right), dim=1))
    total, count = token_loss_sum(actual, pack["labels_at_query"])
    assert int(count) == 2
    loss = total / count
    assert torch.equal(loss, torch.nn.functional.cross_entropy(left[0, :2], torch.tensor([3, 4])))
    loss.backward()
    assert model.embed.weight.grad is not None and torch.isfinite(model.embed.weight.grad).all()


def test_all_unknown_pack_has_zero_loss_and_gradient():
    logits = torch.zeros((1, 3, 32), requires_grad=True)
    total, count = token_loss_sum(logits, [IGNORE] * 3)
    assert total.item() == 0 and count.item() == 0
    total.backward()
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize(
    ("flag", "error"),
    [
        ("bad_split", "custody"),
        ("bad_prompt", "label in prompt"),
        ("missing_receipt", "lacks receipt"),
    ],
)
def test_unadmitted_labels_never_advance_cursor(tmp_path, flag, error):
    path, digest = make_pack(tmp_path, **{flag: True})
    with PackedStream(path, digest, "train") as stream:
        initial = stream.checkpoint()
        with pytest.raises(ValueError, match=error):
            stream.next_pack(5)
        assert stream.checkpoint() == initial


def test_tokenizer_shard_and_checkpoint_digests(tmp_path):
    path, digest = make_pack(tmp_path)
    with pytest.raises(ValueError, match="manifest digest"):
        PackedStream(path, "0" * 64, "train")
    with PackedStream(path, digest, "train") as stream:
        cursor = stream.checkpoint()
        cursor["payload"]["state"]["byte"] = 8
        with pytest.raises(ValueError, match="cursor checksum"):
            PackedStream(path, digest, "train", cursor)
        shard = tmp_path / "rows-0.jsonl"
        shard.write_bytes(shard.read_bytes() + b" ")
        with pytest.raises(ValueError, match="shard digest"):
            stream.next_pack(5)
    (tmp_path / "tokenizer.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="tokenizer digest"):
        PackedStream(path, digest, "train")
