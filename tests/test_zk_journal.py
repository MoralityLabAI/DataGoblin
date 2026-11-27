import hashlib
from pathlib import Path

import pytest

from datagoblin import zk_frontier as frontier
from datagoblin import zk_journal as journal
from datagoblin.zk_replay import Job, Plan, execute
from datagoblin.zk_schedule import PALLAS_BASE, PALLAS_SCALAR, next_state
from datagoblin.zk_sigma import r1cs_satisfied


def plan_for(n, w, mode):
    jobs = []
    for i in range(n):
        if mode == "chain":
            parents = [i-1] if i else []
        elif mode == "window":
            parents = list(range(max(0, i-2), i))
        else:
            parents = list(range(i)) if i == n-1 else []
        source = tuple(127 if (i+j) % 2 else -127 for j in range(w))
        weights = tuple(tuple(2 if j == k else -1 for k in range(w)) for j in range(w))
        jobs.append(Job(f"j{i}", tuple(f"j{p}" for p in parents), source, weights, (0,)*w, 1+i % 4))
    return Plan("tenant", f"{mode}-{n}-{w}", "journal", tuple(jobs))


@pytest.mark.parametrize("n", [1, 8, 9, 17, 32])
@pytest.mark.parametrize("w", [1, 2])
@pytest.mark.parametrize("mode", ["chain", "window", "star"])
def test_full_journal(n, w, mode):
    plan = plan_for(n, w, mode)
    rows = journal.schedule(plan)
    expected = tuple(r["trace"][-1] for r in execute(plan))
    base = frontier.schedule(plan)[0]
    limbs = journal.limb_count(plan)
    assert len(rows[0]["state_in"]) == len(base["state_in"]) + limbs
    assert len(rows[0]["constraints"]) == len(base["constraints"]) + limbs*(w+1)
    for i, row in enumerate(rows):
        assert row["constraints"] == rows[0]["constraints"]
        assert row["state_in"] == rows[0]["state_in"]
        assert row["state_out"] == rows[0]["state_out"]
        assert r1cs_satisfied(row, PALLAS_BASE) and r1cs_satisfied(row, PALLAS_SCALAR)
        state = next_state(row)
        assert journal.unpack(plan, state[-limbs:], i+1) == expected[:i+1]
        if i:
            assert tuple(row["witness"][j] for j in row["state_in"]) == next_state(rows[i-1])
    assert journal.claimed_outputs(plan, next_state(rows[-1])) == expected


@pytest.mark.parametrize("wire", ["incoming", "outgoing", "counter"])
def test_constraint_corruption(wire):
    plan = plan_for(17, 2, "window")
    row = journal.schedule(plan)[9]
    target = {"incoming": row["state_in"][-1], "outgoing": row["state_out"][-1],
              "counter": row["state_out"][0]}[wire]
    row["witness"][target] += 1
    assert not r1cs_satisfied(row, PALLAS_SCALAR)


def test_journal_decoding_rejections():
    plan = plan_for(3, 2, "chain")
    end = next_state(journal.schedule(plan)[-1])
    for word in (-1, 1 << 128, 255, end[-1] + (1 << 120)):
        with pytest.raises(ValueError):
            journal.claimed_outputs(plan, (*end[:-1], word))
    with pytest.raises(ValueError):
        journal.claimed_outputs(plan, (*end[:-1], end[-1] ^ (1 << 32)))


def test_claim_rebuilt_without_replaying(monkeypatch):
    plan = plan_for(9, 2, "window")
    stmt, _ = journal.native_bundle(plan)
    end = tuple(map(int, stmt["zn"]))
    def forbidden(*args, **kwargs):
        raise AssertionError("full schedule called during verification")
    monkeypatch.setattr(journal, "schedule", forbidden)
    assert journal.authorized_statement(plan, end) == stmt
    # Changing an early digit can be syntactically valid, but cannot authorize a proof.
    first = end[-2] & 255
    other = 1 if first == 0 else 0
    altered = (*end[:-2], end[-2]-first+other, end[-1])
    assert journal.claimed_outputs(plan, altered) != journal.claimed_outputs(plan, end)
    assert journal.authorized_statement(plan, altered)["constraints"] == stmt["constraints"]
    assert journal.authorized_statement(plan, altered)["zn"] != stmt["zn"]


def test_manifest_and_fail_closed(tmp_path, monkeypatch):
    plan = plan_for(3, 2, "chain")
    end = next_state(journal.schedule(plan)[-1])
    root = "a"*64
    items = [{"job": "j0", "root": root, "split": "TRAIN", "payload": journal.claimed_outputs(plan, end)[0]}]
    roots = {"j0": root}
    manifest = journal.claim_manifest(plan, end, roots, {root}, items)
    assert manifest["count"] == 1 and "backend" not in manifest
    with pytest.raises(ValueError):
        journal.claim_manifest(plan, end, roots, set(), items)
    with pytest.raises(ValueError):
        journal.claim_manifest(plan, end, roots, {root}, items+items)
    with pytest.raises(ValueError):
        journal.claim_manifest(plan, end, roots, {root}, [{**items[0], "payload": (0, 0)}])
    exe = Path(tmp_path / "fixture-verifier")
    exe.write_bytes(b"fixture only")
    pin = hashlib.sha256(exe.read_bytes()).hexdigest()
    class Rejected:
        returncode = 1
        stdout = b"accepted verify_ms=5"
    class Accepted:
        returncode = 0
        stdout = b"accepted verify_ms=5\n"
    def mock_reject(*args, **kwargs):
        return Rejected()
    def mock_accept(*args, **kwargs):
        return Accepted()
    monkeypatch.setattr(journal.subprocess, "run", mock_reject)
    with pytest.raises(ValueError):
        journal.verify_and_certify(plan, end, b"mock", exe, pin, roots, {root}, items)
    monkeypatch.setattr(journal.subprocess, "run", mock_accept)
    with pytest.raises(ValueError):
        journal.verify_and_certify(plan, end, b"mock", exe, "b"*64, roots, {root}, items)
    # Process transport fixture; only the native smoke establishes a proof was checked.
    receipt = journal.verify_and_certify(plan, end, b"mock", exe, pin, roots, {root}, items)
    assert receipt["manifest"] == manifest
    assert receipt["proof_sha256"] == hashlib.sha256(b"mock").hexdigest()
