"""RV22-T: binary canonical grammar, equivalence and checked-cache safety."""
import sys
from pathlib import Path
from random import Random
from unittest.mock import patch

import pytest


def module():
    here = Path(__file__).resolve().parent
    scripts = here if (here / 'binary_keys.py').exists() else here.parents[1] / 'scripts' / 'research_v2_2'
    with patch.object(sys, 'path', [str(scripts), *sys.path]):
        import binary_keys as m
    return m


def test_binary_roundtrip_and_invariance():
    m = module()
    from fingerprint import hidden_variant
    from refinement import witness

    rng = Random(4261)
    worlds = [m.world(rng, 12 if j%2 == 0 else 16) for j in range(24)]
    worlds.extend(d for n in range(1, 6) for d in witness(n))
    worlds.extend(m.generate(4262, 'dense4', 4, 4))
    for d in worlds:
        key, scale = m.binary_key(d)
        decoded = m.decode(key)
        assert m.encode_record(decoded) == key
        assert m.enc(decoded) == m.packed_key(d)[0]
        assert m.canonical(m.to_world(decoded, scale)) == m.canonical(d)
        transformed = hidden_variant(d, rng)
        assert m.binary_key(transformed)[0] == key
        assert m.canonical(transformed)[0] == m.canonical(d)[0]
        for x in (d, transformed):
            solver = m.DP(x)
            value, tree = solver.run()
            assert m.replay(x, tree) == value
            assert m.check(solver) == solver.edges


def test_decoder_rejects_noncanonical_and_malformed_keys():
    m = module()
    rec = (2, 12, -16, (('x', 1),), ((0, 1), (3, 2)))
    good = m.encode_record(rec)
    assert m.decode(good) == rec
    malformed = [good[:i] for i in range(len(good))]
    malformed.extend((good+b'\0', b'BAD!'+good[4:], b'x'*(m.MAX_BYTES+1)))
    for offset, byte in ((16, 1), (18, 255), (20, 0), (21, 0), (22, 255)):
        bad = bytearray(good)
        bad[offset] = byte
        malformed.append(bytes(bad))
    for candidate in (
        (2, 12, -16, (('A', 1),), ((0, 1), (3, 2))),
        (2, 12, -16, (('x', 0),), ((0, 1), (3, 2))),
        (2, 12, -16, (('x', 3),), ((0, 1), (3, 2))),
        (2, 12, -16, (('x', 1),), ((0, 2), (3, 4))),
        (2, 12, -16, (('x', 1),), ((0, 1), (1, 2))),
        (2, 12, -16, (('x', 1),), ((0, 1), (4, 2))),
        (2, 12, -16, (('x', 1),), ((0, 1), (3, 0))),
        (2, 12, -16, (('x', 1),), ((3, 2), (0, 1))),
        (2, 12, -16, (('x', 1),), ((0, 1), (0, 2))),
        (2, 12, -16, (('z', 1), ('a', 1)), ((0, 1), (7, 2))),
        (2, 12, -16, (('x', 1), ('x', 2)), ((0, 1), (7, 2))),
    ):
        malformed.append(m.encode_record(candidate))
    # Nonminimal but otherwise well-shaped two-byte mass fields.
    wide = bytearray(good[:27])
    wide[20] = 2
    wide.extend(b'\x00\x00\x01\x03\x00\x02')
    malformed.append(bytes(wide))
    for bad in malformed:
        with pytest.raises(ValueError):
            m.decode(bad)
    for bad in (bytearray(good), memoryview(good), None):
        with pytest.raises(ValueError):
            m.decode(bad)


def test_checked_capacity_and_forced_digest_collision():
    m = module()
    ds = m.generate(4263, 'sparse8', 4, 24)
    ref = m.evaluate(ds, 'cold', 2048)[1]
    for a in m.ARMS:
        for cap in (m.RESERVE, 2048, 8192):
            stats, vals = m.evaluate(ds, a, cap)
            assert vals == ref and stats['hits']+stats['solves'] == len(ds)
            assert stats['checks'] == stats['admissions']
            assert stats['peak_accounted'] <= cap
            if cap == m.RESERVE:
                assert stats['hits'] == stats['admissions'] == 0
    with patch.object(m, 'digest', return_value=b'collision'):
        for a in m.ARMS[1:]:
            stats, vals = m.evaluate(ds, a, 8192)
            assert vals == ref and stats['hits'] > 0
            assert stats['checks'] == stats['admissions']
