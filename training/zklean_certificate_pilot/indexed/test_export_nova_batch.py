from export_nova_batch import CASES, MEMORY_DEPTH, RULE_DEPTH, export


def test_public_transport():
    batch = export()
    assert batch['schema'] == 'zkl-affine-batch-v1'
    assert batch['rule_depth'] == RULE_DEPTH
    assert batch['memory_depth'] == MEMORY_DEPTH
    assert len(batch['cases']) == 2 * len(CASES)
    for exact, control in zip(batch['cases'][::2], batch['cases'][1::2], strict=True):
        assert exact['root'] == control['root']
        assert exact['environment_id'] == control['environment_id']
        assert exact['rules'] == control['rules']
        assert exact['goal'] == control['goal']
        assert exact['exact_rows'] == control['exact_rows'] == len(exact['steps'])
        assert len(control['steps']) >= len(exact['steps'])
