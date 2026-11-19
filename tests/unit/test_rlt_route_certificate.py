from itertools import product

from datagoblin.training_gym.rlt_revision import follow
from datagoblin.training_gym.rlt_route_certificate import observe_route, patched_endpoint


def test_prefix_witness_agrees_with_full_walk_across_three_node_maps() -> None:
    nodes = ("a", "b", "c")
    maps = [dict(zip(nodes, destinations, strict=True)) for destinations in product(nodes, repeat=3)]
    for before, after, start, steps in product(maps, maps, nodes, range(7)):
        witness = observe_route(before, start, steps)
        endpoint, cost = patched_endpoint(before, witness, after)
        assert endpoint == follow(after, start, steps)
        earliest = next((i for i, node in enumerate(witness.nodes[:-1]) if before[node] != after[node]), steps)
        assert cost == steps - earliest
