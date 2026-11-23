"""Matched three-way evaluation of bounded auxiliary courier proposals."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

from .controller import Observation, Proposal, Repair, run_episode
from .inky import InkyAdapter
from .native import STOP, NativeCourier, World, action_atom, state_atom, successors

NATIVE_SEARCH_CALLS = 1


def native_candidates(native: NativeCourier, obs: Observation, depth: int, limit: int) -> tuple[tuple[int, ...], ...]:
    """Use one actual native query for search, separate from certification."""
    fuel = " ".join(["Step"] * depth)
    answers = native.query(obs.world, f"(HybridFind {state_atom(obs.state)} ({fuel}))")
    vocabulary = {action_atom(action): action for action in range(STOP)}
    plans = []
    for answer in answers:
        if not answer.startswith("(HybridRoute (") or not answer.endswith("))"):
            raise RuntimeError("Native auxiliary search returned an unexpected answer")
        sequence = answer.removeprefix("(HybridRoute (").removesuffix("))")
        # Hyperon prints each public move as a two-atom parenthesized expression.
        tokens = sequence.replace("(Move ", "Move_").replace(")", "").split()
        route = tuple(vocabulary[t.replace("Move_", "(Move ") + ")" if t.startswith("Move_") else t] for t in tokens)
        if 0 < len(route) <= depth:
            plans.append(route)
    plans.sort(key=lambda route: (len(route), route))
    return tuple(plans[:limit])


def score_repair(
    native: NativeCourier,
    world: World,
    prefix: tuple[int, ...],
    *,
    proposer: Callable,
    external_search: bool = False,
    ceiling: int = 5,
    depth: int = 4,
    limit: int = 2,
) -> dict:
    """Evaluate one supplied public repair opportunity under the shared call ceiling.

    Replaying the prefix, querying native search, certifying candidates, and
    checking STOP all count. The external model can only suggest typed routes.
    """
    before = native.calls
    reserve = NATIVE_SEARCH_CALLS if external_search else 0
    session = native.session(world, ceiling - reserve, len(prefix) + depth)
    if prefix and not session.certify(prefix, macro=True).accepted:
        raise ValueError("Auxiliary fixture prefix has no native proof")
    first_receipts = len(session.receipts)
    if session.state.complete:
        raise ValueError("Auxiliary repair fixtures must start incomplete")

    def wrapped(obs, route_depth, route_limit):
        public = Observation(
            obs.world,
            obs.state,
            (*sorted(successors(obs.world, obs.state)), STOP),
            obs.moves_left,
            obs.remaining_calls + reserve,
        )
        return proposer(public, route_depth, route_limit)

    result = run_episode(session, lambda _obs: Proposal(STOP), Repair("reserve", depth, limit, proposer=wrapped))
    spent = native.calls - before
    search_calls = spent - session.calls
    if search_calls != reserve or spent > ceiling:
        raise RuntimeError("Auxiliary proposer exceeded its separate native search allowance")
    result["calls"] = spent
    result["unused_calls"] = ceiling - spent
    result["native_search_calls"] = search_calls
    result["prefix_receipts"] = first_receipts
    return result


def compare_repair(
    native,
    world: World,
    prefix: tuple[int, ...],
    inky: InkyAdapter,
    model,
    *,
    ceiling: int = 5,
    depth: int = 4,
    limit: int = 2,
) -> dict:
    from .inky import looped_routes

    model_visits = 0

    class MeteredPointer:
        def propose(self, obs):
            nonlocal model_visits
            proposal = model.propose(obs)
            model_visits += proposal.visits
            return proposal

    def model_proposer(obs, horizon, allowance):
        routes = looped_routes(MeteredPointer(), obs, horizon, allowance)
        return routes

    config = {"ceiling": ceiling, "depth": depth, "limit": limit}
    index = len(inky.inferences)
    return {
        "inky": score_repair(native, world, prefix, proposer=inky, **config),
        "inky_inference": inky.inferences[index:],
        "native_search": score_repair(
            native,
            world,
            prefix,
            external_search=True,
            proposer=lambda obs, d, n: native_candidates(native, obs, d, n),
            **config,
        ),
        "looped": {
            **score_repair(native, world, prefix, proposer=model_proposer, **config),
            "auxiliary_recurrent_visits": model_visits,
        },
    }


def public_brief(native: NativeCourier, world: World, prefix: tuple[int, ...], horizon: int, ceiling: int):
    """Build an independently certified task fixture to render for a local model."""
    session = native.session(world, ceiling, horizon)
    receipt = session.certify(prefix, macro=True) if prefix else None
    if receipt is not None and not receipt.accepted:
        raise ValueError("Cannot brief Inky with an uncertified prefix")
    choices = native.checked_successors(world, session.state)
    return Observation(world, session.state, (*sorted(choices), STOP), session.moves_left, session.remaining), (
        asdict(receipt) if receipt else None
    )
