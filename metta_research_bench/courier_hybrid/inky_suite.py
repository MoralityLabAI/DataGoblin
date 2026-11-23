"""Export independently checked public Inky tasks and score provenance-complete replies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

from .auxiliary import compare_repair, native_candidates, public_brief, score_repair
from .experiment import file_sha, json_out, jsonl_out
from .inky import TASKS, InkyAdapter, Reply, looped_routes, request_for, score_answer
from .native import ATTEST_FIRST, ATTEST_SECOND, DELIVER, NativeCourier, load_storyworld
from .oracle import continuations, objective_weights, repair_routes, state_graph
from .split import HERE, load_manifest, plan, worlds

CEILING, DEPTH, LIMIT = 5, 4, 2


def _prefixes(world, graph):
    routes = {world.initial: ()}
    queue = deque([world.initial])
    while queue:
        state = queue.popleft()
        for action, after in sorted(graph[state].items()):
            if after not in routes:
                routes[after] = (*routes[state], action)
                queue.append(after)
    return routes


def _rejection(native, world, prefix, action=None):
    session = native.session(world, CEILING, 12)
    if prefix and not session.certify(prefix, macro=True).accepted:
        raise RuntimeError("Native rejection setup prefix failed")
    receipt = session.stop() if action is None else session.certify((action,))
    if receipt.accepted:
        raise RuntimeError("Native rejection task unexpectedly succeeded")
    return receipt


def cases(native, upstream, model_id, roots_per_family=3):
    protocol = plan("development")
    manifest = load_manifest(HERE / "development_split.json", protocol)
    counts = Counter()
    for root, world in worlds(upstream, protocol, manifest, "score"):
        if counts[root.family] == roots_per_family:
            continue
        counts[root.family] += 1
        graph, distances = state_graph(world, native.checked_successors)
        prefixes = _prefixes(world, graph)
        exact = repair_routes(world, world.initial, 12, 1)[0]
        prefix = exact[:-DEPTH]
        repair_obs, _ = public_brief(native, world, prefix, len(prefix) + DEPTH, CEILING)
        for skill in ("metta-repair-actions", "metta-storyworld-routes"):
            yield root, world, prefix, request_for(repair_obs, skill, model_id, depth=DEPTH, limit=LIMIT), None

        accepted_actions = ()
        for state, path in prefixes.items():
            costs = continuations(state, graph, distances, 12 - len(path), 13 - len(path))
            if costs:
                accepted_actions = tuple(objective_weights(world, state, costs)["shortest"])
            if len(accepted_actions) > 1:
                obs, _ = public_brief(native, world, path, 12, 13)
                yield root, world, path, request_for(obs, "metta-compare-policies", model_id), accepted_actions
                break

        first_attest = next(i for i, action in enumerate(exact) if action in (ATTEST_FIRST, ATTEST_SECOND))
        path = exact[: first_attest + 1]
        obs, _ = public_brief(native, world, path, 12, CEILING)
        receipt = _rejection(native, world, path, exact[first_attest])
        yield (
            root,
            world,
            path,
            request_for(obs, "metta-curate-sources", model_id, rejection=receipt),
            "already_attested",
        )

        for state, path in prefixes.items():
            if state.place == world.desk and state.cargo == 1 and state.evidence != 3 and len(path) < 12:
                obs, _ = public_brief(native, world, path, 12, CEILING)
                receipt = _rejection(native, world, path, DELIVER)
                yield (
                    root,
                    world,
                    path,
                    request_for(obs, "metta-evidence-dockets", model_id, rejection=receipt),
                    ("missing_independent_evidence"),
                )
                break

        receipt = _rejection(native, world, prefix)
        yield (
            root,
            world,
            prefix,
            request_for(repair_obs, "metta-verify-rules", model_id, rejection=receipt),
            ("undelivered_stop"),
        )


def _load_checkpoint(run: Path):
    import torch

    from ..looped.model import weight_hash
    from .learning import CourierPointer

    lock = json.loads((run / "fit_lock.json").read_text(encoding="utf-8"))
    record = next(r for r in lock["fits"] if r["name"] == "shortest" and r["seed"] == 1101)
    file = run / "shortest-1101.pt"
    if file_sha(file) != record["file_sha256"]:
        raise ValueError("Auxiliary comparison checkpoint differs from its fit receipt")
    model = CourierPointer().eval()
    model.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
    if weight_hash(model) != record["checkpoint_weights_sha256"]:
        raise ValueError("Auxiliary comparison weights differ from their fit receipt")
    return model, record["checkpoint_weights_sha256"]


def baselines(upstream, fit_run: Path, out: Path) -> None:
    """Score the two available comparators without generating any external-model reply."""
    pointer, weights = _load_checkpoint(fit_run)
    protocol = plan("development")
    manifest = load_manifest(HERE / "development_split.json", protocol)
    native = NativeCourier(upstream, auxiliary_search=True)
    rows = []
    for root, world in worlds(upstream, protocol, manifest, "score"):
        prefix = repair_routes(world, world.initial, 12, 1)[0][:-DEPTH]
        visits = 0

        class MeteredPointer:
            def propose(self, obs):
                nonlocal visits
                proposal = pointer.propose(obs)
                visits += proposal.visits
                return proposal

        native_result = score_repair(
            native, world, prefix, external_search=True, proposer=lambda obs, d, n: native_candidates(native, obs, d, n)
        )
        model_result = score_repair(
            native, world, prefix, proposer=lambda obs, d, n: looped_routes(MeteredPointer(), obs, d, n)
        )
        rows.append(
            {
                "root": root.world_sha256,
                "family": root.family,
                "native_search": native_result,
                "looped": model_result,
                "looped_recurrent_visits": visits,
            }
        )
    out.mkdir(parents=True, exist_ok=False)
    rows_sha = jsonl_out(out / "baseline_receipts.jsonl", rows)
    json_out(
        out / "baseline_summary.json",
        {
            "roots": len(rows),
            "receipts_sha256": rows_sha,
            "pointer_weights_sha256": weights,
            "looped_recurrent_visits": sum(row["looped_recurrent_visits"] for row in rows),
            "outcomes": {
                method: {
                    "completed": sum(row[method]["verified_completion"] for row in rows),
                    "calls": sum(row[method]["calls"] for row in rows),
                }
                for method in ("native_search", "looped")
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("export", "evaluate", "baselines"))
    parser.add_argument("--model")
    parser.add_argument("--roots-per-family", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--storyworld", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit-run", type=Path)
    parser.add_argument("--replies", type=Path)
    args = parser.parse_args()
    upstream, _ = load_storyworld(args.storyworld)
    if args.phase in {"baselines", "evaluate"}:
        import torch

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        if args.fit_run is None:
            parser.error(f"{args.phase} requires --fit-run")
    if args.phase == "baselines":
        baselines(upstream, args.fit_run, args.out)
        return
    if not args.model:
        parser.error(f"{args.phase} requires --model")
    fixture_native = NativeCourier(upstream)
    fixtures = list(cases(fixture_native, upstream, args.model, args.roots_per_family))
    if args.phase == "export":
        jsonl_out(args.out, ({"root": root.world_sha256, **request} for root, _, _, request, _ in fixtures))
        return
    if args.replies is None:
        parser.error("evaluate requires --replies")
    replies = [Reply(**json.loads(line)) for line in args.replies.read_text(encoding="utf-8").splitlines()]
    by_request = {r.request_sha256: r for r in replies}
    if len(replies) != len(by_request) or by_request.keys() != {
        request["request_sha256"] for _, _, _, request, _ in fixtures
    }:
        raise ValueError("Provide exactly one reply per exported request")
    pointer, weights = _load_checkpoint(args.fit_run)
    native = NativeCourier(upstream, auxiliary_search=True)
    output = []
    for root, world, prefix, request, labels in fixtures:
        skill = request["skill"]
        reply = by_request[request["request_sha256"]]
        base = {
            "root": root.world_sha256,
            "family": root.family,
            "skill": skill,
            "request_sha256": reply.request_sha256,
            "model": reply.model,
            "inference_seconds": reply.inference_seconds,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
        }
        if TASKS[skill] == "propose_routes":
            adapter = InkyAdapter(args.model, lambda req: by_request[req["request_sha256"]], skill)
            result = compare_repair(native, world, prefix, adapter, pointer, ceiling=CEILING, depth=DEPTH, limit=LIMIT)
            output.append({**base, **result})
        else:
            named = {"optimal_actions": labels} if TASKS[skill] == "choose_action" else {"rejection_reason": labels}
            output.append({**base, "correct": score_answer(request, reply, **named)})
    args.out.mkdir(parents=True, exist_ok=False)
    rows_sha = jsonl_out(args.out / "inky_receipts.jsonl", output)
    advice = Counter(row["skill"] for row in output if row.get("correct"))
    repair = [row for row in output if "inky" in row]
    json_out(
        args.out / "inky_summary.json",
        {
            "model": args.model,
            "pointer_weights_sha256": weights,
            "fixture_calls": fixture_native.calls,
            "reply_file_sha256": file_sha(args.replies),
            "result_sha256": rows_sha,
            "skills_attempted": dict(Counter(row["skill"] for row in output)),
            "advice_correct": dict(advice),
            "repair_cases": len(repair),
            "repair": {
                name: {
                    "verified": sum(row[name]["verified_completion"] for row in repair),
                    "native_calls": sum(row[name]["calls"] for row in repair),
                }
                for name in ("inky", "native_search", "looped")
            },
            "looped_recurrent_visits": sum(row["looped"]["auxiliary_recurrent_visits"] for row in repair),
            "inky_input_tokens": sum(row["input_tokens"] for row in output),
            "inky_output_tokens": sum(row["output_tokens"] for row in output),
            "inky_inference_seconds": sum(row["inference_seconds"] for row in output),
        },
    )


if __name__ == "__main__":
    main()
