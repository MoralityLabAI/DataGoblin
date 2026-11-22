"""MOSES-inspired finite-grammar program evolution over MeTTa curation motifs.

This is an original CPU control, not the upstream PeTTa/MeTTa-MOSES
implementation. It learns a small advisory "initial selection invalid?"
predictor from synthetic oracle labels on a partition of TRAIN roots only.

Programs are a constant, one (possibly negated) observed feature, or the
AND/OR of two literals. Their size is bounded before evaluation. Program
fitness uses the discovery partition; confirmation is read only for the
selected program and frozen controls, never for search or early stopping.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .collect import ROOT, sha256_bytes
from .looped_sft import root_id, split_for
from .pattern_miner import Observation, _internal_partition, generate_observations

SCHEMA = "metta-moses-bounded-control-v1"
DEFAULT_OUT = ROOT / "runs" / "metta" / "moses_rule_control"
OPS = ("false", "true", "lit", "and", "or")


@dataclass(frozen=True)
class Program:
    op: str
    literals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError("unknown program op")
        expected = 0 if self.op in ("false", "true") else 1 if self.op == "lit" else 2
        if len(self.literals) != expected:
            raise ValueError("program arity outside bound")
        if any(not s or s == "!" or s.startswith("!!") for s in self.literals):
            raise ValueError("invalid literal")
        if self.op in ("and", "or") and tuple(sorted(self.literals)) != self.literals:
            raise ValueError("compound literals must be canonical")

    @property
    def nodes(self) -> int:
        return 1 + len(self.literals)

    @property
    def key(self) -> str:
        return "/".join((self.op, *self.literals))

    def predict(self, observed: tuple[str, ...]) -> bool:
        if self.op == "false":
            return False
        if self.op == "true":
            return True
        features = set(observed)
        values = [
            (token[1:] not in features) if token.startswith("!") else (token in features)
            for token in self.literals
        ]
        if self.op == "lit":
            return values[0]
        return all(values) if self.op == "and" else any(values)


def make_program(op: str, literals: tuple[str, ...] = ()) -> Program:
    return Program(op, tuple(sorted(literals)) if op in ("and", "or") else literals)


def confusion(program: Program, rows: list[Observation]) -> dict[str, float | int]:
    tp = tn = fp = fn = 0
    for row in rows:
        prediction = program.predict(row.features)
        if prediction and row.invalid:
            tp += 1
        elif prediction:
            fp += 1
        elif row.invalid:
            fn += 1
        else:
            tn += 1
    total = tp + tn + fp + fn
    if total == 0:
        raise ValueError("cannot evaluate on empty root partition")
    positive = tp + fn
    negative = tn + fp
    sensitivity = tp / positive if positive else 1.0
    specificity = tn / negative if negative else 1.0
    balanced = (sensitivity + specificity) / 2.0
    return {
        "roots": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round((tp + tn) / total, 6),
        "balanced_accuracy": round(balanced, 6),
        "false_positive_rate": round(fp / negative, 6) if negative else 0.0,
        "false_negative_rate": round(fn / positive, 6) if positive else 0.0,
    }


def fitness(p: Program, statistics: dict[str, float | int]) -> float:
    return float(statistics["balanced_accuracy"]) - 0.004 * p.nodes


def _feature_vocabulary(rows: list[Observation]) -> list[str]:
    vocabulary = sorted({f for row in rows for f in row.features})
    if not vocabulary or any(not f.replace("_", "").isalnum() for f in vocabulary):
        raise ValueError("bad structural feature vocabulary")
    return vocabulary


def _lineage_digest(rows: list[Observation]) -> str:
    return sha256_bytes("\n".join(sorted(x.root for x in rows)).encode("utf-8"))


def _operators(rng: random.Random, vocabulary: list[str]) -> tuple[str, str]:
    left = rng.choice(vocabulary)
    right = rng.choice(vocabulary)
    return ("!" + left if rng.randrange(2) else left,
            "!" + right if rng.randrange(2) else right)


def mutate(p: Program, rng: random.Random, vocabulary: list[str]) -> Program:
    left, right = _operators(rng, vocabulary)
    choice = rng.randrange(6)
    if choice == 0:
        return make_program("lit", (left,))
    if choice == 1:
        return make_program(rng.choice(("and", "or")), (left, right))
    if choice == 2:
        return make_program(rng.choice(("true", "false")))
    if choice == 3 and p.literals:
        member = rng.randrange(len(p.literals))
        changed = list(p.literals)
        changed[member] = changed[member][1:] if changed[member].startswith("!") else "!" + changed[member]
        return make_program(p.op, tuple(changed))
    if choice == 4 and len(p.literals) == 2:
        return make_program("or" if p.op == "and" else "and", p.literals)
    if p.literals:
        return make_program("lit", (rng.choice(p.literals),))
    return make_program("and", (left, right))


def crossover(a: Program, b: Program, rng: random.Random, vocabulary: list[str]) -> Program:
    left = rng.choice(a.literals) if a.literals else _operators(rng, vocabulary)[0]
    right = rng.choice(b.literals) if b.literals else _operators(rng, vocabulary)[1]
    return make_program(rng.choice(("and", "or")), (left, right))


def _validate_rows(rows: list[Observation]) -> tuple[list[Observation], list[Observation]]:
    roots = [row.root for row in rows]
    if len(rows) < 12 or len(set(roots)) != len(rows):
        raise ValueError("insufficient unique source roots")
    if any(row.split != "train_candidate" or row.family == "conflict" for row in rows):
        raise ValueError("reserved source lineage or heldout label encountered")
    for row in rows:
        parts = row.root.split(":")
        if len(parts) != 3 or parts[0] != "metta-sft-v1" or not parts[2].isdigit():
            raise ValueError("invalid source-root identity")
        seed = int(parts[2])
        if (parts[1] != row.family or root_id(row.family, seed) != row.root
                or split_for(row.family, seed) != "train_candidate"):
            raise ValueError("reserved or mismatched source-root partition")
    if any(not row.features or tuple(sorted(set(row.features))) != row.features for row in rows):
        raise ValueError("malformed observable features")
    discovery = [x for x in rows if _internal_partition(x.root) == "discovery"]
    confirmation = [x for x in rows if _internal_partition(x.root) == "confirmation"]
    if len(discovery) < 8 or len(confirmation) < 3:
        raise ValueError("insufficient disjoint training-root partitions")
    if set(x.root for x in discovery) & set(x.root for x in confirmation):
        raise ValueError("cross-partition lineage overlap")
    return discovery, confirmation


def evolve(rows: list[Observation], *, seed: int = 20261122,
           population_size: int = 40, generations: int = 12) -> dict[str, Any]:
    if not 8 <= population_size <= 128 or not 1 <= generations <= 64:
        raise ValueError("bounded CPU search budgets required")
    discovery, confirmation = _validate_rows(rows)
    rng = random.Random(seed)
    vocabulary = _feature_vocabulary(discovery)
    literal_options = [*(vocabulary), *("!" + f for f in vocabulary)]
    initial = [Program("false"), Program("true")]
    initial += [Program("lit", (lit,)) for lit in literal_options]
    # Seed a bounded, deterministic mix of simple two-atom programs.
    initial += [make_program("and", (left, right))
                for left, right in zip(literal_options[::2], literal_options[1::2])]
    population = {p.key: p for p in initial}
    cache: dict[str, dict[str, float | int]] = {}
    full_archive: dict[str, Program] = dict(population)

    def rank(p: Program) -> tuple[float, float, int, str]:
        metrics = cache.get(p.key)
        if metrics is None:
            metrics = confusion(p, discovery)
            cache[p.key] = metrics
        return (-fitness(p, metrics), -float(metrics["balanced_accuracy"]), p.nodes, p.key)

    history = []
    for generation in range(generations):
        leaders = sorted(population.values(), key=rank)[:population_size]
        best = leaders[0]
        best_stats = cache[best.key]
        history.append({
            "generation": generation,
            "best_program": best.key,
            "best_discovery_balanced_accuracy": best_stats["balanced_accuracy"],
            "best_discovery_fitness": round(fitness(best, best_stats), 6),
            "unique_programs_evaluated_so_far": len(cache),
        })
        keep = leaders[: max(3, population_size // 4)]
        next_pool = {p.key: p for p in keep}
        attempts = 0
        while len(next_pool) < population_size and attempts < population_size * 25:
            attempts += 1
            if rng.randrange(3) == 0:
                candidate = crossover(rng.choice(keep), rng.choice(keep), rng, vocabulary)
            else:
                candidate = mutate(rng.choice(keep), rng, vocabulary)
            next_pool[candidate.key] = candidate
            full_archive[candidate.key] = candidate
        population = next_pool
    winner = min(full_archive.values(), key=rank)
    discovery_metrics = cache[winner.key]
    confirmation_metrics = confusion(winner, confirmation)
    majority = Program("true" if sum(x.invalid for x in discovery) >= len(discovery) / 2 else "false")
    majority_metrics = confusion(majority, confirmation)
    fixed_cost = Program("lit", ("selected_cost_ge_75pct",))
    fixed_cost_metrics = confusion(fixed_cost, confirmation)
    return {
        "schema_version": SCHEMA,
        "algorithm": "original_bounded_genetic_rule_search",
        "upstream_metta_moses_executed": False,
        "model_trained": False,
        "synthetic_task_only": True,
        "source_split": "train_candidate",
        "reserved_root_labels_accessed": 0,
        "new_training_presentations": 0,
        "training_ready": False,
        "seed": seed,
        "population_cap": population_size,
        "generations": generations,
        "search_grammar": "constants, signed literals, and/or of two signed literals",
        "max_program_nodes": 3,
        "candidate_programs_evaluated": len(cache),
        "discovery_roots": len(discovery),
        "confirmation_roots": len(confirmation),
        "discovery_lineage_sha256": _lineage_digest(discovery),
        "confirmation_lineage_sha256": _lineage_digest(confirmation),
        "feature_vocabulary": vocabulary,
        "best_program": {"op": winner.op, "literals": list(winner.literals),
                         "key": winner.key, "nodes": winner.nodes},
        "discovery": discovery_metrics,
        "internal_confirmation": confirmation_metrics,
        "majority_baseline": {"program": majority.key, "metrics": majority_metrics},
        "cost_pressure_baseline": {"program": fixed_cost.key, "metrics": fixed_cost_metrics},
        "confirmation_minus_majority_bacc": round(
            float(confirmation_metrics["balanced_accuracy"]) -
            float(majority_metrics["balanced_accuracy"]), 6),
        "history": history,
        "limitations": [
            "Internal confirmation is a partition of known synthetic TRAIN roots.",
            "Finite grammar with at most two boolean literals, not native MOSES.",
            "Predicting a failed initial selection does not authorize data admission.",
            "Results do not establish generalization to original MeTTa source files.",
        ],
    }


def advisory_metta(report: dict[str, Any]) -> str:
    if report.get("schema_version") != SCHEMA or report.get("training_ready") is not False:
        raise ValueError("advisory program requires safe frozen report")
    p = report["best_program"]
    literals = p["literals"]

    def lit(token: str) -> str:
        feature = token[1:] if token.startswith("!") else token
        expr = f"(HasFeature $world {feature})"
        return f"(not {expr})" if token.startswith("!") else expr

    if p["op"] in ("true", "false"):
        body = "True" if p["op"] == "true" else "False"
    elif p["op"] == "lit":
        body = lit(literals[0])
    else:
        body = f"({p['op']} {lit(literals[0])} {lit(literals[1])})"
    return ("; Syntactic MeTTa teaching form; unverified by native interpreter.\n"
            "; Advisory rule; independent Python oracle controls source admission.\n"
            f"(= (ProposeInvalid $world) {body})\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots-per-family", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to((ROOT / "MeTTa").resolve()):
        parser.error("generated program reports belong outside source tree")
    report = evolve(generate_observations(roots_per_family=args.roots_per_family), seed=args.seed,
                    population_size=args.population, generations=args.generations)
    report["source_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rule_search.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    (args.out / "advisory_rule.metta").write_text(advisory_metta(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "discovery_roots", "confirmation_roots", "candidate_programs_evaluated",
        "best_program", "confirmation_minus_majority_bacc", "training_ready"
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
