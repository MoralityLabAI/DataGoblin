"""Compare STOP-triggered and last-two-call verified exchanges on fresh roots."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from importlib.metadata import version
from itertools import product
from pathlib import Path

import torch

from MeTTa.ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION

from ..experiment import VERIFIERS_VERSION
from ..taskset import MAX_DECISIONS
from .__main__ import revision
from .action_ties import save_lines
from .capped_exchange import read_prior_exchange
from .model import LOOPS, MAX_PROPOSALS, upstream_revision, weight_hash
from .stop_exchange import StopExchange, read_stop_policy, source_cohorts
from .stop_policy import analyze, evaluate, load_archive, refit, save

PROTOCOL = {
    "schema": "metta-last-two-exchange-v1", "frozen_start": 200000, "conflict_start": 210000,
    "frozen_roots_per_family": 80, "conflict_roots": 160, "episode_call_caps": (3, 5),
    "primary": "last_two_5_minus_at_stop_5", "minimum_meaningful_completion_gain": 0.02,
    "decision_cap": MAX_DECISIONS, "proposal_cap": MAX_PROPOSALS, "recurrent_visits": LOOPS,
    "exchange_witness_cap": 2, "evidence_class": "fresh_frozen_budget_allocation_pilot",
    "capped_archive_sha256": "830f4db9e25130f92f5057b50e52faaa281f6d8509e9d4efda05541a6a9dab5f",
}
MODES = ("raw", "at_stop", "last_two", "agreed_removal")


def read_capped_archive(path: Path) -> dict:
    data = (path / "locked_protocol.json").read_bytes()
    if digest(data) != PROTOCOL["capped_archive_sha256"]:
        raise ValueError("Use the registered previous call-ceiling source lock")
    return json.loads(data)


async def run(prior: Path, stop_archive: Path, exchange_archive: Path, capped_archive: Path, out: Path) -> dict:
    memory = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty or version("verifiers") != VERIFIERS_VERSION or version("hyperon") != PACKAGE_VERSION:
        raise ValueError("Use the committed reservation protocol with pinned Prime Verifiers and Hyperon")
    upstream = upstream_revision()
    archived, fits, frames = load_archive(prior)
    previous_stop, _ = read_stop_policy(stop_archive)
    previous_exchange = read_prior_exchange(exchange_archive)
    previous_capped = read_capped_archive(capped_archive)
    cohorts = source_cohorts(archived, previous_stop, previous_exchange, previous_capped, protocol=PROTOCOL)
    here = Path(__file__).parent
    sources = ("stop_witness.metta", "stop_exchange.metta")
    rules = "\n".join((here / name).read_text(encoding="utf-8") for name in sources)
    witness = StopExchange(rules)
    arms = {f"{mode}_{cap}": (mode, cap) for mode, cap in product(MODES, PROTOCOL["episode_call_caps"])}
    modes = {arm: (0.0, None if mode == "raw" else witness) for arm, (mode, _) in arms.items()}
    call_caps = {arm: cap for arm, (_, cap) in arms.items()}
    reservations = {arm: mode for arm, (mode, _) in arms.items() if mode in ("last_two", "agreed_removal")}
    contrasts = tuple((f"{left}_{cap}", f"{right}_{cap}") for (left, right), cap in product(
        (("raw", "at_stop"), ("at_stop", "last_two"), ("at_stop", "agreed_removal"), ("last_two", "agreed_removal")),
        PROTOCOL["episode_call_caps"],
    ))
    out.mkdir(parents=True, exist_ok=False)
    protocol_sha = save(out / "locked_protocol.json", {
        "commit": commit, "upstream": upstream, "torch": str(torch.__version__), "verifiers": VERIFIERS_VERSION,
        "hyperon": PACKAGE_VERSION, "protocol": PROTOCOL, "native_rules_sha256": digest(rules.encode()),
        "action_set_lock_sha256": digest((prior / "locked_protocol.json").read_bytes()), "cohorts": cohorts,
        "stop_lock_sha256": digest((stop_archive / "locked_protocol.json").read_bytes()),
        "previous_exchange_lock_sha256": digest((exchange_archive / "locked_protocol.json").read_bytes()),
        "arm_call_caps": call_caps, "reservations": reservations, "contrasts": contrasts,
    })
    models, receipts = refit(frames, fits)
    fit_sha = save(out / "locked_fits.json", receipts)
    result = {}
    for split in ("frozen", "conflict"):
        rows, prime_sha = await evaluate(cohorts[split], split, models, receipts, modes, out, exchange=True,
                                         call_caps=call_caps, reservations=reservations)
        if any(row["policy"]["policy_verifier_calls"] > call_caps[row["arm"]] for row in rows):
            raise ValueError("A reservation policy exceeded its episode verifier-call cap")
        receipt_sha = save_lines(out / f"{split}_receipts.jsonl", rows)
        result[split] = {**analyze(rows, contrasts), "receipts_sha256": receipt_sha,
                         "prime_verifiers_sha256": prime_sha}
    if any(weight_hash(model) != receipts[seed]["checkpoint_sha256"] for seed, model in models.items()):
        raise ValueError("A frozen model changed during reservation scoring")
    report = {"protocol_sha256": protocol_sha, "fits_sha256": fit_sha, "result": result,
              "native_exchange_calls": witness.native.calls, "memory": memory, "peak_mib": peak_rss_mib(),
              "wall_seconds": time.perf_counter() - started, "training_ready": False}
    save(out / "reserve_exchange_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "stop-archive", "exchange-archive", "capped-archive", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(run(args.prior, args.stop_archive, args.exchange_archive, args.capped_archive, args.out))
    print(canonical(report))


if __name__ == "__main__":
    main()
