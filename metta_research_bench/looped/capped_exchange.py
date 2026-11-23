"""Compare the frozen model and exchange controller under the same episode call ceilings."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch

from MeTTa.ram_hillclimb import canonical, digest, install_memory_ceiling, peak_rss_mib
from MeTTa.ram_hillclimb_native import PACKAGE_VERSION

from ..experiment import VERIFIERS_VERSION
from ..taskset import MAX_DECISIONS
from .__main__ import revision
from .action_ties import save_lines
from .model import LOOPS, MAX_PROPOSALS, upstream_revision, weight_hash
from .stop_exchange import StopExchange, read_stop_policy, source_cohorts
from .stop_policy import analyze, evaluate, load_archive, refit, save

PROTOCOL = {
    "schema": "metta-budgeted-stop-exchange-v1", "frozen_start": 180000, "conflict_start": 190000,
    "frozen_roots_per_family": 80, "conflict_roots": 160, "episode_call_caps": (3, 5, 8, None),
    "primary": "exchange_5_minus_raw_5", "minimum_meaningful_completion_gain": 0.02,
    "decision_cap": MAX_DECISIONS, "proposal_cap": MAX_PROPOSALS, "recurrent_visits": LOOPS,
    "exchange_witness_cap": 2, "evidence_class": "fresh_frozen_compute_control",
    "exchange_archive_sha256": "644ea543cd786c523c3518ffd69c66dca37a8e7cd491f9b00e8311ef885a9708",
}


def arm_caps() -> dict:
    return {f"{arm}_{cap if cap is not None else 'unlimited'}": cap
            for cap in PROTOCOL["episode_call_caps"] for arm in ("raw", "exchange")}


def arm_contrasts() -> tuple:
    return tuple((f"raw_{name}", f"exchange_{name}")
                 for name in (str(cap) if cap is not None else "unlimited" for cap in PROTOCOL["episode_call_caps"]))


def read_prior_exchange(path: Path) -> dict:
    raw = (path / "locked_protocol.json").read_bytes()
    if digest(raw) != PROTOCOL["exchange_archive_sha256"]:
        raise ValueError("The earlier exchange study must match the registered source lock")
    return json.loads(raw)


async def run(prior: Path, stop_archive: Path, exchange_archive: Path, out: Path) -> dict:
    memory = install_memory_ceiling(8192)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    commit, dirty = revision()
    if dirty or version("verifiers") != VERIFIERS_VERSION or version("hyperon") != PACKAGE_VERSION:
        raise ValueError("Use the committed protocol with pinned Prime Verifiers and Hyperon")
    upstream = upstream_revision()
    archived, fits, frames = load_archive(prior)
    previous_stop, _ = read_stop_policy(stop_archive)
    previous_exchange = read_prior_exchange(exchange_archive)
    cohorts = source_cohorts(archived, previous_stop, previous_exchange, protocol=PROTOCOL)
    here = Path(__file__).parent
    sources = ("stop_witness.metta", "stop_exchange.metta")
    rules = "\n".join((here / name).read_text(encoding="utf-8") for name in sources)
    witness = StopExchange(rules)
    call_caps = arm_caps()
    contrasts = arm_contrasts()
    modes = {name: (0.0, witness if name.startswith("exchange_") else None) for name in call_caps}
    out.mkdir(parents=True, exist_ok=False)
    protocol_sha = save(out / "locked_protocol.json", {
        "commit": commit, "upstream": upstream, "torch": str(torch.__version__), "verifiers": VERIFIERS_VERSION,
        "hyperon": PACKAGE_VERSION, "protocol": PROTOCOL, "native_rules_sha256": digest(rules.encode()),
        "action_set_lock_sha256": digest((prior / "locked_protocol.json").read_bytes()),
        "stop_lock_sha256": digest((stop_archive / "locked_protocol.json").read_bytes()),
        "cohorts": cohorts, "arm_call_caps": call_caps, "contrasts": contrasts,
    })
    models, receipts = refit(frames, fits)
    fit_sha = save(out / "locked_fits.json", receipts)
    result = {}
    for split in ("frozen", "conflict"):
        rows, prime_sha = await evaluate(cohorts[split], split, models, receipts, modes, out,
                                         exchange=True, call_caps=call_caps)
        if any(call_caps[row["arm"]] is not None and
               row["policy"]["policy_verifier_calls"] > call_caps[row["arm"]] for row in rows):
            raise ValueError("A policy exceeded its episode verifier-call cap")
        row_sha = save_lines(out / f"{split}_receipts.jsonl", rows)
        result[split] = {**analyze(rows, contrasts), "receipts_sha256": row_sha, "prime_verifiers_sha256": prime_sha}
    if any(weight_hash(model) != receipts[seed]["checkpoint_sha256"] for seed, model in models.items()):
        raise ValueError("A frozen model changed during capped exchange scoring")
    report = {"protocol_sha256": protocol_sha, "fits_sha256": fit_sha, "result": result,
              "native_exchange_calls": witness.native.calls, "memory": memory, "peak_mib": peak_rss_mib(),
              "wall_seconds": time.perf_counter() - started, "training_ready": False}
    save(out / "capped_exchange_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "stop-archive", "exchange-archive", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(canonical(asyncio.run(run(args.prior, args.stop_archive, args.exchange_archive, args.out))))


if __name__ == "__main__":
    main()
