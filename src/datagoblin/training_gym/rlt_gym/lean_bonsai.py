"""Collect checked Lean tactic selections with local OxiBonsai inference."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from contextlib import closing
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from typing import Any

from datagoblin.hashing import sha256_file, sha256_json, sha256_text
from datagoblin.serialization import canonical_json, dump_structured, load_structured

from .. import rlt_lean_grounded as grounded
from ..rlt_lean_labels import held_templates
from ..rlt_lean_ldt import AXIOMS
from ..rlt_lean_rank import TACTICS

CHECKPOINT = "prism-ml/Bonsai-8B-gguf"
CHECKPOINT_SHA256 = "284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54"
SPLITS = {"train": 30000, "same_templates": 33000, "held_templates": 36000}
MAX_TOKENS = 32
AXIOM_AUDIT = re.compile(r"^'dgBonsaiProof' (?:does not depend on any axioms|depends on axioms: \[(.*)\])$", re.M)
LEAN_LIMIT = re.compile(r"heartbeat|timed? out|out of memory|maximum recursion depth", re.I)
CLI_USAGE = re.compile(r"^(\d+) prompt \+ (\d+) generated = \d+ total tokens\b", re.M)


def build_prompt(statement: str, menu: dict[int, str], failed: dict[int, str]) -> str:
    options = "\n".join(f"{i}. {tactic}" for i, tactic in menu.items())
    prompt = f"Lean 4 theorem:\n{statement}\n\nAfter `repeat' intro`, choose one tactic to close every goal."
    if failed:
        failures = "; ".join(f"{i} ({tactic})" for i, tactic in failed.items())
        prompt += f"\nPrevious choices failed verification: {failures}."
    return f"{prompt}\n{options}\n\nReply with the tactic number only. /no_think"


def parse_choice(content: Any, menu: dict[int, str]) -> int | None:
    text = content.strip() if isinstance(content, str) else ""
    text = re.sub(r"^<think>\s*</think>\s*", "", text)
    number = int(text) if re.fullmatch(r"[1-9][0-9]?", text) else None
    return number if number in menu else None


class DirectOxiBonsai:
    def __init__(self, binary: Path, checkpoint: Path, tokenizer: Path, timeout: float) -> None:
        self.binary = binary.resolve(strict=True)
        self.tokenizer = tokenizer.resolve(strict=True)
        self.timeout = timeout
        self.arguments = [
            str(self.binary), "run", "--model", str(checkpoint.resolve(strict=True)),
            "--tokenizer", str(self.tokenizer), "--prompt", "-", "--temperature", "0",
            "--max-tokens", str(MAX_TOKENS), "--max-seq-len", "2048",
        ]
        try:
            version = subprocess.run(
                [str(self.binary), "--version"], capture_output=True, encoding="utf-8", timeout=10, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError("OxiBonsai version check timed out") from exc
        self.version = version.stdout.strip()
        if version.returncode or not self.version.startswith(("oxibonsai ", "oxibonsai-cli ")):
            raise ValueError("--oxibonsai-bin must point to the OxiBonsai CLI executable")

    def choose(self, statement: str, menu: dict[int, str], failed: dict[int, str]) -> dict[str, Any]:
        prompt = build_prompt(statement, menu, failed)
        # `run` tokenizes the raw prompt. Its stdin reader drops one trailing newline.
        framed = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\n"
        started = time.perf_counter()
        try:
            result = subprocess.run(
                self.arguments, input=framed, capture_output=True, encoding="utf-8", errors="replace",
                timeout=self.timeout, check=False, env={**os.environ, "RUST_LOG": "off"},
            )
        except subprocess.TimeoutExpired as exc:
            raise OSError(f"OxiBonsai inference timed out after {self.timeout:g}s") from exc
        if result.returncode:
            detail = result.stderr.strip().splitlines()[-1:] or ["no error output"]
            raise OSError(f"OxiBonsai exited with status {result.returncode}: {detail[0][:300]}")
        usage_match = CLI_USAGE.search(result.stderr)
        usage = None
        if usage_match:
            usage = {"prompt_tokens": int(usage_match[1]), "completion_tokens": int(usage_match[2])}
        return {
            "choice": parse_choice(result.stdout, menu), "request_sha256": sha256_text(framed[:-1]),
            "response_sha256": sha256_text(result.stdout), "finish_reason": None,
            "usage": usage, "wall_s": time.perf_counter() - started,
        }


class OxiBonsai:
    def __init__(self, port: int, timeout: float) -> None:
        self.port = port
        self.timeout = timeout

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with closing(HTTPConnection("127.0.0.1", self.port, timeout=self.timeout)) as connection:
            connection.request(
                "GET" if payload is None else "POST", path,
                body=None if payload is None else canonical_json(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with connection.getresponse() as response:
                if response.status != 200:
                    raise OSError(f"OxiBonsai returned HTTP {response.status}")
                raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("OxiBonsai response exceeded 64 KiB")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object from OxiBonsai")
        return data

    def model_id(self) -> str:
        models = self.request("/v1/models").get("data")
        if (not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict)
                or models[0].get("owned_by") != "oxibonsai" or not isinstance(models[0].get("id"), str)
                or not models[0]["id"]):
            raise ValueError("expected one model on the local OxiBonsai server")
        return models[0]["id"]

    def choose(self, statement: str, menu: dict[int, str], failed: dict[int, str]) -> dict[str, Any]:
        payload = {
            "messages": [{"role": "user", "content": build_prompt(statement, menu, failed)}],
            "temperature": 0.0, "max_tokens": MAX_TOKENS, "stream": False,
        }
        started = time.perf_counter()
        reply = self.request("/v1/chat/completions", payload)
        choices = reply.get("choices")
        if (not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)):
            raise ValueError("expected one chat completion from OxiBonsai")
        choice = choices[0]
        usage = reply.get("usage")
        tokens = ("prompt_tokens", "completion_tokens")
        if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in tokens):
            usage = None
        return {
            "choice": parse_choice(choice["message"].get("content"), menu), "request_sha256": sha256_json(payload),
            "response_sha256": sha256_json(reply), "finish_reason": choice.get("finish_reason"),
            "usage": usage, "wall_s": time.perf_counter() - started,
        }


def fixture_goals(path: Path | None, split: str, seed: int) -> list[dict[str, Any]]:
    make = held_templates if split == "held_templates" else grounded.propositions
    if path is None:
        salt = SPLITS[split] + random.Random(seed).randrange(2048)
        return [{"id": sha256_json(stmt), "family": family, "statement": stmt, "salt": salt}
                for family, stmt, _ in make(salt)]
    fixture = load_structured(path)
    if fixture["schema"] != "datagoblin.lean-grounded-split.v1" or set(fixture["splits"]) != set(SPLITS):
        raise ValueError("expected the Lean gym's split fixture")
    seen = set()
    for cohort, goals in fixture["splits"].items():
        generate = held_templates if cohort == "held_templates" else grounded.propositions
        for goal in goals:
            salt = goal["salt"]
            if type(salt) is not int or not 1 <= salt <= 99999:
                raise ValueError("Lean fixture salt must be a positive bounded integer")
            expected = {family: stmt for family, stmt, _ in generate(salt)}
            if (expected.get(goal["family"]) != goal["statement"] or goal["id"] != sha256_json(goal["statement"])
                    or goal["id"] in seen):
                raise ValueError("fixture contains a changed or repeated Lean goal")
            seen.add(goal["id"])
    return fixture["splits"][split]


def check_tactic(lean: Path, statement: str, tactic: str, cwd: Path) -> dict[str, Any]:
    proof = f"by\n  repeat' intro\n  {tactic}"
    source = f"set_option maxHeartbeats 100000\ntheorem dgBonsaiProof : {statement} := {proof}\n"
    source += "#print axioms dgBonsaiProof\n"
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [str(lean), "-j", "1", "--stdin"], input=source, cwd=cwd,
            encoding="utf-8", capture_output=True, timeout=20, check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "inconclusive", "reason": "lean_timeout", "source_sha256": sha256_text(source),
                "wall_s": time.perf_counter() - started}
    output = process.stdout + process.stderr
    matches = AXIOM_AUDIT.findall(output)
    axioms = [item.strip() for item in matches[0].split(",")] if len(matches) == 1 and matches[0] else []
    if len(matches) == 1 and process.returncode == 0 and set(axioms).issubset(AXIOMS):
        status = "verified"
    elif LEAN_LIMIT.search(output) or (process.returncode != 0 and "error:" not in output):
        status = "inconclusive"
    elif process.returncode or len(matches) == 1:
        status = "rejected"
    else:
        status = "inconclusive"
    return {"status": status, "proof": proof, "axioms": axioms, "source_sha256": sha256_text(source),
            "output_sha256": sha256_text(output), "wall_s": time.perf_counter() - started}


def probe_library(lean: Path) -> tuple[list[dict[str, str]], str]:
    entries = [{"name": f"{module.split('/', 1)[0]}.{name}"}
               for module, names in grounded.LIBRARY.items() for name in names]
    source = "".join(f"#check @{entry['name']}\n" for entry in entries)
    try:
        result = subprocess.run(
            [str(lean), "-j", "1", "--stdin"], input=source, capture_output=True,
            encoding="utf-8", timeout=20, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OSError("Lean lemma probe timed out") from exc
    if result.returncode:
        first_error = next((line for line in result.stdout.splitlines() if "error:" in line), result.stderr.strip())
        raise ValueError(f"the Lean toolchain lacks a registered core lemma: {first_error}")
    return entries, sha256_text(result.stdout)


def run_lean_bonsai(
    out: Path, *, lean: Path, checkpoint: Path, port: int, split: str, fixtures: Path | None,
    seed: int, max_goals: int, attempts: int, request_timeout: float,
    oxibonsai_bin: Path | None = None, tokenizer: Path | None = None,
) -> dict[str, Any]:
    if (oxibonsai_bin is None) != (tokenizer is None):
        raise ValueError("direct inference requires both --oxibonsai-bin and --tokenizer")
    max_timeout = 300 if oxibonsai_bin else 60
    if not (1 <= max_goals <= 128 and 1 <= attempts <= 8 and 1 <= request_timeout <= max_timeout):
        raise ValueError(f"use 1-128 goals, 1-8 attempts, and a 1-{max_timeout} second request timeout")
    if not 1 <= port <= 65535:
        raise ValueError("invalid OxiBonsai port")
    if out.exists():
        raise ValueError("Lean rollout output already exists")
    goals = fixture_goals(fixtures, split, seed)[:max_goals]
    if not goals:
        raise ValueError("the selected fixture split is empty")
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint SHA-256 differs from {CHECKPOINT}/Bonsai-8B.gguf")
    lean = lean.resolve(strict=True)
    engine: OxiBonsai | DirectOxiBonsai
    if oxibonsai_bin is not None and tokenizer is not None:
        engine = DirectOxiBonsai(oxibonsai_bin, checkpoint, tokenizer, request_timeout)
        model_id = CHECKPOINT
        runtime_info = {
            "transport": "stdio", "checkpoint_binding": "subprocess_argument", "model_id": model_id,
            "runtime_version": engine.version, "runtime_binary_sha256": sha256_file(engine.binary),
            "tokenizer_sha256": sha256_file(engine.tokenizer),
        }
    else:
        engine = OxiBonsai(port, request_timeout)
        try:
            model_id = engine.model_id()
        except HTTPException as exc:
            raise OSError(f"OxiBonsai preflight failed: {exc}") from exc
        runtime_info = {"transport": "http", "checkpoint_binding": "operator_started_server",
                        "served_model_id": model_id}
    try:
        version = subprocess.run(
            [str(lean), "--version"], capture_output=True, encoding="utf-8", timeout=20, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise OSError("Lean version check timed out") from exc
    if version.returncode or not version.stdout.strip().startswith("Lean (version 4."):
        raise ValueError("--lean must point to a working Lean 4 toolchain executable")
    parts, library_probe = probe_library(lean)
    out = out.resolve()
    out.mkdir(parents=True)
    plan = {
        "schema": "datagoblin.lean-bonsai.v1", "dynamics_kind": "black_box_refinement",
        "runtime": "cool-japan/oxibonsai", **runtime_info,
        "checkpoint": CHECKPOINT, "checkpoint_file_sha256": checkpoint_hash,
        "split": split, "seed": seed,
        "goals": goals, "fixture_sha256": sha256_file(fixtures) if fixtures else None,
        "lean_version": version.stdout.strip(), "lean_binary_sha256": sha256_file(lean),
        "library_sha256": sha256_json(parts), "library_probe_sha256": library_probe,
        "collector_sha256": sha256_file(Path(__file__)),
        "attempts_per_arm": attempts, "max_completion_tokens": MAX_TOKENS,
        "request_timeout_s": request_timeout, "axioms_allowed": sorted(AXIOMS),
    }
    dump_structured(out / "plan.json", plan)
    totals = {"lexical": Counter(), "oxibonsai": Counter()}
    with (
        (out / "rollouts.jsonl").open("w", encoding="utf-8") as traces,
        (out / "sft.jsonl").open("w", encoding="utf-8") as sft,
    ):
        for goal in goals:
            statement = goal["statement"]
            _, retrieved = grounded.candidates(statement, parts)
            pool = tuple(dict.fromkeys((*retrieved, *TACTICS)))
            menu = dict(enumerate(pool, start=1))
            rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in totals}
            for arm in totals:
                remaining = menu.copy()
                failed: dict[int, str] = {}
                for _ in range(attempts):
                    row: dict[str, Any] = {}
                    if arm == "oxibonsai":
                        try:
                            row = engine.choose(statement, remaining, failed)
                        except (OSError, HTTPException, UnicodeError, KeyError, TypeError, ValueError) as exc:
                            rows[arm].append({"status": "inconclusive", "error": str(exc)})
                            break
                        totals[arm]["responses"] += 1
                        if row["usage"] is None:
                            totals[arm]["unknown_usage"] += 1
                        else:
                            for key in ("prompt_tokens", "completion_tokens"):
                                totals[arm][key] += row["usage"][key]
                        choice = row["choice"]
                    else:
                        choice = next(iter(remaining))
                    if choice is None:
                        rows[arm].append({**row, "status": "invalid_choice"})
                        continue
                    tactic = remaining.pop(choice)
                    verdict = check_tactic(lean, statement, tactic, out)
                    rows[arm].append({**row, "choice": choice, "tactic": tactic, **verdict})
                    if verdict["status"] == "verified":
                        if arm == "oxibonsai" and split == "train":
                            sft.write(canonical_json({"goal": goal, "split": split, "model_id": model_id,
                                                      "checkpoint_sha256": checkpoint_hash, **verdict}) + "\n")
                        break
                    if verdict["status"] == "rejected":
                        failed[choice] = tactic
                statuses = {row["status"] for row in rows[arm]}
                if "verified" in statuses:
                    totals[arm]["verified"] += 1
                elif "inconclusive" in statuses:
                    totals[arm]["inconclusive"] += 1
                else:
                    totals[arm]["exhausted"] += 1
                totals[arm]["invalid_choices"] += sum(row["status"] == "invalid_choice" for row in rows[arm])
                totals[arm]["checks"] += sum("tactic" in item for item in rows[arm])
            traces.write(canonical_json({"goal": goal, "split": split, "pool": pool, "arms": rows}) + "\n")
            traces.flush()
            sft.flush()
    result = {"goals": len(goals), "split": split, "arms": {arm: dict(count) for arm, count in totals.items()},
              "plan_sha256": sha256_file(out / "plan.json"), "rollouts_sha256": sha256_file(out / "rollouts.jsonl"),
              "sft_sha256": sha256_file(out / "sft.jsonl")}
    dump_structured(out / "summary.json", result)
    return result
