"""RLT-HARD-02A: source-pinned Mathlib proof-state intake and native ranker pilot.

Intended as a development gate. Generated Mathlib-dependent theorems are not
preexisting Mathlib benchmark statements. All successful proofs are checked
by the real Lean kernel, then recompiled and axiom-audited independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .external_checked_exit import digest
from .external_native_adapters import LeanKernelChecker
from .live_lean_kernel import LeanNativeSession

MATHLIB_COMMIT = "809c3fb3b5c8f5d7dace56e200b426187516535a"
TOOLCHAIN = "4.12.0"
SEED = 2026112202
SPLITS = ("train", "calibration", "ordinary_test", "ood_test")
UPSTREAM_FILES = {
    "train": "Mathlib/Data/Set/Basic.lean",
    "calibration": "Mathlib/Data/Finset/Card.lean",
    "ordinary_test": "Mathlib/Data/List/Basic.lean",
    "ood_test": "Mathlib/Data/Nat/GCD/Basic.lean",
}
BUDGET = 5
THRESHOLDS = (0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 0.95)
AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _witness(
    split: str, i: int, *, numeric_shift: int = 0, depth_shift: int = 0
) -> tuple[str, str, str]:
    """Fresh derived theorem and a test-only witness candidate (held off the model)."""
    n = 311 + i * 71 + numeric_shift
    if split == "train":
        statement = (
            "∀ (s t u : Set Nat) (f : Nat → Nat), "
            f"s ⊆ t → t ⊆ u → f '' (s ∩ {{{n}}}) ⊆ f '' u"
        )
        prefix = "intro s t u f hst htu"
        proof = (
            "intro y hy\n"
            "rcases hy with ⟨x, hx, rfl⟩\n"
            "exact ⟨x, htu (hst hx.1), rfl⟩"
        )
    elif split == "calibration":
        statement = (
            "∀ (s t u : Finset Nat), s ⊆ t → "
            f"((s ∩ ({{{n}}} ∪ u)).card ≤ (t ∪ u).card)"  # noqa: RUF001
        )
        prefix = "intro s t u hst"
        proof = (
            "apply Finset.card_le_card\n"
            "exact (Finset.inter_subset_left).trans (hst.trans (Finset.subset_union_left))"
        )
    elif split == "ordinary_test":
        statement = (
            "∀ (xs ys : List Nat), xs.length = ys.length → "
            f"(xs ++ [{n}]).length = (ys ++ [{n}]).length"
        )
        prefix = "intro xs ys h"
        proof = (
            "simpa [List.length_append] using "
            "congrArg (fun m : Nat => m + 1) h"
        )
    else:
        terms = ["(b + c)"] * (i + 2 + depth_shift)
        expr = terms[0]
        for term in terms[1:]:
            expr = f"({expr} + {term})"
        tree = "dvd_add hab hac"
        for _ in range(i + 1 + depth_shift):
            tree = f"dvd_add ({tree}) (dvd_add hab hac)"
        if numeric_shift:
            statement = (
                f"∀ (a b c : Nat), (a ∣ b ∧ a ∣ c) → a ∣ {n} → a ∣ ({expr} + {n})"  # noqa: RUF001
            )
            prefix = "intro a b c h hn"
            proof = f"rcases h with ⟨hab, hac⟩\nexact dvd_add ({tree}) hn"
        else:
            statement = f"∀ (a b c : Nat), (a ∣ b ∧ a ∣ c) → a ∣ {expr}"  # noqa: RUF001
            prefix = "intro a b c h"
            proof = f"rcases h with ⟨hab, hac⟩\nexact {tree}"
    return statement, prefix, proof



def matched_candidate_menu(split: str, witness: str, *, witnessed: bool) -> list[str]:
    """Five near-miss proof scripts with exactly matched UTF-8 byte lengths.

    The generator uses source-observable candidate text. Only native Lean
    compiler/kernel receipts supply outcome labels; the construction's
    witness flag is withheld from the learned policy and public task.
    """
    if split == "train":
        mutations = [
            witness.replace("hst hx.1", "hst hx.2"),
            witness.replace("htu (hst", "hst (htu"),
            witness.replace("htu (hst", "htu (htu"),
            witness.replace("htu (hst", "hst (hst"),
            witness.replace("intro y hy", "intro y hx"),
        ]
    elif split == "calibration":
        mutations = [
            witness.replace("inter_subset_left", "inter_subset_right"),
            witness.replace("subset_union_left", "subset_union_right"),
            witness.replace("hst.trans", "hst.symm"),
            witness.replace("card_le_card", "card_lt_card"),
            witness.replace("inter_subset_left", "inter_subset_right").replace(
                "subset_union_left", "subset_union_right"
            ),
        ]
    elif split == "ordinary_test":
        mutations = [witness.replace("m + 1", f"m + {n}") for n in (0, 2, 3, 4, 5)]
    elif split == "ood_test":
        opening, marker, proof = witness.partition("\nexact ")
        if not marker:
            raise ValueError("OOD witness has no explicit proof body")
        mutations = [
            opening + marker + proof.replace("dvd_add hab hac", "dvd_add hac hab"),
            opening + marker + proof.replace("hab", "hac"),
            opening + marker + proof.replace("hac", "hab"),
            opening + marker + proof.replace("dvd_add", "dvd_mul", 1),
            opening + marker + proof.replace("dvd_add", "dvd_mul"),
        ]
    else:
        raise ValueError("unregistered Mathlib source split")
    actions = [witness, *mutations[:4]] if witnessed else mutations
    if len(actions) != BUDGET or len(set(actions)) != BUDGET or (witness in actions) != witnessed:
        raise ValueError("malformed or duplicate near-miss proof candidates")
    if any(any(bad in candidate.lower() for bad in ("sorry", "admit", "run_tac", "unsafe"))
           for candidate in actions):
        raise ValueError("untrusted tactic in near-miss candidate menu")
    # Whitespace neutralizes script-length shortcuts without introducing
    # fabricated proof success, model hints or extra executable tactics.
    max_length = max(len(candidate.encode("utf-8")) for candidate in actions)
    padded = [candidate + " " * (max_length - len(candidate.encode("utf-8"))) for candidate in actions]
    if len({len(candidate.encode("utf-8")) for candidate in padded}) != 1:
        raise ValueError("candidate length matching failed")
    return padded




@dataclass(frozen=True)
class Root:
    root_id: str
    split: str
    family: str
    statement: str
    prefix: str
    menu: tuple[str, ...]
    source_file: str

    def public_task(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source_file": self.source_file,
            "source_identity": self.root_id,
            "upstream_module": UPSTREAM_FILES[self.split],
            "import": "Mathlib",
            "menu_sha256": digest(self.menu),
        }


def generate(*, per_split: int = 8, seed: int = SEED, matched_menus: bool = False) -> list[Root]:
    if type(per_split) is not int or not 4 <= per_split <= 16 or per_split % 4:
        raise ValueError("per_split must be a multiple of 4 in 4..16")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative source seed required")
    roots = []
    for split in SPLITS:
        for i in range(per_split):
            if matched_menus:
                statement, prefix, witness = _witness(
                    split, i, numeric_shift=17 + seed % 41, depth_shift=1 + seed % 3
                )
                actions = matched_candidate_menu(split, witness, witnessed=i % 4 != 3)
            else:
                statement, prefix, witness = _witness(split, i)
                # Original positive control: one long proof and four trivial actions.
                actions = ["rfl", "simp", "constructor", "exact True.intro"]
                actions.append(witness if i % 4 != 3 else "exact False.elim (by assumption)")
            registration = f"{seed}/{split}/{i}/{statement}"
            marker = sha((registration + ("/matched" if matched_menus else "")).encode())[:20]
            random.Random(int(marker, 16)).shuffle(actions)
            source_file = f"source_roots/{split}/MathlibRoot_{i:02d}_{marker}.lean"
            roots.append(Root(
                root_id=f"hard02:{split}:{marker}",
                split=split, family=f"mathlib_{split}",
                statement=statement, prefix=prefix, menu=tuple(actions),
                source_file=source_file,
            ))
    if len({r.root_id for r in roots}) != len(roots):
        raise ValueError("duplicate source identity")
    if len({r.statement for r in roots}) != len(roots):
        raise ValueError("duplicated source theorem")
    if len({r.source_file for r in roots}) != len(roots):
        raise ValueError("source file overlap")
    return roots


def verify_upstream(root: Path, lake_bin: Path) -> dict[str, Any]:
    if not root.is_dir() or not lake_bin.is_file():
        raise ValueError("real Mathlib checkout and Lake binary are required")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], encoding="utf-8", timeout=10
    ).strip()
    if head != MATHLIB_COMMIT:
        raise ValueError(f"unregistered Mathlib commit: {head}")
    toolchain = (root / "lean-toolchain").read_text().strip()
    if toolchain != f"leanprover/lean4:v{TOOLCHAIN}":
        raise ValueError(f"Mathlib toolchain drift: {toolchain}")
    version = subprocess.run(
        [str(lake_bin), "env", "lean", "--version"], cwd=root,
        capture_output=True, text=True, timeout=30, check=True,
    ).stdout.strip()
    if f"version {TOOLCHAIN}" not in version:
        raise ValueError(f"unexpected native Lean: {version}")
    module_sha = {}
    for p in UPSTREAM_FILES.values():
        file = root / p
        if not file.is_file():
            raise ValueError(f"pinned Mathlib module unavailable: {p}")
        module_sha[p] = sha(file.read_bytes())
    lean_bin = subprocess.check_output(
        [str(lake_bin), "env", "lean", "--print-prefix"], cwd=root,
        text=True, timeout=30,
    ).strip()
    binary = Path(lean_bin) / "bin/lean"
    if not binary.is_file():
        raise ValueError("pinned toolchain is missing compiler binary")
    return {
        "mathlib_commit": head, "lean_version": version,
        "lean_binary_sha256": sha(binary.read_bytes()),
        "mathlib_source_sha256": module_sha,
        "lakefile_sha256": sha((root / "lakefile.lean").read_bytes()),
        "lockfile_sha256": sha((root / "lake-manifest.json").read_bytes()),
    }


class MathlibSession(LeanNativeSession):
    def __init__(
        self, lake_bin: Path, mathlib_root: Path, directory: Path,
        roots: list[Root],
    ):
        super().__init__(lake_bin, directory, {i + 1: r.statement for i, r in enumerate(roots)})
        self.lake_bin, self.mathlib_root = lake_bin, mathlib_root
        self.prefixes = {i + 1: r.prefix for i, r in enumerate(roots)}

    def _compile(self, source: Path) -> subprocess.CompletedProcess[str]:
        started = time.perf_counter()
        try:
            output = subprocess.run(
                [str(self.lake_bin), "env", "lean", str(source.resolve())],
                cwd=self.mathlib_root, capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired:
            output = subprocess.CompletedProcess(
                ["lake", "env", "lean"], 124, "", "lean_candidate_timeout"
            )
        self.seconds += time.perf_counter() - started
        return output

    def _source(self, sid: int, name: str, body: str) -> Path:
        target = self.directory / f"{name}.lean"
        lines = [
            "import Mathlib",
            "set_option autoImplicit false",
            f"theorem {name} : {self.statements[sid]} := by",
            *[f"  {line}" for line in self.prefixes[sid].splitlines()],
            *[f"  {line}" for line in body.splitlines()],
            f"#print axioms {name}",
        ]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    def probe(self, sid: int) -> dict[str, Any]:
        """Collect the compiler's real open goal after an authorized prefix."""
        source = self._source(sid, f"dgProbe{sid}", "trace_state")
        result = self._compile(source)
        output = result.stdout + "\n" + result.stderr
        if "⊢" not in output:
            raise ValueError(f"missing Lean native open-goal trace: {output[-700:]}")
        # May exit nonzero because the proof is intentionally unfinished.
        return {
            "goal_pp": output[-3200:], "trace_sha256": sha(output.encode()),
            "returncode": result.returncode, "source_sha256": sha(source.read_bytes()),
        }

    def run(self, sid: int, tactic: str) -> dict[str, Any]:
        if sid not in self.statements or not isinstance(tactic, str):
            raise ValueError("unregistered source state/action")
        self.calls += 1
        hash_input = f"{sid}:{self.statements[sid]}:{self.prefixes[sid]}:{tactic}"
        cid = int.from_bytes(hashlib.sha256(hash_input.encode()).digest()[:8], "big")
        name = f"dgCheck{cid}"
        source = self._source(sid, name, tactic)
        result = self._compile(source)
        passed = result.returncode == 0
        if passed:
            self.proof_by_sid[cid] = (source, name)
        return {
            "ok": passed,
            "state": {"sid": cid, "closed": passed, "goals": [] if passed else ["open"]},
            "error": "" if passed else (result.stdout + result.stderr)[-450:],
        }


def export_sources(base: Path, roots: list[Root]) -> dict[str, str]:
    checksums = {}
    for root in roots:
        file = base / root.source_file
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.exists():
            raise FileExistsError("source root already exists")
        code = (
            "import Mathlib\nset_option autoImplicit false\n"
            f"-- Frozen source identity: {root.root_id}\n"
            f"theorem registeredGoal : {root.statement} := by\n"
            + "\n".join(f"  {line}" for line in root.prefix.splitlines())
            + "\n  trace_state\n"
        )
        file.write_text(code)
        checksums[root.source_file] = sha(file.read_bytes())
    return checksums


def _action_features(
    goal_pp: str, menu: tuple[str, ...], words: tuple[str, ...], *, semantic: bool,
) -> np.ndarray:
    if semantic:
        from .hard02_semantic_features import features as semantic_features
        return semantic_features(goal_pp, menu)
    from datagoblin.training_gym.rlt_lean_train_rank import features as grounded_features
    return grounded_features(goal_pp, menu, [], words)


def fit_policy(
    train: list[dict[str, Any]], words: tuple[str, ...], *, semantic: bool = False,
) -> np.ndarray:
    if not train or any(e["split"] != "train" for e in train):
        raise ValueError("policy is fitted on TRAIN only")
    X = np.concatenate([
        _action_features(row["goal_pp"], row["menu"], words, semantic=semantic) for row in train
    ])
    y = np.array([float(x["receipt"]["passed"]) for row in train for x in row["checks"]])
    if X.shape[0] != len(y) or y.sum() == 0 or y.sum() == len(y):
        raise ValueError("TRAIN labels have no positive/negative support")
    weights = np.zeros(X.shape[1], dtype=np.float64)
    sample = np.where(y > 0, min(12, len(y) / y.sum()), 1.0)
    for _ in range(1200):
        p = 1 / (1 + np.exp(-np.clip(X @ weights, -30, 30)))
        grad = X.T @ ((p - y) * sample) / len(y)
        grad[1:] += 0.04 * weights[1:]
        weights -= 0.18 * grad
    if not np.isfinite(weights).all():
        raise ValueError("nonfinite TRAIN ranker")
    return weights


def rank_scores(
    goal_pp: str, menu: tuple[str, ...], weights: np.ndarray,
    words: tuple[str, ...], *, semantic: bool = False,
) -> np.ndarray:
    x = _action_features(goal_pp, menu, words, semantic=semantic)
    return 1 / (1 + np.exp(-np.clip(x @ weights, -30, 30)))


def order(
    root: Root, arm: str, goal_pp: str, weights: np.ndarray,
    words: tuple[str, ...], *, semantic: bool = False,
) -> list[int]:
    if arm == "learned":
        ps = rank_scores(goal_pp, root.menu, weights, words, semantic=semantic)
        return sorted(range(BUDGET), key=lambda j: (-float(ps[j]), j))
    if arm == "random_full":
        ids = list(range(BUDGET))
        random.Random(int(sha(f"random/{root.root_id}".encode())[:16], 16)).shuffle(ids)
        return ids
    if arm == "trivial_first":
        return sorted(range(BUDGET), key=lambda j: (
            {"rfl": 0, "simp": 1, "constructor": 2}.get(root.menu[j].split()[0], 3), j
        ))
    if arm == "fixed_full":
        return list(range(BUDGET))
    raise ValueError("unknown arm")


def _check(checker: LeanKernelChecker, root: Root, candidate: int) -> dict[str, Any]:
    start = time.perf_counter()
    receipt = checker(root.public_task(), {"id": candidate})
    if receipt["passed"] and (
        receipt["code"] != "kernel_certified"
        or receipt["evidence"].get("kernel_ok") is not True
        or not set(receipt["evidence"].get("axioms", [])) <= AXIOMS
    ):
        raise ValueError("candidate passed without approved native kernel certificate")
    return {
        "candidate_id": candidate, "receipt": receipt,
        "wall_seconds": round(time.perf_counter() - start, 6),
    }


def calibrate(
    cal: list[dict[str, Any]], weights: np.ndarray, words: tuple[str, ...],
    *, semantic: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    if not cal or any(row["split"] != "calibration" for row in cal):
        raise ValueError("only CAL roots can choose the stopping threshold")
    frontier = []
    for threshold in THRESHOLDS:
        successes, attempts = 0, 0
        for row in cal:
            root = row["root"]
            labels = {v["candidate_id"]: v["receipt"]["passed"] for v in row["checks"]}
            ranked = order(root, "learned", row["goal_pp"], weights, words, semantic=semantic)
            ps = rank_scores(row["goal_pp"], root.menu, weights, words, semantic=semantic)
            for ix, candidate in enumerate(ranked):
                if ix and max(float(ps[k]) for k in ranked[ix:]) < threshold:
                    break
                attempts += 1
                if labels[candidate]:
                    successes += 1
                    break
        utility = successes - 0.03 * attempts
        frontier.append({
            "threshold": threshold, "certified": successes,
            "attempts": attempts, "utility": round(utility, 5),
        })
    best = max(frontier, key=lambda p: (p["utility"], p["threshold"]))
    return float(best["threshold"]), frontier


def online(
    row: dict[str, Any], checker: LeanKernelChecker, arm: str,
    weights: np.ndarray, words: tuple[str, ...], threshold: float,
    *, semantic: bool = False,
) -> dict[str, Any]:
    root: Root = row["root"]
    policy_arm = "learned" if arm == "legacy_grounded" else arm
    ordering = order(root, policy_arm, row["goal_pp"], weights, words, semantic=semantic)
    ps = (
        rank_scores(row["goal_pp"], root.menu, weights, words, semantic=semantic)
        if policy_arm == "learned" else None
    )
    attempts = []
    for j, cid in enumerate(ordering):
        if policy_arm == "learned" and j and ps is not None and max(float(ps[k]) for k in ordering[j:]) < threshold:
            break
        checked = _check(checker, root, cid)
        attempts.append(checked)
        if checked["receipt"]["passed"]:
            break
    return {
        "root_id": root.root_id, "split": root.split, "arm": arm,
        "certified": bool(attempts and attempts[-1]["receipt"]["passed"]),
        "attempts": attempts, "stopped_before_budget": len(attempts) < BUDGET
        and not (attempts and attempts[-1]["receipt"]["passed"]),
    }


def run(
    lake: Path, mathlib: Path, out: Path, *, per_split: int = 8,
    seed: int = SEED, matched_menus: bool = False, semantic_features: bool = False,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("evidence directory must be create-only")
    env = verify_upstream(mathlib.resolve(), lake.resolve())
    roots = generate(per_split=per_split, seed=seed, matched_menus=matched_menus)
    out.mkdir(parents=True)
    source_hashes = export_sources(out, roots)
    probes = {}
    campaign = []
    source_revisions = {
        "source": source_hashes, "upstream": env,
        "driver": sha(Path(__file__).read_bytes()),
    }
    if semantic_features:
        from . import hard02_semantic_features
        source_revisions["feature_sha256"] = sha(Path(hard02_semantic_features.__file__).read_bytes())
    verifier_sha = digest(source_revisions)
    with tempfile.TemporaryDirectory(prefix="dg-hard02-") as tmp:
        session = MathlibSession(lake, mathlib, Path(tmp), roots)
        mapping = {
            digest(root.public_task()): {
                "root_sid": i + 1, "menu": dict(enumerate(root.menu))
            } for i, root in enumerate(roots)
        }
        checker = LeanKernelChecker(lean=session, states=mapping, verifier_sha=verifier_sha)
        for i, root in enumerate(roots, 1):
            probes[root.root_id] = session.probe(i)
        train, cal = [], []
        for dest, split in ((train, "train"), (cal, "calibration")):
            for root in (r for r in roots if r.split == split):
                checks = [_check(checker, root, i) for i in range(BUDGET)]
                dest.append({
                    "root": root, "split": split,
                    "goal_pp": probes[root.root_id]["goal_pp"], "checks": checks,
                })
        from datagoblin.training_gym import rlt_lean_grounded
        words = tuple(sorted(set().union(*(
            rlt_lean_grounded.words(row["goal_pp"]) for row in train
        ))))[:128]
        training = [
            {"split": "train", "goal_pp": row["goal_pp"], "menu": row["root"].menu, "checks": row["checks"]}
            for row in train
        ]
        weights = fit_policy(training, words, semantic=semantic_features)
        threshold, frontier = calibrate(cal, weights, words, semantic=semantic_features)
        legacy_weights = None
        legacy_threshold = None
        legacy_frontier = None
        if semantic_features:
            # Direct within-cohort ablation: identical TRAIN labels, CAL roots
            # and held-out theorems, with the legacy representation only.
            legacy_weights = fit_policy(training, words, semantic=False)
            legacy_threshold, legacy_frontier = calibrate(cal, legacy_weights, words, semantic=False)
        for split in ("ordinary_test", "ood_test"):
            for root in (r for r in roots if r.split == split):
                state = {"root": root, "goal_pp": probes[root.root_id]["goal_pp"]}
                arms = ("learned", "fixed_full", "random_full", "trivial_first")
                if semantic_features:
                    arms += ("legacy_grounded",)
                for arm in arms:
                    if arm == "legacy_grounded":
                        if legacy_weights is None or legacy_threshold is None:
                            raise ValueError("missing TRAIN/CAL legacy ablation")
                        campaign.append(online(
                            state, checker, arm, legacy_weights, words, legacy_threshold,
                            semantic=False,
                        ))
                    else:
                        campaign.append(online(
                            state, checker, arm, weights, words, threshold,
                            semantic=semantic_features,
                        ))
        replayed_by_arm = {"learned": 0, "legacy_grounded": 0}
        for result in campaign:
            if result["arm"] not in replayed_by_arm:
                continue
            root = next(r for r in roots if r.root_id == result["root_id"])
            for original in result["attempts"]:
                check = _check(checker, root, original["candidate_id"])
                if check["receipt"] != original["receipt"]:
                    raise ValueError("independent Lean source replay mismatch")
                replayed_by_arm[result["arm"]] += 1
        proof_dir = out / "certified_proofs"
        proof_dir.mkdir()
        certified_sources = {}
        for source, _ in session.proof_by_sid.values():
            target = proof_dir / source.name
            shutil.copyfile(source, target)
            certified_sources[target.relative_to(out).as_posix()] = sha(target.read_bytes())
        compiler_calls = session.calls + session.kernel_replays
        compiler_time = session.seconds
    weights_path = out / "weights.json"
    weights_path.write_text(json.dumps({
        "weights": weights.tolist(), "word_vocabulary": words,
        "training_splits": ["train"], "calibration_threshold": threshold,
        "feature_profile": "lexical_hash_v1" if semantic_features else "grounded_position_v1",
        "legacy_weights": legacy_weights.tolist() if legacy_weights is not None else None,
        "legacy_CAL_threshold": legacy_threshold,
    }, sort_keys=True, indent=2) + "\n")
    source_records = out / "source_records.jsonl"
    with source_records.open("w") as f:
        for root in roots:
            f.write(json.dumps({
                "root_id": root.root_id, "split": root.split,
                "public_task": root.public_task(), "prefix": root.prefix,
                "menu": root.menu, "goal_state": probes[root.root_id],
            }, sort_keys=True) + "\n")
    receipt_path = out / "native_receipts.jsonl"
    with receipt_path.open("w") as f:
        for block in (train, cal):
            for row in block:
                f.write(json.dumps({
                    "root_id": row["root"].root_id, "split": row["split"],
                    "arm": "full_menu_fit" if row["split"] == "train" else "full_menu_cal",
                    "attempts": row["checks"],
                }, sort_keys=True) + "\n")
        for row in campaign:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    by_split = {}
    for split in ("ordinary_test", "ood_test"):
        by_split[split] = {}
        arms = ("learned", "fixed_full", "random_full", "trivial_first")
        if semantic_features:
            arms += ("legacy_grounded",)
        for arm in arms:
            cohort = [r for r in campaign if r["split"] == split and r["arm"] == arm]
            by_split[split][arm] = {
                "kernel_certified": sum(r["certified"] for r in cohort),
                "roots": len(cohort),
                "attempts": sum(len(r["attempts"]) for r in cohort),
                "candidate_wall_seconds": round(sum(
                    a["wall_seconds"] for r in cohort for a in r["attempts"]
                ), 5),
                "failed_continuation_stops": sum(
                    bool(r["stopped_before_budget"]) for r in cohort
                ),
            }
    report = {
        "schema": "datagoblin.rlt_hard02_mathlib_first_dev.v1",
        "evidence_class": "real_mathlib_derived_compiler_states_kernel_proofs",
        "environment": env, "source_seed": seed,
        "candidate_profile": "length_matched_near_miss_v2" if matched_menus else "trivial_controls_v1",
        "feature_profile": "lexical_hash_v1" if semantic_features else "grounded_position_v1",
        "source_roots": len(roots), "per_split": per_split,
        "split_upstream_files": UPSTREAM_FILES,
        "fit_train": {
            "roots": len(train), "successful_candidates": sum(
                a["receipt"]["passed"] for r in train for a in r["checks"]
            ),
            "exhausted_menus": sum(
                not any(a["receipt"]["passed"] for a in r["checks"]) for r in train
            ),
        },
        "calibration": {
            "threshold": threshold, "frontier": frontier,
            "legacy_grounded_threshold": legacy_threshold,
            "legacy_grounded_frontier": legacy_frontier,
        },
        "paired_feature_ablation": bool(semantic_features),
        "development_results": by_split,
        "learned_attempts_replayed": replayed_by_arm["learned"],
        "independent_policy_replays_by_arm": replayed_by_arm,
        "compiler_calls_including_probes_and_audits": compiler_calls + len(roots),
        "compiler_seconds": round(compiler_time, 5),
        "certified_source_sha256": certified_sources,
        "source_sha256": source_hashes,
        "model_sha256": sha(weights_path.read_bytes()),
        "limitations": [
            "new_parametric_theorems_derived_from_pinned_Mathlib_operations",
            "family_disjoint_splits_across_4_domains",
            "fixed_five_candidate_menu_not_free_form_tactic_generation",
            "prefix_probed_by_native_Lean_trace_state",
            "finite_menu_exhaustion_does_not_imply_unprovability",
            "development_only_no_sealed_confirmation_or_SWEbench",
            "trivial_hardness_gate_requires_native_audit_before_promotion",
        ],
    }
    report["files_sha256"] = {
        name: sha((out / name).read_bytes()) for name in (
            "weights.json", "source_records.jsonl", "native_receipts.jsonl"
        )
    }
    (out / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake", type=Path, required=True)
    parser.add_argument("--mathlib", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--source-seed", type=int, default=SEED)
    parser.add_argument("--matched-menus", action="store_true")
    parser.add_argument("--semantic-features", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.lake, args.mathlib, args.out, per_split=args.per_split,
        seed=args.source_seed, matched_menus=args.matched_menus,
        semantic_features=args.semantic_features,
    )
    print(json.dumps({
        "results": report["development_results"], "source_roots": report["source_roots"],
        "training": report["fit_train"], "compiler_calls": report["compiler_calls_including_probes_and_audits"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
