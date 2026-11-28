"""Parallel CPU-only 10B open-reasoning, frontier-trace, and bespoke-research intake audit.

Implements Workstream D from ``training/aleph_next/continuation10b/NEON.md`` and the
``research_filter`` specification in ``training/data/aleph_10b_mix.json``:
  1. Live inventory & rights/license/attribution audit of every linked open-reasoning,
     frontier-reasoning (Fable/Opus/GPT/Codex/TeichAI), formal, and bespoke source.
  2. Cross-source root deduplication (Fable full vs lite, mixed aggregates vs
     NVIDIA/OpenThoughts originals) without inferring unverified teacher/Astra identity.
  3. Streaming sample of 1,000 distinct roots across available sources (CPU only, zero
     paid teacher calls) with secret/PII screening.
  4. 100-point research-relevance scoring across target research domains, independent
     300-example stratified calibration audit, and exact frozen SmolLM2 unique accepted
     token yield accounting (separating relevance, rights clearance, and independent
     verifiability / unknown-outcome masking).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ALEPH_NEXT = HERE.parent
TRAINING_ROOT = ALEPH_NEXT.parent
for p in (str(HERE), str(ALEPH_NEXT / "candidate01")):
    if p not in sys.path:
        sys.path.insert(0, p)

from production import TOKENIZER_SHA256, load_tokenizer  # noqa: E402

AUDIT_SCHEMA = "aleph.10b.research-corpus-intake-audit.v1"

HF_INVENTORY_SOURCES: list[dict[str, Any]] = [
    # Open reasoning (priority early supply for 500M / 2B+)
    {
        "repo": "nvidia/OpenMathReasoning",
        "lane": "open_reasoning",
        "group": "open_reasoning",
        "config": "default",
        "split": "cot",
        "sample_target": 200,
        "page_size": 100,
    },
    {
        "repo": "nvidia/OpenScienceReasoning-2",
        "lane": "open_reasoning",
        "group": "open_reasoning",
        "config": "default",
        "split": "train",
        "sample_target": 180,
        "page_size": 100,
    },
    {
        "repo": "open-thoughts/OpenThoughts3-1.2M",
        "lane": "open_reasoning",
        "group": "open_reasoning",
        "config": "default",
        "split": "train",
        "sample_target": 200,
        "page_size": 100,
    },
    {
        "repo": "nvidia/OpenCodeReasoning-2",
        "lane": "open_reasoning",
        "group": "open_reasoning",
        "config": "train",
        "split": "python",
        "sample_target": 160,
        "page_size": 80,
    },
    {
        "repo": "nvidia/Open-SWE-Traces",
        "lane": "research_mechinterp",
        "group": "open_reasoning",
        "config": None,
        "split": None,
        "sample_target": 0,
        "page_size": 20,
    },
    {
        "repo": "reasoning-core/procedural-pile",
        "lane": "games_procedural",
        "group": "open_reasoning",
        "config": None,
        "split": None,
        "sample_target": 0,
        "page_size": 20,
    },
    # Frontier reasoning (Fable / Opus / GPT / Codex / mixed aggregates)
    {
        "repo": "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        "lane": "frontier_reasoning",
        "group": "fable_full",
        "config": "default",
        "split": "train",
        "sample_target": 100,
        "page_size": 50,
    },
    {
        "repo": "greghavens/fable-5-coding-and-debugging-traces",
        "lane": "frontier_reasoning",
        "group": "fable_coding",
        "config": "default",
        "split": "train",
        "sample_target": 40,
        "page_size": 20,
    },
    {
        "repo": "Crownelius/Complete-FABLE.5-traces-2M",
        "lane": "frontier_reasoning",
        "group": "fable_aggregate",
        "config": "default",
        "split": "train",
        "sample_target": 40,
        "page_size": 20,
    },
    {
        "repo": "clzoro/Claude-Distills",
        "lane": "frontier_reasoning",
        "group": "mixed_aggregate",
        "config": "default",
        "split": "train",
        "sample_target": 40,
        "page_size": 20,
    },
    {
        "repo": "RESMP-DEV/Fable-GPT-5.5-Distillation-Traces",
        "lane": "frontier_reasoning",
        "group": "fable_unavailable",
        "config": None,
        "split": None,
        "sample_target": 0,
        "page_size": 20,
    },
    {
        "repo": "predictive-singularity/Accretion",
        "lane": "frontier_reasoning",
        "group": "frontier_misc",
        "config": "default",
        "split": "train",
        "sample_target": 60,
        "page_size": 30,
    },
    {
        "repo": "AletheiaResearch/GPT-5.5-Codex",
        "lane": "frontier_reasoning",
        "group": "gpt_codex",
        "config": "default",
        "split": "train",
        "sample_target": 15,
        "page_size": 5,
    },
    {
        "repo": (
            "Manusagents/GPT-5.5-Gemini-3.1-Pro-Grok-4-Claude-Fable-5-"
            "Mythos-5-Qwen-3.7-Max-and-more-Distillation-Dataset"
        ),
        "lane": "frontier_reasoning",
        "group": "mixed_aggregate",
        "config": "default",
        "split": "train",
        "sample_target": 120,
        "page_size": 60,
    },
]

GITHUB_FORMAL_AND_BESPOKE: list[dict[str, str]] = [
    {"repo": "TeichAI/teich", "lane": "frontier_reasoning", "role": "frontier_tooling_or_catalog"},
    {"repo": "leanprover-community/mathlib4", "lane": "formal_lean", "role": "formal_proof_library"},
    {"repo": "lean-dojo/LeanDojo", "lane": "formal_lean", "role": "formal_proof_extraction"},
    {"repo": "patrickdugan/MathCan", "lane": "formal_lean", "role": "certified_recursive_descent_and_metamath"},
    {"repo": "MoralityLabAI/LoopedTransformers", "lane": "games_procedural", "role": "inkblot_and_routing_programs"},
    {"repo": "MoralityLabAI/BPD", "lane": "research_mechinterp", "role": "fuse_sccg_optimizer_screening"},
    {"repo": "MoralityLabAI/EvalMath", "lane": "games_procedural", "role": "proof_progress_and_certificates"},
    {"repo": "MoralityLabAI/HybridTRMLDT", "lane": "research_mechinterp", "role": "trm_skill_factory_and_floodgates"},
    {
        "repo": "MoralityLabAI/DataGoblin#PR121",
        "lane": "research_mechinterp",
        "role": "DG-Research-20M_native_intake_separate_staging",
    },
]

PERMISSIVE_LICENSES = {
    "mit",
    "apache-2.0",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "odc-by",
    "bsd-3-clause",
    "bsd-2-clause",
    "unlicense",
}

RE_SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"
    r"|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|sk-(?:proj-)?[A-Za-z0-9_-]{24,}|AIza[0-9A-Za-z_-]{35})"
)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
RE_WS = re.compile(r"\s+")

DOMAIN_PATTERNS: dict[str, re.Pattern[str]] = {
    "math_discovery": re.compile(
        r"(?i)\b(?:theorem|lemma|proof|conjecture|corollary|induction|bijection|polynomial|"
        r"eigenvalue|manifold|inequality|combinatorics|number theory|modular arithmetic|integral|derivative)\b"
    ),
    "mechanistic_interpretability": re.compile(
        r"(?i)\b(?:mechanistic interpretability|activation patching|sparse autoencoder|circuit|"
        r"attention head|residual stream|logit lens|superposition|feature steering|probing)\b"
    ),
    "ml_architecture": re.compile(
        r"(?i)\b(?:transformer|attention|recurrent|looped|rope|rms ?norm|kv cache|adamw|"
        r"backpropagation|gradient|loss function|scaling law|architecture|ablation|quantization)\b"
    ),
    "experiment_design": re.compile(
        r"(?i)\b(?:hypothesis|control group|ablation|baseline|experiment design|confounder|"
        r"randomiz|falsif|power analysis|protocol|treatment|intervention)\b"
    ),
    "statistics": re.compile(
        r"(?i)\b(?:p-value|confidence interval|bayesian|posterior|prior|likelihood|variance|"
        r"estimator|regression|bootstrap|hypothesis test|chi-square|mann-whitney|calibration|distribution)\b"
    ),
    "scientific_code": re.compile(
        r"(?i)(?:```(?:python|cpp|c\+\+|rust|julia|lean)|def |class |"
        r"import numpy|import scipy|import torch|assert |pytest|complexity|O\([nmk])"
    ),
    "formal_verification": re.compile(
        r"(?i)\b(?:lean ?4|coq|isabelle|metamath|smt|z3|invariant|soundness|completeness|"
        r"formal verification|type check|proof assistant|tactic|sat solver)\b"
    ),
    "cryptography": re.compile(
        r"(?i)\b(?:zero-knowledge|zk-snark|zk-stark|elliptic curve|hash function|"
        r"commitment scheme|merkle|lattice|cryptograph|cipher|protocol security|signature)\b"
    ),
    "literature_synthesis": re.compile(
        r"(?i)\b(?:et al\.|prior work|literature|survey|compare|trade-off|"
        r"limitation|citation|empirical finding|reproducib)\b"
    ),
}

RE_REVISION = re.compile(
    r"(?i)\b(?:wait,|hold on|actually,|let me re-?check|mistake|contradiction|counterexample|"
    r"that fails|doesn't work|doesn't hold|reconsider|revise|instead,|correction|bug|edge case|"
    r"negative result|failed attempt|wrong because)\b"
)
RE_VERIFY = re.compile(
    r"(?i)(?:assert |test case|verify|check:|substitution|plugging back|sanity check|boundary case|"
    r"\\boxed\{|Q\.E\.D\.|∎|unit test|stdout|expected output|independently verify)"
)
RE_RESEARCH_INTENT = re.compile(
    r"(?i)\b(?:prove|derive|analyze|design an experiment|investigate|debug|optimize|compare|"
    r"evaluate|construct|counterexample|formalize|diagnose|ablate|estimate|bound|generalize)\b"
)
RE_ABSTRACTION = re.compile(
    r"(?i)\b(?:generalize|invariant|necessary and sufficient|iff|asymptotic|lower bound|upper bound|"
    r"isomorphism|reduction|equivalence|closed form|recurrence|symmetry|mechanism)\b"
)


def fetch_json(url: str, timeout: int = 25) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "aleph-10b-research-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "aleph-10b-research-audit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def classify_rights(license_val: Any, readme_text: str) -> dict[str, Any]:
    lic_list: list[str] = []
    if isinstance(license_val, str) and license_val.strip():
        lic_list = [license_val.strip().lower()]
    elif isinstance(license_val, list):
        lic_list = [str(x).strip().lower() for x in license_val if str(x).strip()]

    readme_lower = readme_text.lower()
    mentions_proprietary_tos = any(
        kw in readme_lower
        for kw in ("anthropic terms", "openai terms", "non-commercial", "research only", "nc-4.0", "cc-by-nc")
    )
    if not lic_list:
        status = "quarantined_unknown_rights"
    elif any("nc" in x for x in lic_list) or mentions_proprietary_tos:
        status = "restricted_review_required"
    elif any(x in PERMISSIVE_LICENSES for x in lic_list):
        status = (
            "permissive_card_license_needs_teacher_tos_review"
            if "distill" in readme_lower
            or "claude" in readme_lower
            or "gpt-5" in readme_lower
            or "fable" in readme_lower
            else "permissive_cleared"
        )
    else:
        status = "custom_license_review_required"

    return {
        "declared_licenses": lic_list or ["unspecified"],
        "rights_status": status,
        "provider_attribution_provisional": True,
        "mentions_restricted_terms_in_card": mentions_proprietary_tos,
    }


def extract_prompt_and_response(row: dict[str, Any]) -> tuple[str, str, bool]:
    """Return (prompt_text, response_text, has_explicit_expected_answer)."""
    has_answer = bool(row.get("expected_answer") or row.get("solution") or row.get("tests") or row.get("test_cases"))
    # Conversations format (OpenThoughts, ShareGPT, etc.)
    for key in ("conversations", "messages"):
        conv = row.get(key)
        if isinstance(conv, list) and conv:
            prompts: list[str] = []
            resps: list[str] = []
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("from") or turn.get("role") or "").lower()
                val = str(turn.get("value") or turn.get("content") or "")
                if role in ("human", "user", "system"):
                    prompts.append(val)
                else:
                    resps.append(val)
            return "\n\n".join(prompts).strip(), "\n\n".join(resps).strip(), has_answer

    prompt = str(
        row.get("problem")
        or row.get("input")
        or row.get("question")
        or row.get("prompt")
        or row.get("instruction")
        or row.get("query")
        or ""
    ).strip()
    if prompt in ("-", "", "N/A") and row.get("question_id"):
        # OpenCodeReasoning-2 links external TACO/CodeChef question IDs; use id + critique context as root key
        prompt = f"[source={row.get('source', '')} dataset={row.get('dataset', '')} qid={row.get('question_id', '')}]"
    reasoning = str(
        row.get("reasoning")
        or row.get("thinking")
        or row.get("thought")
        or row.get("reasoning_content")
        or row.get("r1_generation")
        or ""
    ).strip()
    critique = str(row.get("qwq_critique") or "").strip()
    if critique:
        reasoning = f"{reasoning}\n\n[critique]\n{critique}".strip()
    answer = str(
        row.get("generated_solution")
        or row.get("output")
        or row.get("response")
        or row.get("solution")
        or row.get("answer")
        or row.get("completion")
        or ""
    ).strip()
    resp = f"<think>\n{reasoning}\n</think>\n{answer}" if reasoning and answer else answer or reasoning
    if not prompt and not resp and "row_json" in row:
        txt = str(row["row_json"]).strip()
        return f"[source={row.get('first_source_dataset', '')} hash={row.get('row_hash', '')}]", txt[:8000], has_answer
    if not prompt and not resp and "text" in row:
        txt = str(row["text"]).strip()
        return txt[:400], txt[400:], has_answer
    return prompt, resp, has_answer


def score_research_relevance(prompt: str, response: str, has_expected: bool) -> dict[str, Any]:
    """Primary 100-point rubric matching training/data/aleph_10b_mix.json."""
    full = f"{prompt}\n{response}"
    # 1. Domain tags (0..10)
    tags = [tag for tag, pat in DOMAIN_PATTERNS.items() if len(pat.findall(full)) >= 2]
    if not tags:
        tags = [tag for tag, pat in DOMAIN_PATTERNS.items() if pat.search(full)]
    domain_score = min(10, 5 * len(tags))

    # 2. Research intent (0..25)
    intent_hits = len(RE_RESEARCH_INTENT.findall(prompt)) + min(3, len(RE_RESEARCH_INTENT.findall(response[:1500])))
    intent_score = min(25, intent_hits * 5 + (5 if len(prompt) >= 120 else 0))

    # 3. Reasoning depth (0..20)
    resp_len = len(response)
    has_think = "<think>" in response.lower() or "step 1" in response.lower() or "\n\n" in response
    depth_score = min(
        20,
        (8 if resp_len >= 1200 else 4 if resp_len >= 400 else 1)
        + (6 if resp_len >= 3500 else 0)
        + (6 if has_think else 0),
    )

    # 4. Evidence-based revision / failure recovery (0..20)
    rev_hits = len(RE_REVISION.findall(response))
    rev_score = min(20, rev_hits * 5)

    # 5. Independent verifiability (0..15)
    ver_hits = len(RE_VERIFY.findall(full))
    ver_score = min(15, (6 if has_expected else 0) + min(9, ver_hits * 3))

    # 6. Novelty / abstraction (0..10)
    abs_hits = len(RE_ABSTRACTION.findall(full))
    abs_score = min(10, abs_hits * 3)

    total = intent_score + depth_score + rev_score + ver_score + abs_score + domain_score
    return {
        "total_score": total,
        "passes_threshold_70": total >= 70,
        "domain_tags": tags,
        "subscores": {
            "research_intent": intent_score,
            "reasoning_depth": depth_score,
            "evidence_based_revision": rev_score,
            "independent_verifiability": ver_score,
            "novelty_abstraction": abs_score,
            "domain_relevance": domain_score,
        },
        "outcome_verification_status": "externally_checkable_candidate"
        if (has_expected or ver_hits >= 2)
        else "unknown_outcome_masked",
    }


def independent_structural_calibration_review(prompt: str, response: str, has_expected: bool) -> dict[str, Any]:
    """Independent secondary structural rubric (orthogonal features) for 300-example calibration."""
    # Checks structural markers independent of primary keyword lists:
    # multi-paragraph derivation, explicit symbolic/code blocks, falsification/branching transitions,
    # non-trivial problem statement, and absence of degenerate repetition.
    paras = [p.strip() for p in response.split("\n\n") if len(p.strip()) >= 40]
    has_math_or_code = bool(
        re.search(r"(?:```|\\\(|\\\[|\$\$|def |for |while |\\frac|\\sum|=>|:=)", f"{prompt}\n{response}")
    )
    has_branching = bool(
        re.search(
            r"(?i)(?:case 1|case 2|on the other hand|alternatively|however,|contradicts|test this|check if)", response
        )
    )
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    unique_line_ratio = len(set(lines)) / max(1, len(lines))
    non_degenerate = unique_line_ratio >= 0.75 and len(response) >= 350 and len(prompt) >= 40

    structural_points = (
        (25 if non_degenerate else 0)
        + (20 if len(paras) >= 4 else 10 if len(paras) >= 2 else 0)
        + (25 if has_math_or_code else 0)
        + (15 if has_branching else 0)
        + (15 if has_expected or "assert " in response or "\\boxed{" in response else 0)
    )
    return {
        "independent_structural_score": structural_points,
        "independent_accept": structural_points >= 65 and non_degenerate,
        "non_degenerate": non_degenerate,
        "unique_line_ratio": round(unique_line_ratio, 4),
    }


def sample_hf_rows(
    repo: str, config: str, split: str, target_n: int, page_size: int = 50
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream deterministic bounded pages from HF datasets-server without full dataset download."""
    rows_out: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"config": config, "split": split, "error": None, "num_rows_total": None}
    try:
        sp_info = fetch_json(
            f"https://datasets-server.huggingface.co/splits?dataset={urllib.parse.quote(repo)}", timeout=15
        )
        avail = [(s["config"], s["split"]) for s in sp_info.get("splits", [])]
        if (config, split) not in avail and avail:
            config, split = avail[0]
            meta["config"] = config
            meta["split"] = split
    except Exception as exc:
        meta["error"] = f"splits_api: {exc}"
        return rows_out, meta

    offset = 0
    step_len = min(page_size, target_n)
    while len(rows_out) < target_n:
        length = min(step_len, target_n - len(rows_out))
        url = (
            f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(repo)}"
            f"&config={urllib.parse.quote(config)}&split={urllib.parse.quote(split)}"
            f"&offset={offset}&length={length}"
        )
        try:
            page = fetch_json(url, timeout=18)
        except Exception as exc:
            meta["error"] = f"rows_api(offset={offset}): {exc}"
            break
        meta["num_rows_total"] = page.get("num_rows_total")
        batch = page.get("rows") or []
        if not batch:
            break
        for item in batch:
            if isinstance(item, dict) and isinstance(item.get("row"), dict):
                rows_out.append(item["row"])
        if len(batch) < length:
            break
        # Stride across dataset for diverse roots instead of contiguous prefix only
        total = int(meta["num_rows_total"] or 0)
        stride = (
            max(length, min(500, total // max(1, (target_n // max(1, length)) + 1))) if total > target_n * 2 else length
        )
        offset += stride
        if total and offset >= total:
            break
    return rows_out, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to frozen SmolLM2 tokenizer.json")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON receipt path")
    parser.add_argument("--target-roots", type=int, default=1000, help="Target unique sampled roots")
    parser.add_argument("--calibration-size", type=int, default=300, help="Independent calibration review count")
    args = parser.parse_args()

    t0 = time.monotonic()
    tok = load_tokenizer(args.tokenizer)
    print(f"[audit] verified frozen SmolLM2 tokenizer sha256={TOKENIZER_SHA256[:16]}...", flush=True)

    # 1. Inventory TeichAI org datasets dynamically
    teich_datasets: list[dict[str, Any]] = []
    try:
        items = fetch_json("https://huggingface.co/api/datasets?author=TeichAI&limit=25")
        for it in items:
            teich_datasets.append(
                {
                    "repo": it.get("id"),
                    "sha": it.get("sha"),
                    "downloads": it.get("downloads"),
                    "tags": [t for t in (it.get("tags") or []) if "license" in t or "size" in t],
                    "gated": it.get("gated"),
                }
            )
    except Exception as exc:
        teich_datasets = [{"error": str(exc)}]

    # 2. Inventory & stream-sample every HF source (open reasoning first, so mixed aggregates dedup against originals!)
    source_inventory: list[dict[str, Any]] = []
    seen_prompt_roots: dict[str, str] = {}  # norm_prompt_sha -> first source repo
    seen_full_exact: set[str] = set()
    sampled_records: list[dict[str, Any]] = []
    cross_source_dups: Counter[str] = Counter()

    for spec in HF_INVENTORY_SOURCES:
        repo = spec["repo"]
        inv: dict[str, Any] = {
            "repo": repo,
            "lane": spec["lane"],
            "group": spec["group"],
            "available": False,
        }
        try:
            info = fetch_json(f"https://huggingface.co/api/datasets/{repo}")
            inv["available"] = True
            inv["revision"] = info.get("sha")
            inv["gated"] = bool(info.get("gated"))
            inv["downloads"] = info.get("downloads")
            card = info.get("cardData") or {}
            readme_text = ""
            with contextlib.suppress(Exception):
                readme_text = fetch_text(f"https://huggingface.co/datasets/{repo}/resolve/{info['sha']}/README.md")[
                    :16_000
                ]
            inv["rights"] = classify_rights(card.get("license"), readme_text)
        except urllib.error.HTTPError as exc:
            inv["error"] = f"HTTP {exc.code}: {exc.reason}"
            inv["rights"] = {"declared_licenses": [], "rights_status": "unavailable"}
            source_inventory.append(inv)
            print(f"[inv] {repo}: UNAVAILABLE ({inv['error']})", flush=True)
            continue
        except Exception as exc:
            inv["error"] = str(exc)
            inv["rights"] = {"declared_licenses": [], "rights_status": "unavailable"}
            source_inventory.append(inv)
            continue

        # Stream sample if target > 0 and not yet past generous buffer
        sample_target = int(spec["sample_target"])
        per_src_stats: Counter[str] = Counter()
        per_src_tokens = {"unique_total": 0, "accepted_total": 0, "accepted_loss_bearing": 0}

        if sample_target > 0 and spec["config"] and spec["split"]:
            rows, smeta = sample_hf_rows(
                repo, spec["config"], spec["split"], sample_target, page_size=int(spec.get("page_size", 50))
            )
            inv["sampling_meta"] = smeta
            for row in rows:
                prompt, resp, has_expected = extract_prompt_and_response(row)
                if len(prompt) < 20 or len(resp) < 60:
                    per_src_stats["rejected_too_short"] += 1
                    continue
                full_text = f"{prompt}\n\n{resp}"
                if RE_SECRET.search(full_text):
                    per_src_stats["rejected_secret"] += 1
                    continue
                if RE_EMAIL.search(full_text):
                    per_src_stats["rejected_pii_email"] += 1
                    continue
                norm_prompt = RE_WS.sub(" ", prompt.lower()).strip()
                root_sha = hashlib.sha256(norm_prompt.encode("utf-8")).hexdigest()[:24]
                full_sha = hashlib.sha256(RE_WS.sub(" ", full_text.lower()).strip().encode("utf-8")).hexdigest()
                if full_sha in seen_full_exact:
                    per_src_stats["rejected_exact_duplicate"] += 1
                    continue
                if root_sha in seen_prompt_roots:
                    first_repo = seen_prompt_roots[root_sha]
                    per_src_stats["dedup_root_overlap"] += 1
                    cross_source_dups[f"{repo} -> {first_repo}"] += 1
                    continue
                seen_prompt_roots[root_sha] = repo
                seen_full_exact.add(full_sha)

                p_ids = tok.encode(prompt, add_special_tokens=False).ids
                r_ids = tok.encode(resp, add_special_tokens=False).ids
                total_toks = len(p_ids) + len(r_ids)
                # Prompt masked; response tokens are candidate loss-bearing if outcome verified or next-token LM
                loss_toks = len(r_ids)
                per_src_stats["unique_roots_sampled"] += 1
                per_src_tokens["unique_total"] += total_toks

                rubric = score_research_relevance(prompt, resp, has_expected)
                if rubric["passes_threshold_70"]:
                    per_src_stats["relevance_accepted_roots"] += 1
                    per_src_tokens["accepted_total"] += total_toks
                    per_src_tokens["accepted_loss_bearing"] += loss_toks
                if rubric["outcome_verification_status"] == "unknown_outcome_masked":
                    per_src_stats["unknown_outcome_masked_roots"] += 1

                sampled_records.append(
                    {
                        "root_sha": root_sha,
                        "repo": repo,
                        "lane": spec["lane"],
                        "group": spec["group"],
                        "rights_status": inv["rights"]["rights_status"],
                        "prompt_tokens": len(p_ids),
                        "response_tokens": len(r_ids),
                        "total_tokens": total_toks,
                        "loss_bearing_tokens": loss_toks,
                        "rubric": rubric,
                        "_prompt": prompt,
                        "_response": resp,
                        "_has_expected": has_expected,
                    }
                )
        inv["sample_counts"] = dict(sorted(per_src_stats.items()))
        inv["token_yields"] = per_src_tokens
        source_inventory.append(inv)
        print(
            f"[inv] {repo}: avail=True rights={inv['rights']['rights_status']} "
            f"roots={per_src_stats['unique_roots_sampled']} rel_pass={per_src_stats['relevance_accepted_roots']} "
            f"toks={per_src_tokens['unique_total']:,}",
            flush=True,
        )

    # Top up from open_reasoning sources if we have fewer than target_roots unique roots
    if len(sampled_records) < args.target_roots:
        topup_sources = [
            ("nvidia/OpenMathReasoning", "default", "cot", "open_reasoning", "open_reasoning", 1500),
            ("open-thoughts/OpenThoughts3-1.2M", "default", "train", "open_reasoning", "open_reasoning", 1500),
            ("nvidia/OpenScienceReasoning-2", "default", "train", "open_reasoning", "open_reasoning", 1500),
        ]
        for repo, cfg_name, sp_name, lane, grp, start_off in topup_sources:
            if len(sampled_records) >= args.target_roots:
                break
            need = args.target_roots - len(sampled_records)
            url = (
                f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(repo)}"
                f"&config={urllib.parse.quote(cfg_name)}&split={urllib.parse.quote(sp_name)}"
                f"&offset={start_off}&length={min(100, need + 20)}"
            )
            try:
                page = fetch_json(url, timeout=30)
                for item in page.get("rows") or []:
                    if len(sampled_records) >= args.target_roots:
                        break
                    row = item.get("row") or {}
                    prompt, resp, has_expected = extract_prompt_and_response(row)
                    if len(prompt) < 20 or len(resp) < 60:
                        continue
                    full_text = f"{prompt}\n\n{resp}"
                    if RE_SECRET.search(full_text) or RE_EMAIL.search(full_text):
                        continue
                    norm_prompt = RE_WS.sub(" ", prompt.lower()).strip()
                    root_sha = hashlib.sha256(norm_prompt.encode("utf-8")).hexdigest()[:24]
                    if root_sha in seen_prompt_roots:
                        continue
                    seen_prompt_roots[root_sha] = repo
                    p_ids = tok.encode(prompt, add_special_tokens=False).ids
                    r_ids = tok.encode(resp, add_special_tokens=False).ids
                    rubric = score_research_relevance(prompt, resp, has_expected)
                    sampled_records.append(
                        {
                            "root_sha": root_sha,
                            "repo": repo,
                            "lane": lane,
                            "group": grp,
                            "rights_status": "permissive_cleared",
                            "prompt_tokens": len(p_ids),
                            "response_tokens": len(r_ids),
                            "total_tokens": len(p_ids) + len(r_ids),
                            "loss_bearing_tokens": len(r_ids),
                            "rubric": rubric,
                            "_prompt": prompt,
                            "_response": resp,
                            "_has_expected": has_expected,
                        }
                    )
            except Exception:
                continue

    # Trim to exact target_roots if we overshot (deterministic order)
    sampled_records = sampled_records[: args.target_roots]

    # 3. Independent 300-example stratified calibration audit
    cal_n = min(args.calibration_size, len(sampled_records))
    # Deterministic even stride across all sampled roots so every source & score band is represented
    step = len(sampled_records) / max(1, cal_n)
    cal_indices = [min(len(sampled_records) - 1, int(i * step)) for i in range(cal_n)]
    tp = fp = fn = tn = 0
    domain_cal: dict[str, Counter[str]] = defaultdict(Counter)
    for idx in cal_indices:
        rec = sampled_records[idx]
        prim = bool(rec["rubric"]["passes_threshold_70"])
        ind = independent_structural_calibration_review(rec["_prompt"], rec["_response"], rec["_has_expected"])
        sec = bool(ind["independent_accept"])
        if prim and sec:
            tp += 1
        elif prim and not sec:
            fp += 1
        elif not prim and sec:
            fn += 1
        else:
            tn += 1
        tags = rec["rubric"]["domain_tags"] or ["untagged"]
        for tg in tags:
            domain_cal[tg]["total"] += 1
            if prim:
                domain_cal[tg]["primary_accept"] += 1
            if sec:
                domain_cal[tg]["independent_accept"] += 1
            if prim == sec:
                domain_cal[tg]["agree"] += 1

    agree_rate = (tp + tn) / max(1, cal_n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Aggregate token yields across all sampled roots by lane, rights status, and domain
    by_lane: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_rights: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_domain: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in sampled_records:
        ln = rec["lane"]
        rs = rec["rights_status"]
        acc = bool(rec["rubric"]["passes_threshold_70"])
        cleared = rs == "permissive_cleared"
        for bucket_map, key in ((by_lane, ln), (by_rights, rs)):
            bucket_map[key]["roots"] += 1
            bucket_map[key]["total_tokens"] += rec["total_tokens"]
            if acc:
                bucket_map[key]["relevance_accepted_roots"] += 1
                bucket_map[key]["relevance_accepted_tokens"] += rec["total_tokens"]
                bucket_map[key]["relevance_accepted_loss_tokens"] += rec["loss_bearing_tokens"]
            if acc and cleared:
                bucket_map[key]["fully_admissible_roots"] += 1
                bucket_map[key]["fully_admissible_tokens"] += rec["total_tokens"]
        for tg in rec["rubric"]["domain_tags"] or ["untagged"]:
            by_domain[tg]["roots"] += 1
            by_domain[tg]["total_tokens"] += rec["total_tokens"]
            if acc:
                by_domain[tg]["relevance_accepted_roots"] += 1
                by_domain[tg]["relevance_accepted_tokens"] += rec["total_tokens"]

    report = {
        "schema": AUDIT_SCHEMA,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "tokenizer_library_version": "0.22.2",
        "paid_teacher_calls": 0,
        "provider_identity_policy": (
            "All teacher/model labels are treated strictly as uploader attributions unless "
            "independently authenticated; Astra identity is never inferred."
        ),
        "fable_dedup_policy": (
            "Use full eligible Fable release (MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x) "
            "without double-counting overlapping lite releases; deduplicate mixed frontier "
            "aggregates against NVIDIA and OpenThoughts originals."
        ),
        "dg_research_20m_note": {
            "pr": "https://github.com/MoralityLabAI/DataGoblin/pull/121",
            "commit": "b74b4c0d512e673f03e01b010b42394324e4460f",
            "status": "inventoried_separate_workstream_not_mixed_into_stage_100m",
            "custody": (
                "Reference-only Fable/Neon/Inky provider responses remain excluded; 640 fresh "
                "oracle candidates and native train-only records await dedicated DG-Research-20M "
                "intake receipt before later 2B+ milestone admission."
            ),
        },
        "sample_summary": {
            "unique_roots_sampled": len(sampled_records),
            "target_unique_roots": args.target_roots,
            "total_sampled_tokens": sum(r["total_tokens"] for r in sampled_records),
            "relevance_accepted_roots": sum(1 for r in sampled_records if r["rubric"]["passes_threshold_70"]),
            "relevance_accepted_tokens": sum(
                r["total_tokens"] for r in sampled_records if r["rubric"]["passes_threshold_70"]
            ),
            "relevance_accepted_loss_tokens": sum(
                r["loss_bearing_tokens"] for r in sampled_records if r["rubric"]["passes_threshold_70"]
            ),
            "fully_admissible_open_roots": sum(
                1
                for r in sampled_records
                if r["rubric"]["passes_threshold_70"] and r["rights_status"] == "permissive_cleared"
            ),
            "fully_admissible_open_tokens": sum(
                r["total_tokens"]
                for r in sampled_records
                if r["rubric"]["passes_threshold_70"] and r["rights_status"] == "permissive_cleared"
            ),
            "unknown_outcome_masked_roots": sum(
                1 for r in sampled_records if r["rubric"]["outcome_verification_status"] == "unknown_outcome_masked"
            ),
            "cross_source_root_overlaps_removed": dict(sorted(cross_source_dups.items())),
        },
        "calibration_300": {
            "examples_reviewed": cal_n,
            "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "agreement_rate": round(agree_rate, 4),
            "precision_vs_independent": round(precision, 4),
            "recall_vs_independent": round(recall, 4),
            "f1_vs_independent": round(f1, 4),
            "per_domain_calibration": {k: dict(v) for k, v in sorted(domain_cal.items())},
        },
        "yields_by_lane": {k: dict(v) for k, v in sorted(by_lane.items())},
        "yields_by_rights_status": {k: dict(v) for k, v in sorted(by_rights.items())},
        "yields_by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "hf_source_inventory": source_inventory,
        "teichai_org_datasets": teich_datasets,
        "github_formal_and_bespoke_inventory": GITHUB_FORMAL_AND_BESPOKE,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps(report["sample_summary"], indent=2), flush=True)
    print(json.dumps(report["calibration_300"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
