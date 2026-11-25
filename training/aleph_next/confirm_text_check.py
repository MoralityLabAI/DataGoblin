"""Audit released A/B window losses, count-model baselines and the exact author bootstrap."""

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (137, 163, 191)
ARMS = ("direct", "global_full_damp")
PINS = Path(__file__).with_name("confirm_text_sources.json")
PANEL_HASH = "82761739ea4f078bd1631dba2851be6551109692fbfe5166295429094fe45c51"


def checked_bytes(file, expected):
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Unexpected digest for {file.name}")
    return data


def average(xs):
    return math.fsum(xs) / len(xs)


def raw_model_nll(rows, fixed):
    """Use the scorer's individual natural-log loss sums and observed supervised-byte counts."""
    groups = defaultdict(list)
    seen = set()
    total = []
    positions = 0
    for r in rows:
        identity = r["root"], r["group"], r["order"]
        if (
            r["lane"] != "text"
            or identity in seen
            or r["positions"] != 32
            or not math.isfinite(r["nll"])
            or r["nll"] < 0
        ):
            raise ValueError("Malformed raw score")
        seen.add(identity)
        total.append(r["nll"])
        positions += r["positions"]
        groups[r["group"]].append(r["nll"])
    if seen != fixed or len(seen) != 384 or positions != 12288:
        raise ValueError("Raw score window coverage differs from the original panel")
    per_doc = {}
    for doc, vals in groups.items():
        if len(vals) != 32:
            raise ValueError("Unequal document coverage")
        per_doc[doc] = math.fsum(vals) / (32 * len(vals))
    return math.fsum(total) / positions, per_doc


def raw_ngram_nll(train, held):
    """Rebuild every conditional probability independently from native training byte counts."""
    frequencies, totals = Counter(), Counter()
    for raw in train:
        for pos in range(1, len(raw)):
            for order in range(min(2, pos) + 1):
                context = raw[pos - order : pos]
                frequencies[context, raw[pos]] += 1
                totals[context] += 1
    doc_sums = defaultdict(lambda: [[], []])
    for row in held:
        window = bytes.fromhex(row["hex"])
        for pos in range(1, len(window)):
            y = window[pos]
            p = (1 + frequencies[b"", y]) / (256 + totals[b""])
            doc_sums[row["doc"]][0].append(-math.log(p))
            for order in range(1, min(2, pos) + 1):
                ctx = window[pos - order : pos]
                p = (8 * p + frequencies[ctx, y]) / (8 + totals[ctx])
            doc_sums[row["doc"]][1].append(-math.log(p))
    return {
        name: {doc: average(sums[j]) for doc, sums in doc_sums.items()}
        for j, name in enumerate(("unigram", "interpolated_trigram"))
    }


def four_author_draws(author_pairs):
    outcomes = []
    for indexes in itertools.combinations_with_replacement(range(4), 4):
        weight = 24 // math.prod(math.factorial(k) for k in Counter(indexes).values())
        a = math.fsum(author_pairs[i][0] for i in indexes)
        b = math.fsum(author_pairs[i][1] for i in indexes)
        outcomes.append((a, b, weight))
    if len(outcomes) != 35 or sum(x[-1] for x in outcomes) != 256:
        raise ValueError("Invalid four-author bootstrap weights")
    return outcomes


def exact_percentiles(values):
    """Type-7 quantiles over a discrete weighted sample without expanding all 4^12 draws."""
    values.sort()
    total = sum(weight for _, weight in values)

    def order_statistic(want):
        scanned = 0
        for val, weight in values:
            scanned += weight
            if want < scanned:
                return val
        raise ValueError("Quantile exceeds cumulative probability mass")

    endpoints = []
    for proportion in (0.025, 0.975):
        index = (total - 1) * proportion
        lo, hi = math.floor(index), math.ceil(index)
        lower = order_statistic(lo)
        endpoints.append(lower + (index - lo) * (order_statistic(hi) - lower))
    return endpoints


def exact_document_bootstrap(a_docs, b_docs, groups):
    """Matched fixed seeds already averaged per author; each stratum has four authors."""
    strata = sorted(set(groups.values()))
    blocks = [[(a_docs[d], b_docs[d]) for d in sorted(groups) if groups[d] == s] for s in strata]
    if len(blocks) != 3 or any(len(b) != 4 for b in blocks):
        raise ValueError("Expected three strata of four authors")
    relative, absolute = [], []
    for triple in itertools.product(*(four_author_draws(block) for block in blocks)):
        a = math.fsum(x[0] for x in triple)
        b = math.fsum(x[1] for x in triple)
        w = math.prod(x[2] for x in triple)
        relative.append((1 - b / a, w))
        absolute.append(((b - a) / 12, w))
    return dict(
        count_vectors=len(relative),
        equivalent_ordered_draws=sum(w for _, w in relative),
        relative_gain_interval_95=exact_percentiles(relative),
        B_minus_A_nats_interval_95=exact_percentiles(absolute),
    )


def run(directory, expected_panel=PANEL_HASH):
    # Require a released joint decision before accessing any held losses or native-count baselines.
    decision = json.loads((directory / "decision.json").read_bytes())
    frozen = json.loads(checked_bytes(directory / "freeze.json", decision["frozen"]["freeze_sha256"]))
    roster = json.loads(checked_bytes(Path(frozen["fit_manifest"]), frozen["fit_manifest_sha256"]))
    if (
        frozen["reference_development_competent"] is not True
        or roster["reference_competent"] is not True
        or frozen["text_manifest_sha256"] != expected_panel
        or decision["frozen"]["held_text_sha256"] != expected_panel
    ):
        raise ValueError("Missing competent reference or wrong original panel commitment")
    p = Path(frozen["text_dir"])
    meta = json.loads(checked_bytes(p / "manifest.json", expected_panel))
    held = [json.loads(x) for x in checked_bytes(p / "windows.jsonl", meta["windows_sha256"]).splitlines()]
    fixed = {(r["root"], r["doc"], r["block"]) for r in held}
    groups = {d["id"]: d["stratum"] for d in json.loads(PINS.read_bytes())}
    if len(groups) != 12 or {r["doc"] for r in held} != set(groups):
        raise ValueError("Document IDs differ from the public edition roster")
    train_name = "text-train.jsonl"
    raw_train = checked_bytes(Path(frozen["pack"]) / train_name, meta["native_sha256"][train_name])
    train = [bytes(json.loads(row)["tokens"]) for row in raw_train.splitlines()]
    controls = {k: average(list(v.values())) for k, v in raw_ngram_nll(train, held).items()}
    published = decision["observed"]
    distances = []
    nll = {arm: {} for arm in ARMS}
    docs = {arm: {} for arm in ARMS}
    capsule_hashes = {}
    for arm, s in itertools.product(ARMS, SEEDS):
        seed = str(s)
        cell = f"{arm}-{s}"
        data = (directory / f"score-{cell}.json").read_bytes()
        cap = json.loads(data)
        capsule_hashes[cell] = hashlib.sha256(data).hexdigest()
        stamp = cap["stamp"]
        if (
            stamp["arm"],
            stamp["seed"],
            stamp["checkpoint_sha256"],
            stamp["freeze_sha256"],
            stamp["text_panel_sha256"],
        ) != (arm, s, frozen["fits"][cell]["sha256"], decision["frozen"]["freeze_sha256"], expected_panel):
            raise ValueError("Wrong checkpoint or panel label in capsule")
        value, doc = raw_model_nll(cap["private_text_windows"], fixed)
        if set(doc) != set(groups):
            raise ValueError("Incomplete author roster")
        nll[arm][seed], docs[arm][seed] = value, doc
        distances.extend(abs(val - cap["text"]["per_document"][d]) for d, val in doc.items())
        distances.extend(
            (abs(value - published["text_nll_by_arm_seed"][arm][seed]), abs(value - cap["text"]["nats_per_byte"]))
        )
    a, b = ARMS
    by_seed = {}
    for s in map(str, SEEDS):
        x, y = nll[a][s], nll[b][s]
        gain = 1 - y / x
        by_seed[s] = dict(A_nats_per_byte=x, B_nats_per_byte=y, B_minus_A_nats=y - x, relative_gain_percent=100 * gain)
        distances.append(abs(gain - published["relative_text_gain_by_seed"][s]))
    means = {arm: average(list(byseed.values())) for arm, byseed in nll.items()}
    docmeans = {arm: {d: average([docs[arm][str(s)][d] for s in SEEDS]) for d in groups} for arm in ARMS}
    for d in groups:
        distances.append(abs(docmeans[a][d] - docmeans[b][d] - decision["text_doc_gains_B_over_A_nats"][d]))
    gain = 1 - means[b] / means[a]
    distances.append(abs(gain - published["relative_text_gain"]))
    distances.extend(abs(v - decision["text_controls"][k + "_nats_per_byte"]) for k, v in controls.items())
    exact = exact_document_bootstrap(docmeans[a], docmeans[b], groups)
    monte_carlo = decision["intervals"]["primary"]["relative_text_gain"]
    return dict(
        schema="datagoblin.confirm14.independent_text.v1",
        panel_sha256=expected_panel,
        fit_manifest_sha256=frozen["fit_manifest_sha256"],
        capsule_sha256=capsule_hashes,
        per_seed=by_seed,
        mean_arm_nats_per_byte=means,
        B_minus_A_mean_nats=means[b] - means[a],
        relative_gain_percent=100 * gain,
        native_train_only_baseline_nats_per_byte=controls,
        max_numeric_discrepancy=max(distances),
        agrees_with_published_points=max(distances) < 1e-12,
        exact_stratified_author_bootstrap=exact,
        prespecified_monte_carlo_relative_interval=monte_carlo,
        exact_minus_monte_carlo_endpoint_percentage_points=[
            100 * (x - y) for x, y in zip(exact["relative_gain_interval_95"], monte_carlo, strict=True)
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.release)
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if args.out.exists():
        if args.out.read_bytes() != data:
            raise ValueError("Refusing to replace a different independent receipt")
    else:
        with args.out.open("xb") as target:
            target.write(data)
    print(json.dumps(result, sort_keys=True))
    if not result["agrees_with_published_points"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
