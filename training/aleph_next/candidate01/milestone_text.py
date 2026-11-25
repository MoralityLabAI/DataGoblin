"""Build or reproduce the distinct train-only general-text increment for the 10M milestone."""

import argparse
import gzip
import json
import re
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from urllib.parse import urlsplit

import text_intake as old
from confirm_text import MATCH, body_bytes, folded, native_bytes, panel_data, pins
from milestone_text_overlap import Overlap, duplicate_signature, symbols
from packing import PackedStream
from production import HERE, digest, load_tokenizer, read

GAP = 841096
MARGIN = 2048
AUTHORS = {
    "human": (
        "Deborah Gustlin & Zoe Gustlin",
        "Ann Smart Martin",
        "Todd Smith",
        "Nick Garcia",
        "Jeffery L. Johnson",
        "Athanasios Michaels",
        "Jennifer Marra Henrigillis & Steven Gimbel",
        "Philip S. Peek",
        "Peter L. Smith",
    ),
    "socialsci": (
        "Lynn Meade",
        "Josh Misner and Geoff Carr",
        "David Dorrel & Joseph P. Henderson",
        "Ingrid Robeyns",
        "Richard Garlikov",
        "Mark D. Kelland",
        "Laura Overstreet",
    ),
    "biz": (
        "Nina Burokas",
        "Karen Calendo",
        "Kenneth S. Bigel",
        "Lindon Robison, Steven Hanson, and J. Roy Black",
        "Elisabeth Dellegrazie",
        "Jamie Hammond and Western Technical College",
        "Elizabeth Pulos and Guillermo C. Jimenez",
    ),
    "geo": ("Steven Earle", "Jeremy Patrich", "Paul Webb", "Anders Knudby", "Anne Willem Omta"),
}
BOOK_CAP = 90000
AUTHOR_CAP = 110000
AUTHORS["human"] += (
    "Rajiv Ranjan",
    "Valery Vino",
    "Christopher R Boltz",
    "Walter D. Butler; Aloha Sargent; and Kelsey Smith",
    "Brittany Zemlick",
    "Robert Lloyd, Daniel Mertens, and Matthew Vollrath",
    "Nathan Nobis & Kristina Grob",
    "Elizabeth Burns Coleman",
    "Liza Long, Amy Minervini, and Joel Gladd",
    "Ayman Mohamed and Sadam Issa",
    "Andrew Gurevich",
    "Dan Stone",
    "Xiao Ouyang",
    "Ruth Sonderegger & Ines Kleesattel",
    "Matthew Sharpe",
    "Alexander Westenberg",
    "Laurie M. Apple",
    "Matteo Ravasio",
    "Joel Gladd",
    "Pierre Fasula",
    "Nathan Smith",
    (
        "Beau Branson, Marcus William Hunt, Timothy D. Knepper, Robert Sloan Lee, Steven Steyl, "
        "Hans Van Eyghen, Beau Branson, and Christina Hendricks"
    ),
)
AUTHORS["socialsci"] += (
    "Steven J Luck",
    "Mark Rozen Pettinelli",
    "Gina Peterson and Emily Elam",
    "Katie Coleman",
    "Sheryl Shook",
    "Kelly L. Reddy-Best, Abbey K Elder, and Lesya Hassall",
    "Janet Stramel",
    "Rebecca Laff and Wendy Ruiz",
    "Cindy Stephens, Gina Peterson, Sharon Eyrich, & Jennifer Paris",
    "Ina Goel",
    "Diana Daly",
    "Alise Lamoreaux",
    "Miliann Kang, Donovan Lessard, Laura Heston, and Sonny Nordmarken",
    "Mark Poepsel",
    "Nathan Dawthorne",
    "Nadine T. Fernandez & Katie Nelson",
    "S. Ashley Kistler",
    "Elizabeth B. Pearce",
    "Krischa Esquivel, Emily Elam, Jennifer Paris, & Maricela Tafoya",
    "Susan Rahman, Prateek Sunder, and Dahmitra Jackson",
)
AUTHORS["biz"] += ("Glenn Arnold & Suzanne Kyle", "Robert Danielson", "Susan Kendall")
AUTHORS["geo"] += ("Colby Moorberg & David Crouse",)
AUTHORS["workforce"] = ("BC Cook Articulation Committee", "Wally Baumback", "Kimberly Mosley", "Stephanie Anagnoson")
AUTHORS["phys"] = (
    "Edwin F. Taylor & John Archibald Wheeler",
    "Lloyd Knox",
    "Ryan Cooke",
    "Ethan Nadler",
    "Johan Wevers",
)
SEPARATOR = re.compile(r'(?<=[.!?])\s+(?=[\u201c"\u2018(A-Z0-9])|\n+')
SENTENCE_END = re.compile(r'[.!?](?:[\u201d\u2019"]|\s|$)')
LICENSE_FILES = {
    "CC-BY-4.0.txt": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
    "Apache-2.0.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "TOKENIZER-NOTICE.md": "f8ac682440e62337482b033d76d46bbe7bb4333909bcde5b5183f5ff5223c6c0",
}
MIXED_RIGHTS = re.compile(
    r"all rights reserved|used (?:by|with) permission|except where otherwise noted|non.?commercial|no.?derivatives|"
    r"copyright(?:ed)? (?:by|of)|©|courtesy of|reprinted|reproduced (?:by|with|from)|photo(?:graph)? (?:by|credit)|"
    r"cc[ -]?by[ -]?(?:nc|sa|nd)|creativecommons[.]org/licenses/(?!by/4[.]0)|cc by [123](?:[.]0)?\b|"
    r"openstax|lumen learning",
    re.I,
)


def canonical(value):
    return " ".join(old.words(value))


def keys(row):
    m = row["metadata"]
    a = m["author"].strip()
    group = set(old.family_keys(row))
    for part in re.split(r"\s*(?:,|&|;|\band\b)\s*", a, flags=re.I):
        value = canonical(part)
        if len(value.split()) >= 2:
            group.add("person:" + value)
    return group


class Groups:
    def __init__(self):
        self.parents = {}

    def find(self, x):
        p = self.parents
        p.setdefault(x, x)
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def connect(self, ks):
        roots = sorted({self.find(k) for k in ks})
        for r in roots[1:]:
            self.parents[r] = roots[0]


def domain(row):
    host = urlsplit(row["id"]).hostname or ""
    return host.split(".")[0] if host.endswith(".libretexts.org") else ""


def metadata_ok(r):
    m = r.get("metadata", {})
    return (
        m.get("url") == r["id"]
        and str(m.get("book_url", "")).startswith("https://commons.libretexts.org/book/")
        and all(isinstance(m.get(k), str) and m[k].strip() for k in ("author", "title", "provenance"))
        and bool(old.family_keys(r)[2][6:])
    )


def frozen_prose(base):
    supply = read(HERE / "SUPPLY.json")
    audits = {}
    by_url = {}
    for role in old.ROLES:
        folder = base / "prose" / role
        p = folder / "manifest.json"
        m = read(p)
        if digest(p) != supply["text_roles"][role]["manifest_sha256"]:
            raise ValueError("frozen prose manifest changed")
        attr = folder / m["attribution"]["path"]
        if digest(attr) != m["attribution"]["sha256"]:
            raise ValueError("frozen attribution changed")
        audits[role] = digest(p)
        for line in attr.read_bytes().splitlines():
            row = json.loads(line)
            if row["split"] != role or row["original_url"] in by_url:
                raise ValueError("non-disjoint frozen role")
            by_url[row["original_url"]] = row
    return by_url, audits


def read_candidates(base, source, overlap):
    prior, prior_stamps = frozen_prose(base)
    groups = Groups()
    block = set()
    identified = set()
    candidates = []
    reject = Counter()
    n = 0
    with gzip.open(source, "rt", encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            n += 1
            subject = domain(row)
            m = row["metadata"]
            if row["id"] in prior:
                p = prior[row["id"]]
                if (
                    p["text_sha256"] != old.sha(row["text"].encode())
                    or p["author"] != m["author"]
                    or p["book_url"] != m["book_url"]
                ):
                    raise ValueError("original frozen prose differs from source")
                if row["id"] not in identified:
                    overlap.add(row["text"])
                    identified.add(row["id"])
            if subject == "math" and old.eligibility(row) is None:
                ks = keys(row)
                groups.connect(ks)
                block.update(ks)
            if (
                subject not in AUTHORS
                or not urlsplit(row["id"]).path.startswith("/Bookshelves/")
                or not metadata_ok(row)
            ):
                continue
            ks = keys(row)
            groups.connect(ks)
            if m["author"].strip() not in AUTHORS[subject]:
                continue
            if m.get("license") != old.LICENSE:
                reject["source_license"] += 1
                continue
            if MIXED_RIGHTS.search(row["text"]):
                reject["mixed_or_third_party_terms"] += 1
                continue
            if not 850 <= len(row["text"].encode()) <= 50000 or len(old.words(row["text"])) < 120:
                reject["length"] += 1
                continue
            candidates.append(row)
    if identified != set(prior):
        raise ValueError("missing frozen source documents")
    for doc in pins():
        surname, given = doc["author"].split(",", 1)
        primary = canonical(re.sub(r"\([^)]*\)", "", given) + " " + surname)
        block.update(("by:" + primary, "person:" + primary, "title:" + canonical(doc["title"])))
    blocked = {groups.find(k) for k in block}
    eligible = []
    for r in candidates:
        family = groups.find(next(iter(keys(r))))
        if family in blocked:
            reject["frozen_author_book_component"] += 1
        else:
            eligible.append((r, family))
    return eligible, prior_stamps, reject, n, len(prior)


def baseline(args):
    overlap = Overlap()
    documents = pins()
    for doc in documents:
        prose, _ = body_bytes(doc, args.confirm / "source-cache")
        overlap.add(prose.decode())
    _, held = panel_data(args.confirm / "text", old.PANEL)
    windows = [bytes.fromhex(x["hex"]) for x in held]
    for sources in native_bytes(args.native).values():
        windows.extend(sources)
    tiny = {b[i : i + MATCH] for s in windows for b in (s, folded(s)) for i in range(len(b) - MATCH + 1)}
    return overlap, tiny, len(documents), len(windows)


def short_match(text, tiny):
    raw = text.encode()
    return any(part[i : i + MATCH] in tiny for part in (raw, folded(raw)) for i in range(len(part) - MATCH + 1))


def passages(text, tiny):
    if not short_match(text, tiny):
        return [(0, len(text))]
    spans = []
    left = None
    end = 0
    start = 0
    for stop in [m.end() for m in SEPARATOR.finditer(text)] + [len(text)]:
        if start == stop:
            continue
        unit = text[start:stop]
        if short_match(unit, tiny):
            if left is not None:
                spans.append((left, end))
            left = None
        elif left is None:
            left = start
        elif short_match(text[max(left, start - 100) : min(stop, start + 100)], tiny):
            spans.append((left, end))
            left = start
        end = stop
        start = stop
    if left is not None:
        spans.append((left, end))
    cleaned = []
    for a, b in spans:
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        segment = text[a:b]
        if (
            len(old.words(segment)) >= 145
            and len(SENTENCE_END.findall(segment)) >= 3
            and not short_match(segment, tiny)
        ):
            cleaned.append((a, b))
    return sorted(cleaned, key=lambda x: (x[0] - x[1], x[0]))


def select(args):
    source = args.base / "libretexts-checked.json.gz"
    if digest(source) != old.SOURCE:
        raise ValueError("upstream source checksum mismatch")
    tok = load_tokenizer(args.base / "tokenizer.json")
    overlap, tiny, full_books, window_count = baseline(args)
    candidates, previous, reject, total, prior_docs = read_candidates(args.base, source, overlap)
    queues = defaultdict(list)
    for row, family in candidates:
        queues[domain(row), row["metadata"]["author"].strip()].append((row, family))
    queues = {
        a: deque(sorted(items, key=lambda x: (old.sha(x[0]["id"].encode()), x[0]["id"]))) for a, items in queues.items()
    }
    used = Counter()
    subject_tokens = Counter()
    author_tokens = Counter()
    book_tokens = Counter()
    seen_ids = set()
    rows = []
    processed = 0
    while queues and processed < GAP + MARGIN:
        subject, author = min(queues, key=lambda k: (subject_tokens[k[0]], author_tokens[k[1]], k))
        q = queues[subject, author]
        row, family = q.popleft()
        if not q:
            del queues[subject, author]
        m = row["metadata"]
        book = m["book_url"]
        if row["id"] in seen_ids:
            reject["duplicate_url"] += 1
            continue
        if author_tokens[author] >= AUTHOR_CAP or book_tokens[book] >= BOOK_CAP:
            reject["author_or_book_cap"] += 1
            continue
        original = row["text"]
        options = passages(original, tiny)
        reason = "no_intact_multisentence_passage"
        for a, b in options:
            section = original[a:b]
            if len(SENTENCE_END.findall(section)) < 3:
                continue
            norm, sketch = duplicate_signature(section)
            syms = symbols(section)
            if why := overlap.check(norm, sketch, syms):
                reason = why
                continue
            ids = tok.encode(section, add_special_tokens=False).ids
            if len(ids) + author_tokens[author] > AUTHOR_CAP or len(ids) + book_tokens[book] > BOOK_CAP:
                reason = "author_or_book_cap"
                continue
            break
        else:
            reject[reason] += 1
            continue
        before = len(original[:a].encode())
        citation = {
            "source_text_sha256": old.sha(original.encode()),
            "source_start_byte": before,
            "source_end_byte": before + len(section.encode()),
            "source_text_bytes": len(original.encode()),
        }
        rows.append(
            ({**row, "text": section, "citation": citation}, "libretexts/milestone-general/" + family, ids, norm)
        )
        seen_ids.add(row["id"])
        overlap.add_prepared(norm, sketch, syms)
        used[subject] += 1
        book_tokens[book] += len(ids)
        author_tokens[author] += len(ids)
        subject_tokens[subject] += len(ids)
        processed += len(ids)
    if not rows:
        raise ValueError("no independent general-text rows passed")
    report = {
        "schema": "datagoblin.milestone.general-admission.v1",
        "split": "train",
        "lane": "general",
        "objective": "next_token",
        "positions_requested": GAP,
        "distinct_positions": processed,
        "covers_gap": processed >= GAP,
        "presentations_per_root": 1,
        "source": {
            "dataset": old.REPO,
            "revision": old.REV,
            "sha256": old.SOURCE,
            "records": total,
            "license": old.LICENSE,
        },
        "historical": {
            "prose_manifests": previous,
            "prior_prose_documents": prior_docs,
            "confirm14_panel_sha256": old.PANEL,
            "confirm14_full_books": full_books,
            "small_native_and_confirm14_windows": window_count,
            "byte_match_width": MATCH,
            "folded_word_match_width": 24,
            "near_document": "bottom 32 CRC32 seven-word shingles, overlap >=75% of larger sketch",
        },
        "subject_positions": dict(subject_tokens),
        "subject_records": dict(used),
        "authors": len(author_tokens),
        "books": len(book_tokens),
        "families": len({x[1] for x in rows}),
        "records": len(rows),
        "whole_sections": sum(r["citation"]["source_text_bytes"] == len(r["text"].encode()) for r, _, _, _ in rows),
        "one_passage_per_source_document": True,
        "whole_section_trial": {"records": 223, "positions": 121364, "section_wide_16byte_rejections": 772},
        "candidate_screen_failures": dict(reject),
        "source_ranked_candidates_unvisited": sum(map(len, queues.values())),
        "limits": [
            "English textbook prose from one upstream collection.",
            "Lexical and metadata screening cannot establish that all uncredited source reuse is absent.",
            "Per-section CC BY 4.0 metadata is retained. Embedded third-party notices trigger rejection.",
        ],
    }
    return rows, report


def materialize(args, dest):
    rows, report = select(args)
    old.write_split(dest, "train", args.base / "tokenizer.json", rows)
    folder = dest / "train"
    manifest_path = folder / "manifest.json"
    manifest = read(manifest_path)
    ap = folder / "attribution.jsonl"
    attributes = []
    for line, (row, _, _, _) in zip(ap.read_bytes().splitlines(), rows, strict=True):
        record = json.loads(line)
        record.update(row["citation"])
        record["transformations"] = (
            "One contiguous original UTF-8 source span, tokenized verbatim. Offsets are half-open."
        )
        attributes.append(old.raw(record))
    ap.write_bytes(b"\n".join(attributes) + b"\n")
    manifest["attribution"]["sha256"] = digest(ap)
    manifest["licenses"] = {}
    for name, stamp in LICENSE_FILES.items():
        p = args.base / "operator-package" / "licenses" / name
        if digest(p) != stamp:
            raise ValueError("licensing notice differs from reviewed copy")
        shutil.copyfile(p, folder / name)
        manifest["licenses"][name] = {"path": name, "sha256": stamp}
    manifest_path.write_bytes(old.raw(manifest) + b"\n")
    stamp = digest(manifest_path)
    with PackedStream(manifest_path, stamp, "train") as stream:
        while stream.next_pack(report["distinct_positions"] + 1) is not None:
            pass
        final = stream.checkpoint()["payload"]["state"]
    if final["processed"] != report["distinct_positions"] or final["started_views"] != len(rows):
        raise ValueError("packed stream differs from admitted full documents")
    report["packing"] = {
        k: final[k]
        for k in ("processed", "padding", "targets", "packs", "started_views", "positions_by_lane", "targets_by_lane")
    }
    report["manifest_sha256"] = stamp
    report["attribution_sha256"] = manifest["attribution"]["sha256"]
    (folder / "admission.json").write_bytes(old.raw(report) + b"\n")
    return folder, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("build", "verify", "audit"))
    p.add_argument("--base", type=Path, default=Path("../aleph-candidate01"))
    p.add_argument("--native", type=Path, default=Path("runs/placement/mixed"))
    p.add_argument("--confirm", type=Path, default=Path("../aleph-confirm14"))
    p.add_argument("--out", type=Path, default=Path("../aleph-10m/extra/general"))
    p.add_argument("--audit-output", type=Path)
    args = p.parse_args()
    if args.command == "audit":
        from milestone_text_audit import audit

        result = audit(args.base, args.out)
        if args.audit_output:
            encoded = old.raw(result) + b"\n"
            if args.audit_output.exists() and args.audit_output.read_bytes() != encoded:
                raise ValueError("an existing source audit differs")
            args.audit_output.write_bytes(encoded)
        print(json.dumps(result, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "build" and args.out.exists():
        raise FileExistsError(args.out)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        staging, report = materialize(args, Path(tmp))
        if args.command == "build":
            staging.rename(args.out)
        elif {f.name: digest(f) for f in staging.iterdir()} != {f.name: digest(f) for f in args.out.iterdir()}:
            raise ValueError("independent reconstruction differs from release")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
