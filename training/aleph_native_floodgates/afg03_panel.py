"""AFG-03 token panel builder: hash-bound prompts from the pinned SmolLM2 tokenizer.

Thirty-two source documents, eight per category (general prose, Python
code, worked mathematical reasoning, structured research protocol text).
Each document yields two prompt windows that start on line boundaries;
both windows of a document stay in one split. Per category, documents
1-4 are train, 5-6 CAL and 7-8 development, giving 32/16/16 rows.

General and mathematical documents are authored below. Code comes from
the running interpreter's standard library and research text from
public protocol files in this repository; every source records its
LF-normalized SHA-256. The panel and source manifest are written to a
local directory; the repository keeps only hashes and the inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import afg03_checkpoint as A  # noqa: E402

SPLIT_OF = {1: "train", 2: "train", 3: "train", 4: "train", 5: "cal", 6: "cal", 7: "dev", 8: "dev"}
SEED = "aleph.afg03.panel.v1"
TOKEN_CHECK = ("The capital of France is", [504, 3575, 282, 4649, 314])

GENERAL = [
    "Every Saturday the market on the river road opens before sunrise. Farmers unload crates of apples, "
    "leeks and late tomatoes while the bakers set out loaves that are still warm from the ovens.\n"
    "By nine o'clock the square is crowded with families. Children trade stickers by the fountain, and an "
    "old man plays the accordion for coins beside the flower stall until the church bell rings noon.\n",
    "The storm arrived later than the forecast promised, but it stayed longer than anyone expected. Rain "
    "fell in heavy sheets for most of the night and the gutters along the hill overflowed by midnight.\n"
    "In the morning the streets were covered with leaves and broken branches. Neighbors came out with "
    "brooms and saws, and by lunch most of the road to the school had been cleared again.\n",
    "Dear Maria, thank you for the postcard from the coast. It is pinned above my desk, and every time I "
    "look at the lighthouse I remember the summer we spent painting the fence at your aunt's house.\n"
    "Work has been busy here, but the evenings are getting longer and I have started walking after dinner. "
    "Write back soon and tell me whether you still plan to visit in the spring.\n",
    "The town library occupies an old brick building that once served as a post office. The reading room "
    "has tall windows facing east, so the morning light falls across the long oak tables.\n"
    "On Tuesday evenings volunteers run a reading circle for adults who are learning English. The group "
    "has grown from four members to nearly thirty, and the librarians now set out extra chairs.\n",
    "The night train left the capital a few minutes after ten and headed north through the farmland. Most "
    "passengers fell asleep quickly, but I sat by the window and watched the lights of small villages.\n"
    "Around dawn the tracks began to climb into the mountains. Snow appeared on the pine trees, and the "
    "conductor walked through the cars announcing that breakfast was being served in the dining car.\n",
    "Tomatoes grow best in a sunny spot that receives at least six hours of direct light each day. Water "
    "them deeply at the base of the plant rather than on the leaves, which helps prevent disease.\n"
    "When the first flowers appear, add a thin layer of mulch to keep the soil moist and cool. Tie the "
    "stems loosely to a stake so that the weight of the fruit does not bend or break them.\n",
    "The stone bridge over the valley was finished in the spring of 1862 after four years of work. Its "
    "builders hauled granite from a quarry twenty miles away using carts pulled by teams of oxen.\n"
    "For more than a century it carried farm wagons, then cars, and finally only walkers and cyclists. "
    "Today a small plaque near the eastern arch lists the names of the masons who built it.\n",
    "To make a simple vegetable soup, start by chopping two onions, three carrots and a few stalks of "
    "celery. Warm some olive oil in a large pot and cook the vegetables slowly until they soften.\n"
    "Add a can of tomatoes, a handful of lentils and enough water to cover everything by a few inches. "
    "Let the soup simmer for forty minutes, then season it with salt, pepper and fresh parsley.\n",
]

MATH = [
    "Solve the equation 3x + 5 = 20. First subtract 5 from both sides, which gives 3x = 15. Then divide "
    "both sides by 3 to isolate the variable, so x = 5.\n"
    "Check the answer by substitution: 3 times 5 is 15, and 15 plus 5 is 20. The left side equals the "
    "right side, so the solution x = 5 is correct.\n",
    "Find the sum of the integers from 1 to 100. Pair the first and last terms: 1 + 100 = 101, then "
    "2 + 99 = 101, and so on. There are 50 such pairs, each adding to 101.\n"
    "Therefore the total is 50 times 101, which equals 5050. In general the sum of the first n positive "
    "integers is n(n + 1)/2, and for n = 100 this formula also gives 5050.\n",
    "Two fair six-sided dice are rolled. What is the probability that the sum is 7? There are 6 times 6, "
    "or 36, equally likely outcomes in total.\n"
    "The pairs that add to 7 are (1,6), (2,5), (3,4), (4,3), (5,2) and (6,1), which is 6 outcomes. So "
    "the probability is 6/36, and this fraction simplifies to 1/6.\n",
    "A right triangle has legs of length 6 and 8. To find the hypotenuse c, apply the Pythagorean theorem: "
    "c squared equals 6 squared plus 8 squared.\n"
    "That is c^2 = 36 + 64 = 100. Taking the positive square root gives c = 10. The side lengths 6, 8 "
    "and 10 are a scaled version of the familiar 3, 4, 5 right triangle.\n",
    "A jacket costs 80 dollars and is on sale for 25 percent off. The discount is 25 percent of 80, "
    "which is 0.25 times 80, or 20 dollars.\n"
    "Subtracting the discount from the original price gives 80 - 20 = 60 dollars. Equivalently, the sale "
    "price is 75 percent of the original, and 0.75 times 80 is also 60.\n",
    "Find the greatest common divisor of 84 and 126. Factor each number into primes: 84 = 2^2 * 3 * 7 "
    "and 126 = 2 * 3^2 * 7.\n"
    "The common prime factors with the smallest exponents are 2, 3 and 7. Multiplying them gives "
    "2 * 3 * 7 = 42, so gcd(84, 126) = 42. As a check, 84/42 = 2 and 126/42 = 3.\n",
    "Solve x^2 - 5x + 6 = 0 by factoring. We need two numbers whose product is 6 and whose sum is -5; "
    "the numbers -2 and -3 work.\n"
    "So the equation becomes (x - 2)(x - 3) = 0. A product is zero only when a factor is zero, which "
    "gives x = 2 or x = 3. Both values satisfy the original equation.\n",
    "Claim: the sum of two even integers is even. Let a and b be even, so a = 2m and b = 2n for some "
    "integers m and n.\n"
    "Then a + b = 2m + 2n = 2(m + n). Since m + n is an integer, a + b is two times an integer, which "
    "is the definition of an even number. This completes the proof.\n",
]

CODE = ["bisect.py", "heapq.py", "textwrap.py", "fractions.py",
        "colorsys.py", "fnmatch.py", "shlex.py", "graphlib.py"]

RESEARCH = ["training/aleph_next/ANCHOR14_PROTOCOL.md", "training/aleph_next/ARCH_PROTOCOL.md",
            "training/aleph_next/CONTINUAL_PROTOCOL.md", "training/aleph_next/DECODER_PROTOCOL.md",
            "training/aleph_next/NONLINEAR_PROTOCOL.md", "training/aleph_next/SCALE_PROTOCOL.md",
            "MeTTa/training_data/DIRECT_QUERY_PROTOCOL.md", "MeTTa/training_data/PUBLIC_SPLIT_PROTOCOL.md"]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def lf(path):
    return Path(path).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def line_starts(text, predicate):
    """Character offsets of lines that satisfy the predicate."""
    out, pos = [], 0
    for line in text.split("\n"):
        if predicate(line):
            out.append(pos)
        pos += len(line)+1
    return out


def windows(tok, text, starts, lengths):
    """Two prompts from line boundaries; the second begins after the first ends."""
    first = tok.encode(text[starts[0]:], add_special_tokens=False).ids[:lengths[0]]
    end_char = starts[0]+len(tok.decode(first))
    later = [s for s in starts[1:] if s >= end_char]
    if len(first) != lengths[0] or not later:
        raise ValueError("source document too short for two windows")
    second = tok.encode(text[later[0]:], add_special_tokens=False).ids[:lengths[1]]
    if len(second) != lengths[1]:
        raise ValueError("second window too short")
    return [(starts[0], first), (later[0], second)]


def documents():
    docs = []
    for i, text in enumerate(GENERAL, 1):
        docs.append(("general", i, f"general-{i:02d}", text, {"kind": "authored"}))
    for i, text in enumerate(MATH, 1):
        docs.append(("math", i, f"math-{i:02d}", text, {"kind": "authored"}))
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    for i, name in enumerate(CODE, 1):
        text = lf(stdlib/name)
        docs.append(("code", i, f"code-{Path(name).stem}", text,
                     {"kind": "python_stdlib", "path": name, "python": sys.version.split()[0]}))
    for i, rel in enumerate(RESEARCH, 1):
        text = lf(ROOT/rel)
        docs.append(("research", i, f"research-{Path(rel).stem.lower()}", text,
                     {"kind": "repository_file", "path": rel}))
    return docs


def starts_for(category, text):
    if category == "code":
        return line_starts(text, lambda s: s.startswith("def ") or s.startswith("    def "))
    if category == "research":
        return line_starts(text, lambda s: len(s) > 40 and not s.startswith(("#", "|", "```", "    ")))
    return line_starts(text, lambda s: len(s) > 0)


def build(tokenizer_path):
    from tokenizers import Tokenizer

    raw = Path(tokenizer_path).read_bytes()
    if sha(raw) != A.TOKENIZER:
        raise ValueError("tokenizer is not the pinned SmolLM2 tokenizer.json")
    tok = Tokenizer.from_str(raw.decode("utf-8"))
    if tok.encode(TOKEN_CHECK[0], add_special_tokens=False).ids != TOKEN_CHECK[1]:
        raise ValueError("tokenizer self-check")
    rows, manifest = [], []
    for category, index, group, text, source in documents():
        rng = random.Random(f"{SEED}:{group}")
        lengths = [rng.randint(12, 32), rng.randint(12, 32)]
        split = SPLIT_OF[index]
        for w, (offset, ids) in enumerate(windows(tok, text, starts_for(category, text), lengths)):
            rows.append({"split": split, "group": group, "ids": ids})
            manifest.append({"split": split, "group": group, "category": category, "window": w,
                             "source": source, "source_sha256": sha(text.encode("utf-8")),
                             "char_offset": offset, "tokens": len(ids),
                             "ids_sha256": A.N.digest(ids)})
    panel = {"schema": A.PANEL_SCHEMA, "tokenizer_sha256": A.TOKENIZER, "rows": rows}
    _, receipt = A.parse_panel(panel, 49152)
    return panel, manifest, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="fresh local directory")
    a = p.parse_args()
    panel, manifest, receipt = build(a.tokenizer)
    a.out.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(panel, sort_keys=True, separators=(",", ":"))+"\n").encode()
    (a.out/"afg03-panel.json").write_bytes(raw)
    inventory = {"schema": "aleph.afg03.panel_inventory.v1", "panel_file_sha256": sha(raw),
                 "panel_receipt": receipt, "rows": manifest}
    (a.out/"panel_inventory.json").write_bytes((json.dumps(inventory, sort_keys=True, indent=2)+"\n").encode())
    print(json.dumps({"panel_file_sha256": sha(raw), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
