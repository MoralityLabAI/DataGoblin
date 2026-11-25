"""Train-time overlap screens for frozen text sources and incoming complete documents."""

import re
import zlib
from array import array
from collections import Counter, defaultdict

from text_intake import duplicate_signature, words

WIDTH = 12
STRIDE = 13
SPAN = WIDTH + STRIDE - 1
BASE = 1000003
MASK = (1 << 64) - 1
POWER = pow(BASE, WIDTH - 1, 1 << 64)
DIGITS = re.compile(r"\d+")


def symbols(text):
    return array("I", (zlib.crc32(DIGITS.sub("#", word).encode()) for word in words(text)))


def anchors(ids):
    if len(ids) < WIDTH:
        return
    value = 0
    for token in ids[:WIDTH]:
        value = (value * BASE + token) & MASK
    yield 0, value
    for i in range(WIDTH, len(ids)):
        value = ((value - ids[i - WIDTH] * POWER) * BASE + ids[i]) & MASK
        yield i - WIDTH + 1, value


class Overlap:
    """Every shared 24-word span contains a 12-word reference anchor on a 13-word grid."""

    def __init__(self):
        self.docs = []
        self.locations = defaultdict(list)
        self.sketches = []
        self.inverted = defaultdict(list)
        self.normalized = set()

    def add(self, text):
        norm, sketch = duplicate_signature(text)
        self.add_prepared(norm, sketch, symbols(text))

    def add_prepared(self, norm, sketch, ids):
        j = len(self.docs)
        self.docs.append(ids)
        self.normalized.add(norm)
        for k in sketch:
            self.inverted[k].append(j)
        self.sketches.append(sketch)
        for start, h in anchors(ids):
            if start % STRIDE == 0:
                self.locations[h].append((j, start))

    def check(self, norm, sketch, ids):
        if norm in self.normalized:
            return "exact_or_numeral_template"
        hits = Counter(j for k in sketch for j in self.inverted.get(k, ()))
        if any(n >= 0.75 * max(len(sketch), len(self.sketches[j])) for j, n in hits.items()):
            return "near_document"
        for i, h in anchors(ids):
            for j, k in self.locations.get(h, ()):
                ref = self.docs[j]
                if ids[i : i + WIDTH] != ref[k : k + WIDTH]:
                    continue
                left = 0
                while left < SPAN - WIDTH and left < min(i, k) and ids[i - left - 1] == ref[k - left - 1]:
                    left += 1
                width = WIDTH + left
                tail = 0
                while (
                    width < SPAN
                    and i + WIDTH + tail < len(ids)
                    and k + WIDTH + tail < len(ref)
                    and ids[i + WIDTH + tail] == ref[k + WIDTH + tail]
                ):
                    width += 1
                    tail += 1
                if width == SPAN:
                    return "shared_24_words"
        return None
