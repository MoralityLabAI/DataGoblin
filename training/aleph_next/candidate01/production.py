"""Frozen Aleph-A tokenizer. This module never fetches weights or starts a trainer."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER_SHA256 = "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c"
TOKENIZER_REV = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
MODEL_REPO = "HuggingFaceTB/SmolLM2-135M"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def load_tokenizer(path):
    import tokenizers

    if digest(path) != TOKENIZER_SHA256 or tokenizers.__version__ != "0.22.2":
        raise ValueError("tokenizer artifact or implementation differs from freeze")
    tok = tokenizers.Tokenizer.from_file(str(path))
    tok.encode_special_tokens = True  # Treat printed control-token spellings in documents as text.
    if tok.get_vocab_size() != 49152 or tok.encode("∀ x, x = x", add_special_tokens=False).ids != [
        22052,
        218,
        1792,
        28,
        1792,
        446,
        1792,
    ]:
        raise ValueError("unexpected tokenizer geometry or Unicode fixture")
    if 0 in tok.encode("<|endoftext|>", add_special_tokens=False).ids:
        raise ValueError("literal document text became a control token")
    return tok


def read(path):
    return json.loads(Path(path).read_bytes())
