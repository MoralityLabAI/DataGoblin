"""Research controller protocol v2.2; frozen v2.1 task and worker semantics."""

import hashlib
from pathlib import Path

from datagoblin.research_v2_1.suite import digest
from datagoblin.research_v2_1.suite import implementation_digest as v21_digest

VERSION = "datagoblin-research-v2.2"


def implementation_digest() -> str:
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    files["research_v2_1_impl_sha256"] = v21_digest()
    return digest(files)
