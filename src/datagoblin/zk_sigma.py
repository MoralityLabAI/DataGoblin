"""Experimental, non-succinct Ristretto255 arithmetic proof; not Nova."""
from __future__ import annotations

import ctypes as ct
import ctypes.util
import hashlib
import secrets
from dataclasses import dataclass, field

from datagoblin.zk_replay import Job, Plan, canonical, clamp, require, validate

Q = 2**252 + 27742317777372353535851937790883648493
ZERO = bytes(32)
BACKEND = "ristretto-sigma-dev-v1"
Expr = tuple[tuple[int, int], ...]


def scalar(x: int) -> bytes:
    return (x % Q).to_bytes(32, "little")


class Group:
    def __init__(self) -> None:
        path = ctypes.util.find_library("sodium")
        if path is None:
            raise RuntimeError("libsodium with Ristretto255 required")
        self.lib = ct.CDLL(path)
        require(self.lib.sodium_init() >= 0, "sodium init")
        for name, n in (("from_hash", 2), ("add", 3), ("is_valid_point", 1)):
            fn = getattr(self.lib, "crypto_core_ristretto255_" + name)
            fn.argtypes, fn.restype = [ct.c_void_p] * n, ct.c_int
        for suffix, n in (("", 3), ("_base", 2)):
            fn = getattr(self.lib, "crypto_scalarmult_ristretto255" + suffix)
            fn.argtypes, fn.restype = [ct.c_void_p] * n, ct.c_int
        self.g = self.call("crypto_scalarmult_ristretto255_base", scalar(1))
        self.h = self.call("crypto_core_ristretto255_from_hash", hashlib.sha512(b"zk-dg/sigma/H/v1").digest())
        require(self.h not in (ZERO, self.g), "bad generator")

    def call(self, name: str, *args: bytes) -> bytes:
        out = ct.create_string_buffer(32)
        require(getattr(self.lib, name)(out, *args) == 0, "group operation")
        return out.raw

    def add(self, a: bytes, b: bytes) -> bytes:
        if a == ZERO:
            return b
        if b == ZERO:
            return a
        return self.call("crypto_core_ristretto255_add", a, b)

    def mul(self, x: int, a: bytes) -> bytes:
        x %= Q
        if x == 0 or a == ZERO:
            return ZERO
        if x == 1:
            return a
        return self.call("crypto_scalarmult_ristretto255", scalar(x), a)

    def commit(self, v: int, r: int) -> bytes:
        return self.add(self.mul(v, self.g), self.mul(r, self.h))

    def point(self, text: str) -> bytes:
        require(type(text) is str and len(text) == 64, "point encoding")
        p = bytes.fromhex(text)
        require(p.hex() == text and self.lib.crypto_core_ristretto255_is_valid_point(p) == 1, "invalid point")
        return p


def var(i: int) -> Expr:
    return ((i, 1),)


def const(v: int) -> Expr:
    return ((0, v),)


def neg(e: Expr) -> Expr:
    return tuple((i, -a) for i, a in e)


@dataclass
class Circuit:
    values: list[int] = field(default_factory=lambda: [1])
    public: dict[int, int] = field(default_factory=lambda: {0: 1})
    linear: list[Expr] = field(default_factory=list)
    products: list[tuple[Expr, Expr, Expr]] = field(default_factory=list)

    def alloc(self, v: int, public: bool = False) -> Expr:
        i = len(self.values)
        self.values.append(v)
        if public:
            self.public[i] = v
        return var(i)

    def value(self, expr: Expr) -> int:
        return sum(self.values[i] * a for i, a in expr)

    def range(self, expr: Expr, bits: int) -> None:
        v = self.value(expr)
        # Verifier construction uses dummy witnesses; only proof verification checks constraints.
        terms = []
        for j in range(bits):
            bit = self.alloc((v >> j) & 1)
            self.products.append((bit, bit, bit))
            terms.append((bit[0][0], -(1 << j)))
        self.linear.append(expr + tuple(terms))

    def clip(self, q: Expr, y: Expr, bound_bits: int) -> None:
        self.range(y + const(127), 8)
        self.range(const(127) + neg(y), 8)
        delta = self.value(q) - self.value(y)
        a, b = self.alloc(max(delta, 0)), self.alloc(max(-delta, 0))
        self.range(a, bound_bits)
        self.range(b, bound_bits)
        self.linear.append(q + neg(y) + neg(a) + b)
        self.products.append((a, const(127) + neg(y), ()))
        self.products.append((b, const(127) + y, ()))


def build(job: Job, x: tuple[int, ...], y: tuple[int, ...], witness: bool) -> Circuit:
    require(len(x) == len(y) == len(job.source) and 1 <= len(x) <= 2 and job.steps <= 32, "dev circuit limits")
    validate(Plan("dev", "dev", "dev", (Job(job.name, (), x, job.weights, job.bias, job.steps),)))
    require(all(type(v) is int and -127 <= v <= 127 for v in y), "invalid output")
    c = Circuit()
    h = [c.alloc(v, True) for v in x]
    for t in range(job.steps):
        nxt = []
        for i, row in enumerate(job.weights):
            q = tuple((e[0][0], w) for e, w in zip(h, row, strict=True)) + const(job.bias[i] + x[i])
            last = t == job.steps - 1
            z = c.alloc(y[i] if last else (clamp(c.value(q)) if witness else 0), last)
            bits = (127 * sum(abs(w) for w in row) + abs(job.bias[i] + x[i]) + 127).bit_length()
            c.clip(q, z, max(1, bits))
            nxt.append(z)
        h = nxt
    return c


def point_expr(group: Group, points: list[bytes], expr: Expr) -> bytes:
    p = ZERO
    for i, a in expr:
        p = group.add(p, group.mul(a, points[i]))
    return p


def dot(values: list[int], expr: Expr) -> int:
    return sum(values[i] * a for i, a in expr) % Q


def challenge(statement: dict, commitments: list[str], announcements: list[list[str]]) -> int:
    raw = canonical([BACKEND, statement, commitments, announcements])
    return int.from_bytes(hashlib.sha512(b"zk-dg/fs/v1\0" + raw).digest(), "little") % Q


def rand() -> int:
    return secrets.randbelow(Q)


def statement(job: Job, x: tuple[int, ...], y: tuple[int, ...], context: str) -> dict:
    require(type(context) is str and len(context) == 64 and bytes.fromhex(context).hex() == context, "context digest")
    return {"context": context, "job": job.name, "weights": job.weights, "bias": job.bias,
            "steps": job.steps, "input": x, "output": y}


def prove(job: Job, x: tuple[int, ...], y: tuple[int, ...], context: str) -> dict:
    c = build(job, x, y, True)
    s = statement(job, x, y, context)
    require(all(c.value(e) == 0 for e in c.linear), "invalid linear witness")
    require(all(c.value(a) * c.value(b) == c.value(z) for a, b, z in c.products), "invalid product witness")
    group = Group()
    blinds = [0 if i in c.public else rand() for i in range(len(c.values))]
    points = [group.commit(v, r) for v, r in zip(c.values, blinds, strict=True)]
    commits = [p.hex() for p in points]
    announcements, secrets_and_witnesses = [], []
    for e in c.linear:
        k = rand()
        announcements.append([group.mul(k, group.h).hex()])
        secrets_and_witnesses.append(((k,), (dot(blinds, e),)))
    for a, b, z in c.products:
        u, v, w = rand(), rand(), rand()
        bp = point_expr(group, points, b)
        av = c.value(a)
        announcements.append([group.commit(u, v).hex(), group.add(group.mul(u, bp), group.mul(w, group.h)).hex()])
        secrets_and_witnesses.append(((u, v, w), (av, dot(blinds, a), dot(blinds, z) - av * dot(blinds, b))))
    e = challenge(s, commits, announcements)
    responses = [[scalar(k + e * v).hex() for k, v in zip(ks, vs, strict=True)] for ks, vs in secrets_and_witnesses]
    return {"backend": BACKEND, "commitments": commits, "announcements": announcements, "responses": responses}


def decode_scalar(text: str) -> int:
    require(type(text) is str and len(text) == 64, "scalar encoding")
    raw = bytes.fromhex(text)
    value = int.from_bytes(raw, "little")
    require(raw.hex() == text and value < Q, "noncanonical scalar")
    return value


def verify(job: Job, x: tuple[int, ...], y: tuple[int, ...], context: str, proof: dict) -> bool:
    try:
        c = build(job, x, y, False)
        s = statement(job, x, y, context)
        require(set(proof) == {"backend", "commitments", "announcements", "responses"}, "proof fields")
        require(proof["backend"] == BACKEND, "wrong backend")
        commits, ann, res = (proof[k] for k in ("commitments", "announcements", "responses"))
        count = len(c.linear) + len(c.products)
        require(all(type(v) is list for v in (commits, ann, res)), "proof encoding")
        require(len(commits) == len(c.values) and len(ann) == len(res) == count, "proof shape")
        group = Group()
        points = [group.point(p) for p in commits]
        require(all(points[i] == group.mul(v, group.g) for i, v in c.public.items()), "public witness")
        for j, (a, r) in enumerate(zip(ann, res, strict=True)):
            linear = j < len(c.linear)
            require(type(a) is list and type(r) is list and len(a) == (1 if linear else 2)
                    and len(r) == (1 if linear else 3), "constraint proof shape")
        # Parse all encodings before hashing to enforce the unique transcript format.
        ts = [[group.point(p) for p in row] for row in ann]
        zs = [[decode_scalar(v) for v in row] for row in res]
        e = challenge(s, commits, ann)
        for j, expr in enumerate(c.linear):
            p = point_expr(group, points, expr)
            require(group.mul(zs[j][0], group.h) == group.add(ts[j][0], group.mul(e, p)), "linear proof")
        for j, (a, b, z) in enumerate(c.products, len(c.linear)):
            ap, bp, zp = (point_expr(group, points, expr) for expr in (a, b, z))
            u, v, w = zs[j]
            require(group.commit(u, v) == group.add(ts[j][0], group.mul(e, ap)), "product opening")
            require(
                group.add(group.mul(u, bp), group.mul(w, group.h)) == group.add(ts[j][1], group.mul(e, zp)),
                "product proof",
            )
        return True
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False


def nova_step_export(job: Job, x: tuple[int, ...], h: tuple[int, ...], counter: int, context: str) -> dict:
    """Backend-neutral fixed-shape R1CS step; no Nova folding occurs here."""
    width = len(x)
    require(width == len(h) and 1 <= width <= 2, "step width")
    validate(Plan("dev", "dev", "dev", (Job(job.name, (), x, job.weights, job.bias, 1),)))
    require(all(type(v) is int and -127 <= v <= 127 for v in h), "state range")
    require(type(counter) is int and 0 <= counter < 32, "step counter")
    statement(job, x, h, context)
    c = Circuit()
    xs = [c.alloc(v, True) for v in x]
    hs = [c.alloc(v, True) for v in h]
    cnt = c.alloc(counter, True)
    ctx = [c.alloc(int(context[i:i+32], 16), True) for i in (0, 32)]
    for e in (*xs, *hs):
        c.range(e + const(127), 8)
        c.range(const(127) + neg(e), 8)
    c.range(cnt, 5)
    ys = []
    for i, row in enumerate(job.weights):
        q = tuple((e[0][0], w) for e, w in zip(hs, row, strict=True)) + const(job.bias[i]) + xs[i]
        y = c.alloc(clamp(c.value(q)), True)
        c.clip(q, y, 19)
        ys.append(y)
    xn = [c.alloc(v, True) for v in x]
    cn = c.alloc(counter + 1, True)
    ctxn = [c.alloc(c.value(e), True) for e in ctx]
    for old, new in zip((*xs, *ctx), (*xn, *ctxn), strict=True):
        c.linear.append(old + neg(new))
    c.linear.append(cnt + const(1) + neg(cn))
    def idx(exprs: list[Expr]) -> list[int]:
        return [e[0][0] for e in exprs]  # wire 0 is constant one

    return {"format": "zk-dg-sparse-integer-r1cs-v1", "relation": "fixed-model-recurrent-step-v1",
            "state_in": idx([*xs, *hs, cnt, *ctx]), "state_out": idx([*xn, *ys, cn, *ctxn]),
            "constraints": [(e, const(1), ()) for e in c.linear] + c.products,
            "witness": c.values, "model": [job.weights, job.bias]}


def r1cs_satisfied(export: dict, modulus: int) -> bool:
    values = export["witness"]
    return modulus > 2**128 and values[0] == 1 and all(
        (sum(values[i] * a for i, a in left) * sum(values[i] * a for i, a in r)
         - sum(values[i] * a for i, a in o)) % modulus == 0
        for left, r, o in export["constraints"]
    )
