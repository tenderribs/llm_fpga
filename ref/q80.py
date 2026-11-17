"""
numpy reference for llama2.c's int8 (Q8_0) inference, runq.c.

matmul() and quantize() reproduce runq.c bit-for-bit; they define what the
FPGA has to compute. forward() follows runq.c's forward() step by step but
uses numpy's float reductions and math library, so its logits match runq.c
closely, not exactly.
"""

from dataclasses import dataclass
import struct

import numpy as np

f32 = np.float32


@dataclass
class QTensor:
    """Q8_0 tensor: q is int8, s holds one float32 scale per group of gs
    consecutive elements of q (row-major)."""
    q: np.ndarray
    s: np.ndarray


def quantize(x: np.ndarray, gs: int) -> QTensor:
    """runq.c quantize(): per group of gs, scale = max|x| / 127 and
    q = round(x / scale), rounding halves away from zero (C round())."""
    x = x.astype(f32).reshape(-1, gs)
    scale = np.abs(x).max(axis=1) / f32(127.0)
    v = (x / scale[:, None]).astype(np.float64)  # C promotes to double for round()
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    return QTensor(q.reshape(-1), scale)


def matmul(x: QTensor, w: QTensor, n: int, d: int, gs: int) -> np.ndarray:
    """runq.c matmul(): W (d, n) @ x (n,) -> (d,).

    For each output row, per group g of gs inputs:
        ival = sum of int8 * int8 products   (exact, int32)
        val  = val + (float(ival) * w.s[row, g]) * x.s[g]
    in group order, all float32 and rounded after every operation. numpy
    evaluates each step for all rows at once, which keeps that per-row order.
    """
    G = n // gs
    xq = x.q.astype(np.int32).reshape(G, gs)
    wq = w.q.astype(np.int32).reshape(d, G, gs)
    ival = np.einsum("dgk,gk->dg", wq, xq)  # int32, exact
    ws = w.s.reshape(d, G)
    val = np.zeros(d, dtype=f32)
    for g in range(G):
        val = val + (ival[:, g].astype(f32) * ws[:, g]) * x.s[g]
    return val


def qmv_stream(w: QTensor, n: int, d: int, gs: int, R: int = 16, P: int = 32):
    """Lay W out in the order src/sources/qmv.sv consumes it:
        for each block of R rows: for each group: for each row in the block:
            for each of the gs/P beats of the group: P weights
    Returns (q, s): q is (beats, P) int8, s is (beats,) float32, the row's
    scale for that beat's group."""
    assert n % gs == 0 and gs % P == 0 and d % R == 0, (n, d, gs, R, P)
    G, bpg = n // gs, gs // P
    q = w.q.reshape(d // R, R, G, bpg, P).transpose(0, 2, 1, 3, 4).reshape(-1, P)
    s = np.repeat(w.s.reshape(d // R, R, G).transpose(0, 2, 1).reshape(-1), bpg)
    return q, s.astype(f32)


def read_matmul(path: str):
    """Read an mm_NNN.bin dump from ref/runq_dump.c.
    Returns (x, w, out, n, d, gs)."""
    with open(path, "rb") as f:
        n, d, gs, _ = struct.unpack("4i", f.read(16))
        xq = np.fromfile(f, np.int8, n)
        xs = np.fromfile(f, "<f4", n // gs)
        wq = np.fromfile(f, np.int8, n * d)
        ws = np.fromfile(f, "<f4", n * d // gs)
        out = np.fromfile(f, "<f4", d)
    return QTensor(xq, xs), QTensor(wq, ws), out, n, d, gs


class Model:
    """A Q8_0 "version 2" checkpoint, as written by ref/export_q80.py."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            magic, version = struct.unpack("Ii", f.read(8))
            assert magic == 0x616B3432 and version == 2, "not a Q8_0 v2 checkpoint"
            (self.dim, self.hidden_dim, self.n_layers, self.n_heads, self.n_kv_heads,
             self.vocab_size, self.seq_len) = struct.unpack("7i", f.read(28))
            shared = struct.unpack("B", f.read(1))[0]
            self.gs = struct.unpack("i", f.read(4))[0]
            f.seek(256)
            data = f.read()

        self.head_size = self.dim // self.n_heads
        self.kv_dim = self.n_kv_heads * self.head_size
        L, dim, hid, kv = self.n_layers, self.dim, self.hidden_dim, self.kv_dim
        off = 0

        def floats(n):
            nonlocal off
            a = np.frombuffer(data, "<f4", n, off)
            off += 4 * n
            return a

        def qtensors(count, numel):
            nonlocal off
            out = []
            for _ in range(count):
                q = np.frombuffer(data, np.int8, numel, off)
                off += numel
                out.append(QTensor(q, floats(numel // self.gs)))
            return out

        self.rms_att = floats(L * dim).reshape(L, dim)
        self.rms_ffn = floats(L * dim).reshape(L, dim)
        self.rms_final = floats(dim)
        self.q_tokens = qtensors(1, self.vocab_size * dim)[0]
        self.tok_emb = (self.q_tokens.q.reshape(-1, self.gs).astype(f32)
                        * self.q_tokens.s[:, None]).reshape(self.vocab_size, dim)
        self.wq = qtensors(L, dim * dim)
        self.wk = qtensors(L, dim * kv)
        self.wv = qtensors(L, dim * kv)
        self.wo = qtensors(L, dim * dim)
        self.w1 = qtensors(L, dim * hid)
        self.w2 = qtensors(L, hid * dim)
        self.w3 = qtensors(L, dim * hid)
        self.wcls = self.q_tokens if shared else qtensors(1, dim * self.vocab_size)[0]
        assert off == len(data), f"{len(data) - off} unread bytes"

        self.key_cache = np.zeros((L, self.seq_len, kv), f32)
        self.value_cache = np.zeros((L, self.seq_len, kv), f32)

    def mm(self, x: np.ndarray, w: QTensor, n: int, d: int) -> np.ndarray:
        return matmul(quantize(x, self.gs), w, n, d, self.gs)

    def forward(self, token: int, pos: int) -> np.ndarray:
        dim, hs, kv = self.dim, self.head_size, self.kv_dim
        kv_mul = self.n_heads // self.n_kv_heads
        x = self.tok_emb[token].copy()

        def rmsnorm(x, w):
            ss = f32(1.0) / np.sqrt(np.mean(x * x, dtype=f32) + f32(1e-5))
            return w * (ss * x)

        # RoPE rotation for this position, per pair of elements
        head_dim = np.arange(0, dim, 2) % hs
        freq = f32(1.0) / np.power(f32(10000.0), head_dim.astype(f32) / f32(hs))
        val = f32(pos) * freq
        fcr, fci = np.cos(val), np.sin(val)

        def rope(v):
            n = v.size // 2
            v0, v1 = v[0::2].copy(), v[1::2].copy()
            v[0::2] = v0 * fcr[:n] - v1 * fci[:n]
            v[1::2] = v0 * fci[:n] + v1 * fcr[:n]

        for l in range(self.n_layers):
            xb = rmsnorm(x, self.rms_att[l])
            xq = quantize(xb, self.gs)
            q = matmul(xq, self.wq[l], dim, dim, self.gs)
            k = matmul(xq, self.wk[l], dim, kv, self.gs)
            v = matmul(xq, self.wv[l], dim, kv, self.gs)
            rope(q)
            rope(k)
            self.key_cache[l, pos] = k
            self.value_cache[l, pos] = v

            xb = np.zeros(dim, f32)
            for h in range(self.n_heads):
                qh = q[h * hs:(h + 1) * hs]
                kvh = (h // kv_mul) * hs
                keys = self.key_cache[l, :pos + 1, kvh:kvh + hs]
                att = (keys @ qh) / np.sqrt(f32(hs))
                att = np.exp(att - att.max())
                att /= att.sum()
                xb[h * hs:(h + 1) * hs] = att @ self.value_cache[l, :pos + 1, kvh:kvh + hs]

            x = x + self.mm(xb, self.wo[l], dim, dim)

            xb = rmsnorm(x, self.rms_ffn[l])
            xq = quantize(xb, self.gs)
            hb = matmul(xq, self.w1[l], dim, self.hidden_dim, self.gs)
            hb2 = matmul(xq, self.w3[l], dim, self.hidden_dim, self.gs)
            hb = hb * (f32(1.0) / (f32(1.0) + np.exp(-hb))) * hb2
            x = x + self.mm(hb, self.w2[l], self.hidden_dim, dim)

        x = rmsnorm(x, self.rms_final)
        return self.mm(x, self.wcls, dim, self.vocab_size)
