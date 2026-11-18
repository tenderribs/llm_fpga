"""
Test vectors for attn.sv (src/sim/tb_attn.sv): one attention job per
directory, the expected sums computed in float32 in runq.c's order.

  a.hex      the A RAM words (q for scores, the weights a for the weighted sum)
  beats.hex  the K / V beats as the unit reads them (256-bit, lane 0 in the
             low 32 bits)
  y.hex      expected results by index (scores: padded to a multiple of 128)

usage: python3 sim/gen_attn.py OUTDIR   (writes OUTDIR/<case>/ and prints
       the tb parameters of each case)
"""

from pathlib import Path
import sys

import numpy as np

f32 = np.float32
CASES = [  # name, mode, hs, nt
    ("s48_37", 0, 48, 37),
    ("s48_1", 0, 48, 1),
    ("s64_32", 0, 64, 32),
    ("w48_37", 1, 48, 37),
    ("w48_1", 1, 48, 1),
    ("w64_100", 1, 64, 100),
    ("w16_20", 1, 16, 20),
    ("s48_200", 0, 48, 200),
    ("s16_129", 0, 16, 129),
    ("w8_9", 1, 8, 9),
]


def hexw(x: np.ndarray) -> list:
    return [f"{int(u):08x}" for u in x.astype("<f4").view(np.uint32)]


def beat_lines(floats: np.ndarray) -> list:
    """floats (multiple of 8) -> 256-bit hex lines, lane 0 lowest"""
    w = floats.astype("<f4").view(np.uint32).reshape(-1, 8)
    return ["".join(f"{int(u):08x}" for u in row[::-1]) for row in w]


def gen(out: Path, name: str, mode: int, hs: int, nt: int, rng) -> str:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    if mode == 0:
        nblk = (nt + 127) // 128
        q = rng.standard_normal(hs).astype(f32)
        k = (rng.standard_normal((nblk * 128, hs)) * 0.7).astype(f32)  # rows past nt: padding
        y = np.empty(nblk * 128, f32)
        for t in range(nblk * 128):
            s = f32(0.0)
            for i in range(hs):
                s = f32(s + f32(q[i] * k[t, i]))
            y[t] = s
        # [blk][i][t % 128]
        stream = k.reshape(nblk, 128, hs).transpose(0, 2, 1).reshape(-1)
        a = q
    else:
        a = rng.random(nt).astype(f32)
        a = (a / a.sum(dtype=f32)).astype(f32)
        v = rng.standard_normal((nt, hs)).astype(f32)
        y = np.empty(hs, f32)
        for i in range(hs):
            s = f32(0.0)
            for t in range(nt):
                s = f32(s + f32(a[t] * v[t, i]))
            y[i] = s
        stream = v.reshape(-1)
    (d / "a.hex").write_text("\n".join(hexw(a)) + "\n")
    lines = beat_lines(stream)
    (d / "beats.hex").write_text("\n".join(lines) + "\n")
    (d / "y.hex").write_text("\n".join(hexw(y)) + "\n")
    return f"{name} {mode} {hs} {nt} {len(lines)} {len(y)}"


def main() -> int:
    out = Path(sys.argv[1])
    rng = np.random.default_rng(7)
    for c in CASES:
        print(gen(out, *c, rng))
    chain(out)
    return 0


def chain(out: Path):
    """chain48_37/y.hex: the vector unit's softmax of s48_37's scores (scaled
    by 1 / sqrt(48)) as the weights for w48_37's V, for tb_llm_core"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_vpu import softmax_scaled
    scores = np.array([int(l, 16) for l in open(out / "s48_37/y.hex")], np.uint32).view(f32)[:37]
    rows = [int(l, 16) for l in open(out / "w48_37/beats.hex")]
    v = np.array([[(r >> (32 * k)) & 0xFFFFFFFF for k in range(8)] for r in rows],
                 np.uint32).view(f32).reshape(37, 48)
    a = softmax_scaled(scores, f32(np.sqrt(f32(48))))
    y = np.empty(48, f32)
    for i in range(48):
        acc = f32(0.0)
        for t in range(37):
            acc = f32(acc + f32(a[t] * v[t, i]))
        y[i] = acc
    d = out / "chain48_37"
    d.mkdir(parents=True, exist_ok=True)
    (d / "y.hex").write_text("\n".join(hexw(y)) + "\n")


if __name__ == "__main__":
    sys.exit(main())
