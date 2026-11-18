"""
Test vectors for vpu.sv (src/sim/tb_vpu.sv): inputs and the results of the
firmware's float code (fw/model/model.c), computed in float32 numpy in the
same order (sim/fp_ref.py for m_expf).

  OUTDIR/<case>/in.hex    input words (QUANT: x; SILU: a then b; SOFTMAX: scores)
  OUTDIR/<case>/out.hex   expected: QUANT: 32-byte lines then scales;
                          SILU: results; SOFTMAX: attention weights

usage: python3 sim/gen_vpu.py OUTDIR   (prints the tb parameters per case)
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fp_ref import bits, m_expf  # noqa: E402

f32 = np.float32


def quantize(x: np.ndarray):
    """model.c quantize(): per group of 32, scale = max|x| / 127,
    q = round half away (x / scale)"""
    q = np.empty(len(x), np.int8)
    s = np.empty(len(x) // 32, f32)
    with np.errstate(all="ignore"):
        for g in range(len(x) // 32):
            xg = x[32 * g:32 * g + 32]
            scale = f32(np.max(np.abs(xg))) / f32(127)
            v = (xg / scale).astype(f32).astype(np.float64)
            r = np.where(np.isnan(v), -1, np.where(v >= 0, np.floor(v + 0.5), -np.floor(-v + 0.5)))
            q[32 * g:32 * g + 32] = r.astype(np.int64).astype(np.int8)
            s[g] = scale
    return q, s


def silu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    e = m_expf((-a).astype(f32))
    d = (f32(1) / (f32(1) + e).astype(f32)).astype(f32)
    return ((a * d).astype(f32) * b).astype(f32)


def softmax_scaled(y: np.ndarray, d: f32) -> np.ndarray:
    s = (y / d).astype(f32)
    mx = s[0]
    for v in s[1:]:
        if v > mx:
            mx = v
    e = m_expf((s - mx).astype(f32))
    total = f32(0)
    for v in e:
        total = f32(total + v)
    return (e / total).astype(f32)


def words(a) -> list:
    return [f"{int(u):08x}" for u in np.asarray(a).view(np.uint32)]


def main() -> int:
    out = Path(sys.argv[1])
    rng = np.random.default_rng(11)
    cases = []

    def write(name, op, n, inp, exp):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "in.hex").write_text("\n".join(inp) + "\n")
        (d / "out.hex").write_text("\n".join(exp) + "\n")
        cases.append(f"{name} {op} {n} {len(inp)} {len(exp)}")

    for n in (288, 768):
        x = (rng.standard_normal(n) * rng.uniform(0.1, 3)).astype(f32)
        x[64:96] = 0  # an all-zero group: scale 0, 0 / 0
        q, s = quantize(x)
        lines = ["".join(f"{int(b) & 0xFF:02x}" for b in q[32 * g:32 * g + 32][::-1])
                 for g in range(n // 32)]
        write(f"quant{n}", 1, n, words(x), lines + words(s))

    n = 768
    a = (rng.standard_normal(n) * 2).astype(f32)
    b = rng.standard_normal(n).astype(f32)
    write("silu768", 2, n, words(a) + words(b), words(silu(a, b)))

    d = f32(np.sqrt(f32(48)))
    for n in (1, 37, 129):
        y = (rng.standard_normal(n) * 8).astype(f32)
        write(f"softmax{n}", 3, n, words(y), words(softmax_scaled(y, d)))

    for c in cases:
        print(c, f"{int(bits(d)):08x}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
