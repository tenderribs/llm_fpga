"""
Test vectors for the quantized matmul engine (src/sources/qmv.sv) and its
float units.

  fp <outdir> [--count N]
      add.hex, mul.hex: lines "a b y" (fp32 bit patterns) with y = a+b or a*b
      in numpy float32 (round to nearest even). i2f.hex: lines "a y" with
      y = float32(int32 a). Cases whose inputs or result would be subnormal,
      inf or NaN are left out: the units flush/saturate those by design.

  rand-dump <path> <n> <d> <gs>
      a random matmul in runq_dump.c's format (for group sizes the model
      dumps don't cover); expected output from q80.matmul

  mm <outdir> <dump.bin> [--r 16] [--p 32]
      a runq.c matmul dump (ref/runq_dump.c) laid out for qmv:
        cfg.hex  n, d
        xq.hex   n/P words of P bytes (byte i = lane i, lane 0 least significant)
        xs.hex   n/GS x scales
        w.hex    weight beats in qmv stream order, each {w_s (32b), w_q (P bytes)}
        y.hex    expected outputs (fp32 bit patterns, from runq.c)
"""

import argparse as ap
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ref"))
import q80  # noqa: E402

f32 = np.float32
TINY = np.finfo(f32).tiny  # smallest normal


def is_normal_or_zero(x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    return np.isfinite(x) & ((a == 0) | (a >= TINY))


def bits(x) -> np.ndarray:
    return np.asarray(x, dtype=f32).view(np.uint32)


def from_bits(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint32).view(f32)


def rand_floats(rng, n, exp_lo=40, exp_hi=210):
    sign = rng.integers(0, 2, n, dtype=np.uint32) << 31
    exp = rng.integers(exp_lo, exp_hi, n, dtype=np.uint32) << 23
    frac = rng.integers(0, 1 << 23, n, dtype=np.uint32)
    return from_bits(sign | exp | frac)


def short_mantissa(rng, n, nbits):
    """floats whose significand has only nbits significant bits: products and
    sums of these hit exact rounding ties often"""
    m = rng.integers(1, 1 << nbits, n).astype(np.float64)
    e = rng.integers(-20, 20, n)
    s = rng.choice([-1.0, 1.0], n)
    return (s * m * np.exp2(e)).astype(f32)


def gen_fp(outdir: Path, count: int):
    rng = np.random.default_rng(1)
    outdir.mkdir(parents=True, exist_ok=True)
    q = count // 4

    # --- add
    a = rand_floats(rng, count)
    b = rand_floats(rng, count)
    # close exponents (real alignment shifts, not just "b vanishes")
    near = from_bits((bits(a[:q]) & 0x807FFFFF)
                     | ((((bits(a[:q]) >> 23) & 0xFF)
                         + rng.integers(-26, 27, q).astype(np.uint32)) << 23 & 0x7F800000))
    b[:q] = np.where(rng.integers(0, 2, q) == 1, near, -near)
    # near-cancellation: b = -a with the last few bits changed
    b[q:2 * q] = -from_bits(bits(a[q:2 * q]) + rng.integers(-40, 41, q).astype(np.uint32))
    # ties
    a[2 * q:3 * q] = short_mantissa(rng, q, 24)
    b[2 * q:3 * q] = short_mantissa(rng, q, 24)
    special = [(1.0, 2.0 ** -24), (1.0, 3 * 2.0 ** -24), (1.0 + 2.0 ** -23, 2.0 ** -24),
               (1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (-0.0, -0.0), (0.0, -0.0),
               (5.0, 0.0), (0.0, -3.0), (1.0, -(1.0 - 2.0 ** -24)), (2.0 ** 100, 1.0)]
    a = np.concatenate([a, f32([s[0] for s in special])])
    b = np.concatenate([b, f32([s[1] for s in special])])
    with np.errstate(all="ignore"):
        y = a + b
    keep = is_normal_or_zero(a) & is_normal_or_zero(b) & is_normal_or_zero(y)
    write_vec(outdir / "add.hex", [bits(a[keep]), bits(b[keep]), bits(y[keep])])

    # --- mul
    a = rand_floats(rng, count, 70, 185)
    b = rand_floats(rng, count, 70, 185)
    a[:q] = short_mantissa(rng, q, 12)
    b[:q] = short_mantissa(rng, q, 13)
    special = [(0.0, 5.0), (-0.0, 5.0), (-2.0, 0.0), (1.0, -1.0), (1.5, 1.5),
               (2.0 ** -63, 2.0 ** -63), (2.0 ** 63, 2.0 ** 64), (2.0 ** 64, 2.0 ** 64)]
    a = np.concatenate([a, f32([s[0] for s in special])])
    b = np.concatenate([b, f32([s[1] for s in special])])
    with np.errstate(all="ignore"):
        y = a * b
    keep = is_normal_or_zero(a) & is_normal_or_zero(b) & is_normal_or_zero(y)
    write_vec(outdir / "mul.hex", [bits(a[keep]), bits(b[keep]), bits(y[keep])])

    # --- int32 -> float, including values that need rounding (> 2^24)
    i = rng.integers(-(1 << 31), 1 << 31, count, dtype=np.int64)
    i[:q] = rng.integers(-(1 << 21), 1 << 21, q)  # the range qmv actually uses
    i[q:2 * q] = (rng.integers(-(1 << 8), 1 << 8, q) << rng.integers(0, 23, q)) \
        + rng.choice([-(1 << 7), 0, 1 << 7], q)  # ties around 2^24..2^31
    i = np.clip(i, -(1 << 31), (1 << 31) - 1)
    i = np.concatenate([i, [0, 1, -1, (1 << 31) - 1, -(1 << 31), (1 << 24) + 1, (1 << 24) + 3]])
    i32 = i.astype(np.int32)
    write_vec(outdir / "i2f.hex", [i32.view(np.uint32), bits(i32.astype(f32))])

    for name in ["add", "mul", "i2f"]:
        n = sum(1 for _ in open(outdir / f"{name}.hex"))
        print(f"{name}: {n} vectors")


def write_vec(path: Path, cols):
    with open(path, "w") as f:
        for row in zip(*cols):
            f.write(" ".join(f"{int(v):08X}" for v in row) + "\n")


def gen_mm(outdir: Path, dump: Path, R: int, P: int):
    x, w, out, n, d, gs = q80.read_matmul(str(dump))
    assert n % gs == 0 and gs % P == 0 and d % R == 0, (n, d, gs)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "cfg.hex", "w") as f:
        f.write(f"{n:08X}\n{d:08X}\n")
    with open(outdir / "xq.hex", "w") as f:
        for beat in x.q.reshape(-1, P):
            f.write(beat.tobytes()[::-1].hex().upper() + "\n")
    with open(outdir / "xs.hex", "w") as f:
        f.write("\n".join(f"{v:08X}" for v in bits(x.s)) + "\n")

    wq, ws = q80.qmv_stream(w, n, d, gs, R, P)
    ws = bits(ws)
    with open(outdir / "w.hex", "w") as f:
        for s, q in zip(ws, wq):
            f.write(f"{int(s):08X}" + q.tobytes()[::-1].hex().upper() + "\n")

    with open(outdir / "y.hex", "w") as f:
        f.write("\n".join(f"{v:08X}" for v in bits(out)) + "\n")
    print(f"{dump.name}: n={n} d={d} gs={gs}, {len(wq)} beats -> {outdir}")
    return n, d, gs


def gen_rand_dump(path: Path, n: int, d: int, gs: int, seed: int = 0):
    """A random matmul in runq_dump.c's format, with the output from
    q80.matmul (bit-exact with runq.c, see ref/check_ref.py)."""
    rng = np.random.default_rng(seed)
    x = q80.quantize(rng.standard_normal(n).astype(f32), gs)
    w = q80.quantize((rng.standard_normal((d, n)) * 0.05).astype(f32), gs)
    out = q80.matmul(x, w, n, d, gs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array([n, d, gs, 0], "<i4").tobytes())
        for a in (x.q, x.s.astype("<f4"), w.q, w.s.astype("<f4"), out.astype("<f4")):
            f.write(a.tobytes())


if __name__ == "__main__":
    parser = ap.ArgumentParser(prog="gen_qmv.py", description=__doc__.split("\n")[1])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_fp = sub.add_parser("fp")
    p_fp.add_argument("outdir", type=Path)
    p_fp.add_argument("--count", type=int, default=20000)
    p_mm = sub.add_parser("mm")
    p_mm.add_argument("outdir", type=Path)
    p_mm.add_argument("dump", type=Path)
    p_mm.add_argument("--r", type=int, default=16)
    p_mm.add_argument("--p", type=int, default=32)
    p_rd = sub.add_parser("rand-dump", help="write a random matmul dump")
    p_rd.add_argument("path", type=Path)
    p_rd.add_argument("n", type=int)
    p_rd.add_argument("d", type=int)
    p_rd.add_argument("gs", type=int)
    args = parser.parse_args()
    if args.cmd == "fp":
        gen_fp(args.outdir, args.count)
    elif args.cmd == "rand-dump":
        gen_rand_dump(args.path, args.n, args.d, args.gs)
    else:
        gen_mm(args.outdir, args.dump, args.r, args.p)
