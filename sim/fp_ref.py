"""
Bit-exact numpy float32 versions of the firmware's float helpers
(fw/model/mathf.c), for test vectors.
"""

import numpy as np

f32 = np.float32


def bits(x) -> np.ndarray:
    return np.asarray(x, dtype=f32).view(np.uint32)


def from_bits(u) -> np.ndarray:
    return np.asarray(u, dtype=np.uint32).view(f32)


LOG2E, LN2_HI, LN2_LO = f32(1.44269502163), f32(0.693145751953125), f32(1.42860676533e-6)
BIG, SMALL = f32(88.7228394), f32(-103.972084)
HORNER = [f32(1) / f32(720), f32(1) / f32(120), f32(1) / f32(24), f32(1) / f32(6), f32(0.5),
          f32(1), f32(1)]
C5040 = f32(1) / f32(5040)


def m_expf(x: np.ndarray) -> np.ndarray:
    """fw/model/mathf.c m_expf, elementwise (x float32 array)"""
    x = np.asarray(x, f32)
    with np.errstate(all="ignore"):
        t = x * LOG2E
        u = np.where(t < 0, t - f32(0.5), t + f32(0.5)).astype(f32)
        n = np.trunc(u).astype(np.int32)
        fn = n.astype(f32)
        r = ((x - fn * LN2_HI) - fn * LN2_LO).astype(f32)
        p = np.full_like(x, C5040)
        for c in HORNER:
            p = (c + r * p).astype(f32)
        big_n = n > 127
        small_n = n < -126
        p = np.where(big_n, p * f32(2), p).astype(f32)
        p = np.where(small_n, p * from_bits(np.uint32(0x0D800000)), p).astype(f32)
        n = np.where(big_n, n - 1, np.where(small_n, n + 100, n))
        y = (p * from_bits(((n + 127).astype(np.uint32)) << 23)).astype(f32)
        y = np.where(x > BIG, f32(np.inf), y)
        y = np.where(x < SMALL, f32(0), y)
    return y.astype(f32)
