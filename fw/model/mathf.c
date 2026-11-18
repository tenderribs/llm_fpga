// Float-only expf / sincosf / roundf for the model code.
//
// Not correctly rounded (about 1-2 ulp), but plain float arithmetic in a
// fixed order, so they give the same bits on the board's FPU as on an x86
// host (both IEEE single precision, round to nearest even, no FMA with
// -ffp-contract=off). That is what lets the host build of the model be a
// bit-exact golden for the board. The C library's versions differ between
// the two (newlib vs glibc), which would break that.
#include <stdint.h>
#include <string.h>
#include "model.h"

static float from_bits(uint32_t u) {
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// e^x = 2^n e^r, r = x - n ln2 in [-ln2/2, ln2/2]; ln2 in two parts so n*LN2_HI
// is exact (LN2_HI has 15 significant bits, |n| < 2^8)
float m_expf(float x) {
    const float LOG2E = 1.44269502163f, LN2_HI = 0.693145751953125f, LN2_LO = 1.42860676533e-6f;
    if (x != x) return x;
    if (x > 88.7228394f) return from_bits(0x7F800000u);  // +inf
    if (x < -103.972084f) return 0.0f;
    float t = x * LOG2E;
    int32_t n = (int32_t)(t < 0.0f ? t - 0.5f : t + 0.5f);
    float r = (x - (float)n * LN2_HI) - (float)n * LN2_LO;
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (1.0f / 6 + r * (1.0f / 24 + r * (1.0f / 120
              + r * (1.0f / 720 + r * (1.0f / 5040)))))));
    // scale by 2^n, in two steps where 2^n alone isn't a normal float
    if (n > 127) {
        p *= 2.0f;
        n -= 1;
    } else if (n < -126) {
        p *= from_bits(0x0D800000u);  // 2^-100
        n += 100;
    }
    return p * from_bits((uint32_t)(n + 127) << 23);
}

// m_expf on four values at once, the same operations on each: four
// independent chains the FPU overlaps (one chain waits on every operation's
// latency). Outside the range where 2^n is a normal number, lane by lane.
void m_expf4(const float *x, float *y) {
    const float LOG2E = 1.44269502163f, LN2_HI = 0.693145751953125f, LN2_LO = 1.42860676533e-6f;
    for (int k = 0; k < 4; k++) {
        if (!(x[k] > -87.0f && x[k] < 88.0f)) {
            for (int j = 0; j < 4; j++) y[j] = m_expf(x[j]);
            return;
        }
    }
#define LANE(k)                                                                        \
    float t##k = x[k] * LOG2E;                                                        \
    int32_t n##k = (int32_t)(t##k < 0.0f ? t##k - 0.5f : t##k + 0.5f);               \
    float r##k = (x[k] - (float)n##k * LN2_HI) - (float)n##k * LN2_LO;               \
    float p##k = 1.0f / 5040;
    LANE(0) LANE(1) LANE(2) LANE(3)
#undef LANE
#define STEP(c) p0 = (c) + r0 * p0; p1 = (c) + r1 * p1; p2 = (c) + r2 * p2; p3 = (c) + r3 * p3;
    // the same Horner steps as m_expf: 1 + r (1 + r (1/2 + r (1/6 + ...)))
    STEP(1.0f / 720) STEP(1.0f / 120) STEP(1.0f / 24) STEP(1.0f / 6) STEP(0.5f) STEP(1.0f)
    STEP(1.0f)
#undef STEP
    y[0] = p0 * from_bits((uint32_t)(n0 + 127) << 23);
    y[1] = p1 * from_bits((uint32_t)(n1 + 127) << 23);
    y[2] = p2 * from_bits((uint32_t)(n2 + 127) << 23);
    y[3] = p3 * from_bits((uint32_t)(n3 + 127) << 23);
}

// x = k pi/2 + r, |r| <= pi/4; pi/2 in three parts, the first two with 12
// significant bits so k * part is exact for |k| < 2^12 (|x| < ~6400)
void m_sincosf(float x, float *s, float *c) {
    const float TWO_OVER_PI = 0.636619746685f;
    const float PIO2_HI = 1.5703125f, PIO2_MID = 4.837512969970703e-4f,
                PIO2_LO = 7.549790126404332e-8f;
    float a = x < 0.0f ? -x : x;
    int32_t k = (int32_t)(a * TWO_OVER_PI + 0.5f);
    float r = ((a - (float)k * PIO2_HI) - (float)k * PIO2_MID) - (float)k * PIO2_LO;
    float r2 = r * r;
    float sr = r + r * r2 * (-1.0f / 6 + r2 * (1.0f / 120 + r2 * (-1.0f / 5040
               + r2 * (1.0f / 362880))));
    float cr = 1.0f + r2 * (-0.5f + r2 * (1.0f / 24 + r2 * (-1.0f / 720 + r2 * (1.0f / 40320
               + r2 * (-1.0f / 3628800)))));
    float sa, ca;
    switch (k & 3) {
    case 0: sa = sr; ca = cr; break;
    case 1: sa = cr; ca = -sr; break;
    case 2: sa = -sr; ca = -cr; break;
    default: sa = -cr; ca = sr; break;
    }
    *s = x < 0.0f ? -sa : sa;
    *c = ca;
}

// round half away from zero, like C round() (exact)
float m_roundf(float x) {
    float a = x < 0.0f ? -x : x;
    if (!(a < 8388608.0f)) return x;  // already an integer (or NaN)
    float t = (float)(int32_t)a;
    if (a - t >= 0.5f) t += 1.0f;
    return x < 0.0f ? -t : t;
}
