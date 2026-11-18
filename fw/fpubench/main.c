// FPU cost probe for top_soc's MicroBlaze V: CPU cycles per operation for
// dependent chains (latency) and independent streams (throughput), plus the
// model's float helpers. Ends with "FPUBENCH DONE".
#include <stdint.h>
#include <stdio.h>
#include "llm_hw.h"
#include "model.h"

static inline uint32_t rdcycle(void) {
    uint32_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

#define N 1024
static volatile float sink;
static float buf[N];

static void report(const char *what, uint32_t dt, int ops) {
    printf("%-40s %6.1f cycles\n", what, (double)dt / ops);
}

int main(void) {
    printf("\n=== fpubench ===\n");
    for (int i = 0; i < N; i++) buf[i] = 0.001f * (float)(i + 1) - 0.5f;
    volatile float vin = 1.0001f;
    float a = vin, b = vin, c = vin, d = vin, x = vin;
    uint32_t t0, dt;

    t0 = rdcycle();
    for (int i = 0; i < N; i++) a = a + x;
    dt = rdcycle() - t0; sink = a; report("fadd, dependent", dt, N);

    a = b = c = d = vin;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) { a = a + x; b = b + x; c = c + x; d = d + x; }
    dt = rdcycle() - t0; sink = a + b + c + d; report("fadd, 4 independent (per op)", dt, 4 * N);

    a = vin;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) a = a * x;
    dt = rdcycle() - t0; sink = a; report("fmul, dependent", dt, N);

    a = vin;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) a = a / x;
    dt = rdcycle() - t0; sink = a; report("fdiv, dependent", dt, N);

    a = b = c = d = vin;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) { a = a / x; b = b / x; c = c / x; d = d / x; }
    dt = rdcycle() - t0; sink = a + b + c + d; report("fdiv, 4 independent (per op)", dt, 4 * N);

    a = vin;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) a = __builtin_sqrtf(a + 1.0f);
    dt = rdcycle() - t0; sink = a; report("fsqrt + fadd, dependent", dt, N);

    t0 = rdcycle();
    for (int i = 0; i < N; i++) sink = m_expf(buf[i]);
    dt = rdcycle() - t0; report("m_expf", dt, N);

    static float out[N];
    t0 = rdcycle();
    for (int i = 0; i < N; i += 4) m_expf4(buf + i, out + i);
    dt = rdcycle() - t0; sink = out[7]; report("m_expf4 (per value)", dt, N);

    t0 = rdcycle();
    for (int i = 0; i < N; i++) sink = m_roundf(buf[i] * 200.0f);
    dt = rdcycle() - t0; report("m_roundf (+ fmul)", dt, N);

    int32_t acc = 0;
    t0 = rdcycle();
    for (int i = 0; i < N; i++) {
        int32_t r;
        float v = buf[i] * 200.0f;
        __asm__ volatile("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(v));
        acc += r;
    }
    dt = rdcycle() - t0; sink = (float)acc; report("fcvt.w.s rmm (+ fmul)", dt, N);

    // check rmm == m_roundf over a range
    int bad = 0;
    for (int i = -300000; i <= 300000; i++) {
        float v = (float)i * 0.000437f;
        int32_t r;
        __asm__ volatile("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(v));
        if ((float)r != m_roundf(v)) bad++;
    }
    float halves[] = {0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 126.5f, -126.5f, 0.49999997f};
    for (unsigned i = 0; i < sizeof(halves) / 4; i++) {
        int32_t r;
        __asm__ volatile("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(halves[i]));
        if ((float)r != m_roundf(halves[i])) bad++;
    }
    printf("fcvt.w.s rmm vs m_roundf: %d differences\n", bad);
    printf("FPUBENCH DONE\n");
    for (;;) {}
}
