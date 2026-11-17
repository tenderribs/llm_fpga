// Bring-up test for top_soc: the CPU, its FPU, DDR3 (cached and uncached),
// and the matmul engine. Prints a report on the console (UART) and ends with
// "BRINGUP PASS" or "BRINGUP FAIL".
//
// The matmul test needs the stories15M image on the microSD card (stored
// with host/llm_host -s): it runs layer 0's wq (288x288, at DDR offset 0) on
// the engine, then the same matmul in software on the CPU's FPU, the way
// runq.c does it, and compares the two bit for bit.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "llm_hw.h"

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%-44s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) failures++;
}

static inline uint32_t rdcycle(void) {
    uint32_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

// ---- qmv DDR format (see src/sources/wstream_reader.sv): beats in the order
// (block of 16 rows, group, row in block), in chunks of 8 beats: one beat of
// the 8 scales, then the 8 weight beats. Group size 32 = one beat per group.
static void w_group(const uint8_t *w, int n, int row, int g, const int8_t **q, float *s) {
    int G = n / 32;
    uint32_t b = ((uint32_t)(row / 16) * G + g) * 16 + row % 16;
    const uint8_t *chunk = w + (b / 8) * 9 * 32;
    memcpy(s, chunk + (b % 8) * 4, 4);
    *q = (const int8_t *)(chunk + 32 + (b % 8) * 32);
}

// runq.c matmul() on the CPU
static void matmul_sw(float *y, const int8_t *xq, const float *xs, const uint8_t *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int g = 0; g < n / 32; g++) {
            const int8_t *q;
            float ws;
            w_group(w, n, i, g, &q, &ws);
            int32_t ival = 0;
            for (int k = 0; k < 32; k++) ival += (int32_t)xq[g * 32 + k] * (int32_t)q[k];
            val += ((float)ival * ws) * xs[g];
        }
        y[i] = val;
    }
}

static void matmul_hw(float *y, const int8_t *xq, const float *xs, uint32_t ddr_off, int n, int d) {
    for (int i = 0; i < n / 4; i++) {
        uint32_t word;
        memcpy(&word, xq + 4 * i, 4);
        LLM_XQ[i] = word;
    }
    for (int g = 0; g < n / 32; g++) {
        uint32_t word;
        memcpy(&word, &xs[g], 4);
        LLM_XS[g] = word;
    }
    LLM_MM_ADDR = ddr_off;
    LLM_MM_ND = (uint32_t)n | ((uint32_t)d << 16);
    LLM_MM_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
    for (int i = 0; i < d; i++) {
        uint32_t word = LLM_Y[i];
        memcpy(&y[i], &word, 4);
    }
}

int main(void) {
    printf("\n=== top_soc bring-up: MicroBlaze V ===\n");
    check(LLM_ID == 0x314D4C4Cu, "llm_core ID register");

    // CPU clock vs the core's cycle counter (both should be the 225 MHz clock)
    uint32_t c0 = rdcycle(), t0 = LLM_CYCLES;
    for (volatile int i = 0; i < 100000; i++) {}
    uint32_t c1 = rdcycle(), t1 = LLM_CYCLES;
    printf("CPU cycles %lu over %lu fabric cycles\n", (unsigned long)(c1 - c0),
           (unsigned long)(t1 - t0));

    // FPU sanity: 1/3 rounds to nearest even, sqrt is correctly rounded
    volatile float a = 1.0f, b = 3.0f, two = 2.0f;
    float third = a / b, root2 = __builtin_sqrtf(two);
    uint32_t tb, rb;
    memcpy(&tb, &third, 4);
    memcpy(&rb, &root2, 4);
    check(tb == 0x3EAAAAABu && rb == 0x3FB504F3u, "FPU: 1/3 and sqrt(2) bit patterns");

    // DDR through the data cache (write-through) and through the uncached port
    volatile uint32_t *cached = (volatile uint32_t *)(DDR_BASE + 0x1F000000u);
    volatile uint32_t *uncached = (volatile uint32_t *)(DDR_UNCACHED + 0x00100000u);
    int bad = 0;
    for (uint32_t i = 0; i < 65536; i++) cached[i] = i * 2654435761u;
    for (uint32_t i = 0; i < 65536; i++) bad += cached[i] != i * 2654435761u;
    check(bad == 0, "DDR cached: 256 KB write / read");
    bad = 0;
    for (uint32_t i = 0; i < 1024; i++) uncached[i] = ~(i * 40503u);
    for (uint32_t i = 0; i < 1024; i++) bad += uncached[i] != ~(i * 40503u);
    check(bad == 0, "DDR uncached: 4 KB write / read");

    // the SD image
    uint32_t st;
    while ((st = LLM_STATUS) & ST_SD_BUSY) {}
    printf("SD: %s, image %lu bytes at DDR 0x%08lx\n",
           (st & ST_SD_OK) ? "loaded" : "not loaded", (unsigned long)LLM_SD_BYTES,
           (unsigned long)LLM_SD_ADDR);
    if (st & ST_SD_OK) {
        enum { N = 288, D = 288 };
        static int8_t xq[N];
        static float xs[N / 32], y_hw[D], y_sw[D];
        for (int i = 0; i < N; i++) xq[i] = (int8_t)((i * 37 + 11) % 255 - 127);
        for (int g = 0; g < N / 32; g++) xs[g] = 0.01f * (float)(g + 1);

        uint32_t h0 = LLM_CYCLES;
        matmul_hw(y_hw, xq, xs, 0, N, D);
        uint32_t h1 = LLM_CYCLES;
        matmul_sw(y_sw, xq, xs, (const uint8_t *)DDR_BASE, N, D);
        uint32_t h2 = LLM_CYCLES;

        int diff = 0;
        for (int i = 0; i < D; i++) diff += memcmp(&y_hw[i], &y_sw[i], 4) != 0;
        printf("matmul 288x288: engine %lu cycles (incl. register I/O), CPU %lu cycles\n",
               (unsigned long)(h1 - h0), (unsigned long)(h2 - h1));
        printf("y[0..2] = %.6f %.6f %.6f\n", (double)y_hw[0], (double)y_hw[1], (double)y_hw[2]);
        check(diff == 0, "matmul: engine == CPU, bit for bit");
    } else {
        printf("(no SD image: matmul test skipped)\n");
    }

    printf("BRINGUP %s\n", failures ? "FAIL" : "PASS");
    return 0;
}
