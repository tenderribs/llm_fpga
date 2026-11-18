// Memory latency / bandwidth probe for top_soc's MicroBlaze V: reads through
// the data cache with different strides, the uncached port, and local memory.
// Prints CPU cycles per access ("MEMBENCH DONE" at the end).
#include <stdint.h>
#include <stdio.h>
#include "llm_hw.h"

static inline uint32_t rdcycle(void) {
    uint32_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

static uint32_t lmb_buf[8192];

static void run(const char *what, volatile uint32_t *p, uint32_t n, uint32_t stride_words) {
    uint32_t sum = 0, t0 = rdcycle();
    for (uint32_t i = 0; i < n; i++) sum += p[i * stride_words];
    uint32_t dt = rdcycle() - t0;
    printf("%-34s %6lu reads: %6.1f cycles each (sum %08lx)\n", what, (unsigned long)n,
           (double)dt / n, (unsigned long)sum);
}

int main(void) {
    printf("\n=== membench ===\n");
    volatile uint32_t *ddr = (volatile uint32_t *)(DDR_BASE + 0x08000000u);  // cold region
    run("DDR cached, stride 4 B", ddr, 16384, 1);
    run("DDR cached, stride 16 B", ddr + 0x100000, 16384, 4);
    run("DDR cached, stride 32 B", ddr + 0x200000, 16384, 8);
    run("DDR cached, stride 64 B", ddr + 0x300000, 16384, 16);
    run("DDR cached, stride 4 B (warm 4 KB)", ddr, 1024, 1);
    run("DDR uncached, stride 4 B", (volatile uint32_t *)(DDR_UNCACHED + 0x100000u), 4096, 1);
    run("llm_core register (CYCLES)", &LLM_CYCLES, 4096, 0);
    run("local memory, stride 4 B", lmb_buf, 8192, 1);
    uint32_t t0 = rdcycle();
    for (uint32_t i = 0; i < 4096; i++) ddr[0x400000 + i * 8] = i;
    printf("%-34s %6d writes: %6.1f cycles each\n", "DDR cached store, stride 32 B", 4096,
           (double)(rdcycle() - t0) / 4096);
    printf("MEMBENCH DONE\n");
    for (;;) {}
}
