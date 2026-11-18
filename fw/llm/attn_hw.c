// Attention on the attention unit (src/sources/attn.sv): overrides the CPU
// defaults in fw/model/model.c, same results bit for bit.
//
// The KV cache lives in DDR at offset KV_DDR (512 MB), which the CPU reaches
// uncached, so what the unit reads is what the CPU wrote (the data cache is
// write-back). Per layer and KV head, 4 KB aligned:
//   K: blocks of 16 timesteps, each [i][t % 16] (a beat = 8 timesteps' i)
//   V: rows [t][i]
#include <stdint.h>
#include "llm_hw.h"
#include "model.h"

#define KV_DDR 0x20000000u  // engine address; CPU: DDR_UNCACHED

static uint32_t align4k(uint32_t x) { return (x + 4095u) & ~4095u; }

static uint32_t k_slot(const Config *p) {
    int hs = p->dim / p->n_heads;
    return align4k((uint32_t)((p->seq_len + 15) / 16) * 16u * (uint32_t)hs * 4u);
}

static uint32_t v_slot(const Config *p) {
    int hs = p->dim / p->n_heads;
    return align4k((uint32_t)p->seq_len * (uint32_t)hs * 4u);
}

static uint32_t k_addr(const Config *p, int l, int kvh) {
    return KV_DDR + (uint32_t)(l * p->n_kv_heads + kvh) * k_slot(p);
}

static uint32_t v_addr(const Config *p, int l, int kvh) {
    return KV_DDR + (uint32_t)(p->n_layers * p->n_kv_heads) * k_slot(p) +
           (uint32_t)(l * p->n_kv_heads + kvh) * v_slot(p);
}

static volatile uint32_t *cpu_ptr(uint32_t ddr) {
    return (volatile uint32_t *)(DDR_UNCACHED + (ddr - KV_DDR));
}

void att_store_kv(RunState *s, const Config *p, int l, int pos) {
    int hs = p->dim / p->n_heads;
    const uint32_t *k = (const uint32_t *)s->k, *v = (const uint32_t *)s->v;
    for (int h = 0; h < p->n_kv_heads; h++) {
        volatile uint32_t *kb = cpu_ptr(k_addr(p, l, h)) + (pos / 16) * 16 * hs + pos % 16;
        volatile uint32_t *vb = cpu_ptr(v_addr(p, l, h)) + pos * hs;
        for (int i = 0; i < hs; i++) {
            kb[i * 16] = k[h * hs + i];
            vb[i] = v[h * hs + i];
        }
    }
}

static void run(uint32_t addr, int hs, int nt, int mode) {
    LLM_AT_ADDR = addr;
    LLM_AT_CFG = (uint32_t)hs | ((uint32_t)nt << 16);
    LLM_AT_START = (uint32_t)mode;
    while (LLM_STATUS & ST_MM_BUSY) {}
}

void att_scores(RunState *s, const Config *p, int l, int kvh, const float *q, float *att, int nt) {
    (void)s;
    int hs = p->dim / p->n_heads;
    const uint32_t *qw = (const uint32_t *)q;
    for (int i = 0; i < hs; i++) LLM_AT_A[i] = qw[i];
    run(k_addr(p, l, kvh), hs, nt, 0);
    uint32_t *aw = (uint32_t *)att;
    for (int t = 0; t < nt; t++) aw[t] = LLM_YL[t];
}

void att_wsum(RunState *s, const Config *p, int l, int kvh, const float *att, float *xb, int nt) {
    (void)s;
    int hs = p->dim / p->n_heads;
    const uint32_t *aw = (const uint32_t *)att;
    for (int t = 0; t < nt; t++) LLM_AT_A[t] = aw[t];
    run(v_addr(p, l, kvh), hs, nt, 1);
    uint32_t *xw = (uint32_t *)xb;
    for (int i = 0; i < hs; i++) xw[i] = LLM_YL[i];
}
