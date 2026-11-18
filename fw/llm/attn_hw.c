// Attention on the attention unit (src/sources/attn.sv) and the vector unit:
// overrides the CPU defaults in fw/model/model.c, same results bit for bit.
//
// The KV cache lives in DDR at offset KV_DDR (512 MB). Per layer and KV
// head, 4 KB aligned:
//   K: blocks of 128 timesteps, each [i][t % 128] (a beat = 8 timesteps' i)
//   V: rows [t][i]
// Each layer's k, v and q go to the V RAM over the local bus; DMA jobs write
// k and v into the cache (K's words 512 bytes apart, with byte strobes), and
// the vector unit copies each head's q into the attention unit.
#include <stdint.h>
#include <string.h>
#include "llm_hw.h"
#include "model.h"

#define KV_DDR 0x20000000u  // engine address
enum { V_K = 4096, V_V = 5120, V_Q = 6144, V_SM = 2048 };  // see vpu_hw.c

void vpu_run(uint32_t op, int n, int src, int src2, int dst, uint32_t f);

static uint32_t align4k(uint32_t x) { return (x + 4095u) & ~4095u; }

static uint32_t k_slot(const Config *p) {
    int hs = p->dim / p->n_heads;
    return align4k((uint32_t)((p->seq_len + 127) / 128) * 128u * (uint32_t)hs * 4u);
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

static void dma(int src, int n, uint32_t dst, uint32_t stride) {
    LLM_DMA_SRC = (uint32_t)src;
    LLM_DMA_N = (uint32_t)n;
    LLM_DMA_DST = dst;
    LLM_DMA_STRIDE = stride;
    LLM_DMA_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
}

uint64_t g_kv_cycles;  // profile: storing k / v (and staging q)

void att_store_kv(RunState *s, const Config *p, int l, int pos) {
    uint32_t t0 = LLM_CYCLES;
    int hs = p->dim / p->n_heads, kv_dim = p->n_kv_heads * hs;
    const uint32_t *k = (const uint32_t *)s->k, *v = (const uint32_t *)s->v;
    const uint32_t *q = (const uint32_t *)s->q;
    for (int i = 0; i < kv_dim; i++) {
        LLM_VL[V_K + i] = k[i];
        LLM_VL[V_V + i] = v[i];
    }
    for (int i = 0; i < p->dim; i++) LLM_VL[V_Q + i] = q[i];
    uint32_t koff = (uint32_t)((pos / 128) * hs * 128 + pos % 128) * 4u;
    for (int h = 0; h < p->n_kv_heads; h++) {
        dma(V_K + h * hs, hs, k_addr(p, l, h) + koff, 128 * 4);
        dma(V_V + h * hs, hs, v_addr(p, l, h) + (uint32_t)(pos * hs) * 4u, 4);
    }
    g_kv_cycles += LLM_CYCLES - t0;
}

// one head: q into the attention unit, scores, the vector unit's softmax
// straight into the unit's A RAM, weighted sum, xb[h] back
void att_head(RunState *s, const Config *p, int l, int h, int kvh, int pos, float sqrt_hs) {
    int hs = p->dim / p->n_heads, nt = pos + 1;
    vpu_run(VPU_COPY, hs, V_Q + h * hs, 0, 0, 0);
    LLM_AT_ADDR = k_addr(p, l, kvh);
    LLM_AT_CFG = (uint32_t)hs | ((uint32_t)nt << 16);
    LLM_AT_START = 0;
    while (LLM_STATUS & ST_MM_BUSY) {}
    uint32_t f;
    memcpy(&f, &sqrt_hs, 4);
    vpu_run(VPU_SOFTMAX, nt, 0, 0, V_SM, f);
    LLM_AT_ADDR = v_addr(p, l, kvh);
    LLM_AT_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
    uint32_t *xw = (uint32_t *)(s->xb + h * hs);
    for (int i = 0; i < hs; i++) xw[i] = LLM_YL[i];
}
