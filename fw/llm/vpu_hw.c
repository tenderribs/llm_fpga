// The work around the matmuls on the vector unit (src/sources/vpu.sv):
// overrides the CPU defaults in fw/model/model.c, same results bit for bit.
//
// V RAM (4096 words): SwiGLU's output at V_SILU, softmax scratch at V_SM,
// the matmul input x at V_X. Y: w1's results at 0, w3's at Y_W3.
#include <string.h>
#include "llm_hw.h"
#include "model.h"

enum { V_SILU = 0, V_SM = 2048, V_X = 3072, Y_W3 = 8192 };

extern const QTensor *g_xram;
void mm_run(uint32_t w_off, int n, int d, int yoff);
uint32_t prof_now(void);

void vpu_run(uint32_t op, int n, int src, int src2, int dst, uint32_t f) {
    LLM_VPU_N = (uint32_t)n;
    LLM_VPU_SRC = (uint32_t)src;
    LLM_VPU_SRC2 = (uint32_t)src2;
    LLM_VPU_DST = (uint32_t)dst;
    LLM_VPU_F = f;
    LLM_VPU_OP = op;
    while (LLM_STATUS & ST_MM_BUSY) {}
}

// x into the V RAM over the local bus, quantized into the matmul's x RAM
void mm_quant(QTensor *qx, const float *x, int n, int gs) {
    (void)gs;
    const uint32_t *xw = (const uint32_t *)x;
    for (int i = 0; i < n; i++) LLM_VL[V_X + i] = xw[i];
    vpu_run(VPU_QUANT | VPU_SRC_V, n, V_X, 0, 0, 0);
    g_xram = qx;
}

// w1 and w3 (x RAM: s->xq), SwiGLU and its quantize, all without the CPU
// touching the data: the x RAM ends up holding s->hq for w2
void ffn_hidden(const Model *m, RunState *s, int l) {
    int dim = m->c.dim, hidden = m->c.hidden_dim;
    mm_run(model_mat(m, l, MAT_W1), dim, hidden, 0);
    mm_run(model_mat(m, l, MAT_W3), dim, hidden, Y_W3);
    uint32_t t0 = prof_now();
    vpu_run(VPU_SILU, hidden, 0, Y_W3, V_SILU, 0);
    vpu_run(VPU_QUANT | VPU_SRC_V, hidden, V_SILU, 0, 0, 0);
    prof_cycles[PROF_FFN] += prof_now() - t0;
    g_xram = &s->hq;
}

// one head: q into the attention unit, scores, the vector unit's softmax
// straight into the unit's A RAM, weighted sum, xb[h] back
void att_head_hw(RunState *s, const Config *p, int h, int pos, float sqrt_hs,
                 uint32_t k_addr, uint32_t v_addr) {
    int hs = p->dim / p->n_heads, nt = pos + 1;
    const uint32_t *qw = (const uint32_t *)(s->q + h * hs);
    for (int i = 0; i < hs; i++) LLM_AT_A[i] = qw[i];
    LLM_AT_ADDR = k_addr;
    LLM_AT_CFG = (uint32_t)hs | ((uint32_t)nt << 16);
    LLM_AT_START = 0;
    while (LLM_STATUS & ST_MM_BUSY) {}
    uint32_t f;
    memcpy(&f, &sqrt_hs, 4);
    vpu_run(VPU_SOFTMAX, nt, 0, 0, V_SM, f);
    LLM_AT_ADDR = v_addr;
    LLM_AT_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
    uint32_t *xw = (uint32_t *)(s->xb + h * hs);
    for (int i = 0; i < hs; i++) xw[i] = LLM_YL[i];
}
