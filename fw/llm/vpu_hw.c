// The work around the matmuls on the vector unit (src/sources/vpu.sv):
// overrides the CPU defaults in fw/model/model.c, same results bit for bit.
//
// V RAM (8192 words): SwiGLU's output at V_SILU, softmax scratch at V_SM,
// the matmul input x at V_X, k / v / q at 4096 / 5120 / 6144 (attn_hw.c).
// Y: w1's results at 0, w3's at Y_W3.
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
