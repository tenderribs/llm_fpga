// runq.c's forward() over a model image (see model.h). The float operations
// are runq.c's, in runq.c's order; only where the weights come from and
// matmul() differ.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

#ifdef MODEL_LIBM
#define EXPF(x) expf(x)
static void EXPF4(const float *x, float *y) {
    for (int k = 0; k < 4; k++) y[k] = expf(x[k]);
}
#define ROUNDI(x) ((int32_t)round(x))
static void SINCOSF(float x, float *s, float *c) { *c = cosf(x); *s = sinf(x); }
#define ROPE_FREQ(e) (1.0f / powf(10000.0f, (e)))
#else
#define EXPF(x) m_expf(x)
#define EXPF4(x, y) m_expf4(x, y)
#if defined(__riscv)
// round half away from zero to an integer in one instruction (exact, so the
// same as m_roundf)
static inline int32_t ROUNDI(float x) {
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#else
#define ROUNDI(x) ((int32_t)m_roundf(x))
#endif
#define SINCOSF(x, s, c) m_sincosf(x, s, c)
// 10000^-e = e^(-e ln 10000)
#define ROPE_FREQ(e) (1.0f / m_expf((e) * 9.21034049987793f))
#endif

// The loops below work on four independent elements at a time where they
// can, so the CPU's pipelined FPU overlaps them; each element still gets
// runq.c's operations, and every sum still adds in runq.c's order.

#ifdef MODEL_PROFILE
uint64_t prof_cycles[PROF_N];
#define PROF_BEGIN() uint32_t prof_t_ = prof_now()
#define PROF_END(sec) (prof_cycles[sec] += prof_now() - prof_t_)
#define PROF(sec, ...) do { PROF_BEGIN(); __VA_ARGS__; PROF_END(sec); } while (0)
#else
#define PROF_BEGIN() do {} while (0)
#define PROF_END(sec) do {} while (0)
#define PROF(sec, ...) do { __VA_ARGS__; } while (0)
#endif

const char *model_open(Model *m, const uint8_t *image) {
    const ImageHeader *h = (const ImageHeader *)image;
    if (memcmp(h->magic, "LLMMODEL", 8) != 0) return "no model image (bad magic)";
    if (h->version != 1) return "unsupported model image version";
    m->image = image;
    m->c = (Config){h->dim, h->hidden_dim, h->n_layers, h->n_heads, h->n_kv_heads,
                    h->vocab_size, h->seq_len, h->gs};
    if (m->c.gs != 32) return "group size must be 32 (the engine's)";
    if (h->n_mats != (uint32_t)(m->c.n_layers * MATS_PER_LAYER + 1)) return "bad matrix count";
    m->rms_att = (const float *)(image + h->off_rms_att);
    m->rms_ffn = (const float *)(image + h->off_rms_ffn);
    m->rms_final = (const float *)(image + h->off_rms_final);
    m->emb_q = (const int8_t *)(image + h->off_emb_q);
    m->emb_s = (const float *)(image + h->off_emb_s);
    m->mat_off = h->mat_off;
    return NULL;
}

uint32_t model_mat(const Model *m, int l, int which) {
    return m->mat_off[l < m->c.n_layers ? l * MATS_PER_LAYER + which : m->c.n_layers * MATS_PER_LAYER];
}

// The platform may put the small, hot buffers in faster memory; the KV cache
// and the logits always come from calloc.
__attribute__((weak)) void *fast_calloc(size_t n, size_t size) { return calloc(n, size); }

int run_state_alloc(RunState *s, const Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = fast_calloc(p->dim, sizeof(float));
    s->xb = fast_calloc(p->dim, sizeof(float));
    s->xb2 = fast_calloc(p->dim, sizeof(float));
    s->hb = fast_calloc(p->hidden_dim, sizeof(float));
    s->hb2 = fast_calloc(p->hidden_dim, sizeof(float));
    s->xq = (QTensor){fast_calloc(p->dim, 1), fast_calloc(p->dim, sizeof(float))};
    s->hq = (QTensor){fast_calloc(p->hidden_dim, 1), fast_calloc(p->hidden_dim, sizeof(float))};
    s->q = fast_calloc(p->dim, sizeof(float));
    s->k = fast_calloc(kv_dim, sizeof(float));
    s->v = fast_calloc(kv_dim, sizeof(float));
    s->att = fast_calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // RoPE's frequency depends only on the pair's place in the head, and its
    // sin / cos only on that and the position: computed once, and once per
    // token, instead of in every layer
    int head_size = p->dim / p->n_heads;
    s->rope_freq = fast_calloc(head_size / 2, sizeof(float));
    s->rope_cos = fast_calloc(head_size / 2, sizeof(float));
    s->rope_sin = fast_calloc(head_size / 2, sizeof(float));
    if (!s->rope_freq || !s->rope_cos || !s->rope_sin) return 0;
    for (int k = 0; k < head_size / 2; k++) {
        int head_dim = 2 * k;
        s->rope_freq[k] = ROPE_FREQ(head_dim / (float)head_size);
    }
    return s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->xq.q && s->xq.s && s->hq.q &&
           s->hq.s && s->q && s->k && s->v && s->att && s->logits && s->key_cache &&
           s->value_cache;
}

static void rmsnorm(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    int j = 0;
    for (; j + 4 <= size; j += 4) {  // o may be x: read all four first
        float a0 = weight[j] * (ss * x[j]), a1 = weight[j + 1] * (ss * x[j + 1]);
        float a2 = weight[j + 2] * (ss * x[j + 2]), a3 = weight[j + 3] * (ss * x[j + 3]);
        o[j] = a0;
        o[j + 1] = a1;
        o[j + 2] = a2;
        o[j + 3] = a3;
    }
    for (; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void div4(float *x, int size, float d) {  // x[i] /= d
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float a0 = x[i] / d, a1 = x[i + 1] / d, a2 = x[i + 2] / d, a3 = x[i + 3] / d;
        x[i] = a0;
        x[i + 1] = a1;
        x[i + 2] = a2;
        x[i + 3] = a3;
    }
    for (; i < size; i++) x[i] /= d;
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        float d[4] = {x[i] - max_val, x[i + 1] - max_val, x[i + 2] - max_val, x[i + 3] - max_val};
        EXPF4(d, x + i);
        sum += x[i];
        sum += x[i + 1];
        sum += x[i + 2];
        sum += x[i + 3];
    }
    for (; i < size; i++) {
        x[i] = EXPF(x[i] - max_val);
        sum += x[i];
    }
    div4(x, size, sum);
}

static void quantize(QTensor *qx, const float *x, int n, int gs) {
    for (int group = 0; group < n / gs; group++) {
        // the largest |x| (exact, so any order gives the same; fabsf is
        // never -0 or NaN here, where fmaxf equals runq.c's compare)
        float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
        for (int i = 0; i < gs; i += 4) {
            m0 = fmaxf(m0, fabsf(x[group * gs + i]));
            m1 = fmaxf(m1, fabsf(x[group * gs + i + 1]));
            m2 = fmaxf(m2, fabsf(x[group * gs + i + 2]));
            m3 = fmaxf(m3, fabsf(x[group * gs + i + 3]));
        }
        float wmax = fmaxf(fmaxf(m0, m1), fmaxf(m2, m3));
        float scale = wmax / 127.0f;
        qx->s[group] = scale;
        const float *xg = x + group * gs;
        int8_t *qg = qx->q + group * gs;
        for (int i = 0; i < gs; i += 4) {  // gs is a multiple of 4 (32)
            float a0 = xg[i] / scale, a1 = xg[i + 1] / scale;
            float a2 = xg[i + 2] / scale, a3 = xg[i + 3] / scale;
            qg[i] = (int8_t)ROUNDI(a0);
            qg[i + 1] = (int8_t)ROUNDI(a1);
            qg[i + 2] = (int8_t)ROUNDI(a2);
            qg[i + 3] = (int8_t)ROUNDI(a3);
        }
    }
}

// ---- attention on the CPU, over the KV cache in the run state: the
// default for the platform hooks (the board overrides them with the
// attention unit). Four timesteps at a time: independent sums the CPU can
// overlap; every sum still adds its terms in runq.c's order.

__attribute__((weak)) void att_store_kv(RunState *s, const Config *p, int l, int pos) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int loff = l * p->seq_len * kv_dim;
    memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
    memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));
}

__attribute__((weak)) void att_scores(RunState *s, const Config *p, int l, int kvh,
                                      const float *q, float *att, int nt) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    const float *kh = s->key_cache + l * p->seq_len * kv_dim + kvh * head_size;
    int t = 0;
    for (; t + 4 <= nt; t += 4) {
        const float *k0 = kh + t * kv_dim, *k1 = k0 + kv_dim, *k2 = k1 + kv_dim,
                    *k3 = k2 + kv_dim;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (int i = 0; i < head_size; i++) {
            s0 += q[i] * k0[i];
            s1 += q[i] * k1[i];
            s2 += q[i] * k2[i];
            s3 += q[i] * k3[i];
        }
        att[t] = s0;
        att[t + 1] = s1;
        att[t + 2] = s2;
        att[t + 3] = s3;
    }
    for (; t < nt; t++) {
        const float *k = kh + t * kv_dim;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += q[i] * k[i];
        att[t] = score;
    }
}

__attribute__((weak)) void att_wsum(RunState *s, const Config *p, int l, int kvh,
                                    const float *att, float *xb, int nt) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = p->dim / p->n_heads;
    const float *vh = s->value_cache + l * p->seq_len * kv_dim + kvh * head_size;
    memset(xb, 0, head_size * sizeof(float));
    int t = 0;
    for (; t + 4 <= nt; t += 4) {
        const float *v0 = vh + t * kv_dim, *v1 = v0 + kv_dim, *v2 = v1 + kv_dim,
                    *v3 = v2 + kv_dim;
        float a0 = att[t], a1 = att[t + 1], a2 = att[t + 2], a3 = att[t + 3];
        for (int i = 0; i < head_size; i++)
            xb[i] = (((xb[i] + a0 * v0[i]) + a1 * v1[i]) + a2 * v2[i]) + a3 * v3[i];
    }
    for (; t < nt; t++) {
        const float *v = vh + t * kv_dim;
        float a = att[t];
        for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
    }
}

float *forward(const Model *m, RunState *s, int token, int pos) {
    const Config *p = &m->c;
    float *x = s->x;
    int dim = p->dim, gs = p->gs;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // the token's embedding row, dequantized (runq.c dequantizes the whole
    // table up front, with the same multiply)
    const int8_t *eq = m->emb_q + (size_t)token * dim;
    const float *es = m->emb_s + (size_t)token * dim / gs;
    for (int i = 0; i < dim; i++) x[i] = eq[i] * es[i / gs];

    // RoPE's sin / cos for this position (the same in every layer)
    PROF(PROF_ROPE, for (int k = 0; k < head_size / 2; k++) {
        float val = pos * s->rope_freq[k];
        SINCOSF(val, &s->rope_sin[k], &s->rope_cos[k]);
    });

    for (int l = 0; l < p->n_layers; l++) {
        PROF(PROF_NORM, rmsnorm(s->xb, x, m->rms_att + l * dim, dim));

        PROF(PROF_QUANT, quantize(&s->xq, s->xb, dim, gs));
        PROF(PROF_MM, qmatmul(s->q, &s->xq, model_mat(m, l, MAT_WQ), dim, dim));
        PROF(PROF_MM, qmatmul(s->k, &s->xq, model_mat(m, l, MAT_WK), dim, kv_dim));
        PROF(PROF_MM, qmatmul(s->v, &s->xq, model_mat(m, l, MAT_WV), dim, kv_dim));

        // RoPE
        {
        PROF_BEGIN();
        for (int i = 0; i < dim; i += 2) {
            int k = (i % head_size) / 2;
            float fcr = s->rope_cos[k], fci = s->rope_sin[k];
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        PROF_END(PROF_ROPE);
        }

        {
        PROF_BEGIN();
        att_store_kv(s, p, l, pos);
        float sqrt_hs = sqrtf(head_size);
        for (int h = 0; h < p->n_heads; h++) {
            float *att = s->att + h * p->seq_len;
            att_scores(s, p, l, h / kv_mul, s->q + h * head_size, att, pos + 1);
            div4(att, pos + 1, sqrt_hs);
            softmax(att, pos + 1);
            att_wsum(s, p, l, h / kv_mul, att, s->xb + h * head_size, pos + 1);
        }
        PROF_END(PROF_ATT);
        }

        PROF(PROF_QUANT, quantize(&s->xq, s->xb, dim, gs));
        PROF(PROF_MM, qmatmul(s->xb2, &s->xq, model_mat(m, l, MAT_WO), dim, dim));
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        PROF(PROF_NORM, rmsnorm(s->xb, x, m->rms_ffn + l * dim, dim));

        PROF(PROF_QUANT, quantize(&s->xq, s->xb, dim, gs));
        PROF(PROF_MM, qmatmul(s->hb, &s->xq, model_mat(m, l, MAT_W1), dim, hidden_dim));
        PROF(PROF_MM, qmatmul(s->hb2, &s->xq, model_mat(m, l, MAT_W3), dim, hidden_dim));

        // SwiGLU
        {
        PROF_BEGIN();
        int i = 0;
        for (; i + 4 <= hidden_dim; i += 4) {
            float nv[4] = {-s->hb[i], -s->hb[i + 1], -s->hb[i + 2], -s->hb[i + 3]}, e[4];
            EXPF4(nv, e);
            float d0 = 1.0f / (1.0f + e[0]), d1 = 1.0f / (1.0f + e[1]);
            float d2 = 1.0f / (1.0f + e[2]), d3 = 1.0f / (1.0f + e[3]);
            float v0 = s->hb[i] * d0, v1 = s->hb[i + 1] * d1;
            float v2 = s->hb[i + 2] * d2, v3 = s->hb[i + 3] * d3;
            s->hb[i] = v0 * s->hb2[i];
            s->hb[i + 1] = v1 * s->hb2[i + 1];
            s->hb[i + 2] = v2 * s->hb2[i + 2];
            s->hb[i + 3] = v3 * s->hb2[i + 3];
        }
        for (; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + EXPF(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        PROF_END(PROF_FFN);
        }

        PROF(PROF_QUANT, quantize(&s->hq, s->hb, hidden_dim, gs));
        PROF(PROF_MM, qmatmul(s->xb, &s->hq, model_mat(m, l, MAT_W2), hidden_dim, dim));
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    PROF(PROF_NORM, rmsnorm(x, x, m->rms_final, dim));
    PROF(PROF_QUANT, quantize(&s->xq, x, dim, gs));
    PROF(PROF_MM, qmatmul(s->logits, &s->xq, model_mat(m, p->n_layers, 0), dim, p->vocab_size));
    return s->logits;
}

int sample_argmax(const float *p, int n) {
    int max_i = 0;
    float max_p = p[0];
    for (int i = 1; i < n; i++) {
        if (p[i] > max_p) {
            max_i = i;
            max_p = p[i];
        }
    }
    return max_i;
}
