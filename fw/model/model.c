// runq.c's forward() over a model image (see model.h). The float operations
// are runq.c's, in runq.c's order; only where the weights come from and
// matmul() differ.
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

#ifdef MODEL_LIBM
#define EXPF(x) expf(x)
#define ROUNDF(x) round(x)
static void SINCOSF(float x, float *s, float *c) { *c = cosf(x); *s = sinf(x); }
#define ROPE_FREQ(e) (1.0f / powf(10000.0f, (e)))
#else
#define EXPF(x) m_expf(x)
#define ROUNDF(x) m_roundf(x)
#define SINCOSF(x, s, c) m_sincosf(x, s, c)
// 10000^-e = e^(-e ln 10000)
#define ROPE_FREQ(e) (1.0f / m_expf((e) * 9.21034049987793f))
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

int run_state_alloc(RunState *s, const Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QTensor){calloc(p->dim, 1), calloc(p->dim, sizeof(float))};
    s->hq = (QTensor){calloc(p->hidden_dim, 1), calloc(p->hidden_dim, sizeof(float))};
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc((size_t)p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc((size_t)p->n_layers * p->seq_len * kv_dim, sizeof(float));
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
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}

static void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++)
        if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = EXPF(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

static void quantize(QTensor *qx, const float *x, int n, int gs) {
    for (int group = 0; group < n / gs; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < gs; i++) {
            float val = fabsf(x[group * gs + i]);
            if (val > wmax) wmax = val;
        }
        float scale = wmax / 127.0f;
        qx->s[group] = scale;
        for (int i = 0; i < gs; i++) {
            float quant_value = x[group * gs + i] / scale;
            qx->q[group * gs + i] = (int8_t)ROUNDF(quant_value);
        }
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

    for (int l = 0; l < p->n_layers; l++) {
        rmsnorm(s->xb, x, m->rms_att + l * dim, dim);

        quantize(&s->xq, s->xb, dim, gs);
        qmatmul(s->q, &s->xq, model_mat(m, l, MAT_WQ), dim, dim);
        qmatmul(s->k, &s->xq, model_mat(m, l, MAT_WK), dim, kv_dim);
        qmatmul(s->v, &s->xq, model_mat(m, l, MAT_WV), dim, kv_dim);

        // RoPE
        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = ROPE_FREQ(head_dim / (float)head_size);
            float val = pos * freq;
            float fcr, fci;
            SINCOSF(val, &fci, &fcr);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        int loff = l * p->seq_len * kv_dim;
        memcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float));
        memcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float));

        for (int h = 0; h < p->n_heads; h++) {
            const float *q = s->q + h * head_size;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                const float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) score += q[i] * k[i];
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) xb[i] += a * v[i];
            }
        }

        quantize(&s->xq, s->xb, dim, gs);
        qmatmul(s->xb2, &s->xq, model_mat(m, l, MAT_WO), dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb2[i];

        rmsnorm(s->xb, x, m->rms_ffn + l * dim, dim);

        quantize(&s->xq, s->xb, dim, gs);
        qmatmul(s->hb, &s->xq, model_mat(m, l, MAT_W1), dim, hidden_dim);
        qmatmul(s->hb2, &s->xq, model_mat(m, l, MAT_W3), dim, hidden_dim);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + EXPF(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        quantize(&s->hq, s->hb, hidden_dim, gs);
        qmatmul(s->xb, &s->hq, model_mat(m, l, MAT_W2), hidden_dim, dim);
        for (int i = 0; i < dim; i++) x[i] += s->xb[i];
    }

    rmsnorm(x, x, m->rms_final, dim);
    quantize(&s->xq, x, dim, gs);
    qmatmul(s->logits, &s->xq, model_mat(m, p->n_layers, 0), dim, p->vocab_size);
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
