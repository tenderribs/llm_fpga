// llama2.c int8 (Q8_0) inference: runq.c's forward(), tokenizer and greedy
// sampler, reading the model from a model image (host/build_image.py).
//
// The same sources build for the board (fw/llm, matmuls on the engine) and
// for the host (fw/golden), so the board's output can be checked bit for bit.
// The float math runs in runq.c's order (no FMA: -ffp-contract=off); expf,
// sinf, cosf and powf come from fw/model/mathf.c unless MODEL_LIBM is defined,
// in which case the C library's are used and the host build must reproduce
// runq.c exactly.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char magic[8];  // "LLMMODEL"
    uint32_t version, total_bytes;
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, gs;
    uint32_t off_rms_att, off_rms_ffn, off_rms_final, off_emb_q, off_emb_s, off_tok, tok_bytes;
    uint32_t n_mats, reserved;
    uint32_t mat_off[];  // per layer wq wk wv wo w1 w2 w3, then the classifier
} ImageHeader;

enum { MAT_WQ, MAT_WK, MAT_WV, MAT_WO, MAT_W1, MAT_W2, MAT_W3, MATS_PER_LAYER };

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, gs;
} Config;

typedef struct {
    int8_t *q;
    float *s;
} QTensor;

typedef struct {
    Config c;
    const uint8_t *image;
    const float *rms_att, *rms_ffn, *rms_final;
    const int8_t *emb_q;
    const float *emb_s;
    const uint32_t *mat_off;
} Model;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float *key_cache, *value_cache;
    QTensor xq, hq;
} RunState;

// 0 on success, else a message
const char *model_open(Model *m, const uint8_t *image);
// offset of matrix `which` of layer l (l = n_layers: the classifier)
uint32_t model_mat(const Model *m, int l, int which);
int run_state_alloc(RunState *s, const Config *c);
float *forward(const Model *m, RunState *s, int token, int pos);
int sample_argmax(const float *p, int n);

// Provided by the platform: y[d] = W x, W the matrix at image offset w_off
// (qmv DDR format, n inputs, d outputs), x quantized in groups of 32.
void qmatmul(float *y, const QTensor *x, uint32_t w_off, int n, int d);

// ---- tokenizer (runq.c's)
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

const char *tokenizer_open(Tokenizer *t, const Model *m);
void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);
const char *decode(const Tokenizer *t, int prev_token, int token);
// the piece, or "" if it is a raw byte that isn't printable or whitespace
const char *printable(const char *piece);

// ---- float math (fw/model/mathf.c)
float m_expf(float x);
void m_sincosf(float x, float *s, float *c);
float m_roundf(float x);
