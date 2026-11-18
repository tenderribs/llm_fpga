// Llama 2 inference on top_soc: runq.c's forward() (fw/model) on the
// MicroBlaze V, matmuls on the int8 engine, attention on the attention unit,
// the model image in DDR3 (from the microSD card at boot, or uploaded with
// host/image_tool.py).
//
// Interactive: waits for a line from the host (CONSOLE_IN, host/chat.py),
// generates greedily from it as the prompt, prints the story, the speed and
// "LLM DONE", and waits again. Lines starting with '/':
//   /steps N         generate up to N tokens (default and maximum: seq_len)
//   /trace PROMPT    run the full N steps (no stop at the end of the story)
//                    and then print one line per step, "T pos token
//                    logits-hash", which must equal fw/build/golden's output
//                    for the same image, prompt and steps
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_hw.h"
#include "model.h"
#include "trace.h"

static uint64_t g_mm_cycles;  // fabric cycles spent in qmatmul
static int g_keep_logits;     // copy the classifier's output (tracing)
static int g_vocab;
uint32_t prof_now(void) { return LLM_CYCLES; }

// the run state's small buffers in local memory (single-cycle, unlike DDR
// behind the data cache); calloc once it is full
static uint8_t g_arena[48 * 1024] __attribute__((aligned(8)));
static size_t g_arena_used;

void *fast_calloc(size_t n, size_t size) {
    size_t bytes = (n * size + 7) & ~(size_t)7;
    if (g_arena_used + bytes > sizeof(g_arena)) return calloc(n, size);
    void *p = g_arena + g_arena_used;
    g_arena_used += bytes;
    memset(p, 0, bytes);
    return p;
}

// qmv's x RAM holds this tensor: put there by the vector unit (mm_quant,
// ffn_hidden in vpu_hw.c), so qmatmul doesn't write it again
const QTensor *g_xram;

// run a matmul of what the x RAM holds; results to Y from word yoff
void mm_run(uint32_t w_off, int n, int d, int yoff) {
    uint32_t t0 = LLM_CYCLES;
    LLM_MM_YOFF = (uint32_t)yoff;
    LLM_MM_ADDR = w_off;
    LLM_MM_ND = (uint32_t)n | ((uint32_t)d << 16);
    LLM_MM_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
    g_mm_cycles += LLM_CYCLES - t0;
}

void qmatmul(float *y, const QTensor *x, uint32_t w_off, int n, int d) {
    if (x != g_xram) {
        const uint32_t *xq = (const uint32_t *)x->q;  // calloc'd: word aligned
        const uint32_t *xs = (const uint32_t *)x->s;
        for (int i = 0; i < n / 4; i++) LLM_XQ[i] = xq[i];
        for (int g = 0; g < n / 32; g++) LLM_XS[g] = xs[g];
        g_xram = x;
    }
    mm_run(w_off, n, d, 0);
    // the classifier's logits are only needed for the trace: the next token
    // comes from the hardware argmax
    uint32_t t0 = LLM_CYCLES;
    if (d != g_vocab || g_keep_logits) {
        uint32_t *yw = (uint32_t *)y;
        for (int i = 0; i < d; i++) yw[i] = LLM_YL[i];
    }
    g_mm_cycles += LLM_CYCLES - t0;
}

static void fail(const char *msg) {
    printf("LLM FAIL: %s\n", msg);
    for (;;) {}
}

// one line from the host (CONSOLE_IN), without the newline
static void read_line(char *buf, int size) {
    int n = 0;
    for (;;) {
        uint32_t c = LLM_CONSOLE_IN;
        if (!(c & 0x100)) continue;
        c &= 0xFF;
        if (c == '\n') break;
        if (c != '\r' && n < size - 1) buf[n++] = (char)c;
    }
    buf[n] = '\0';
}

static void generate(const Model *m, Tokenizer *tok, RunState *s, const char *prompt, int steps,
                     int trace) {
    static int prompt_tokens[1024 + 3];
    static uint32_t *trace_buf;  // in DDR: local memory is full
    if (!trace_buf && !(trace_buf = malloc(2 * sizeof(uint32_t) * (size_t)m->c.seq_len)))
        fail("out of memory");
    int n_prompt = 0;
    encode(tok, prompt, 1, 0, prompt_tokens, &n_prompt);
    if (n_prompt > steps) n_prompt = steps;

    g_keep_logits = trace;
    g_mm_cycles = 0;
    memset(prof_cycles, 0, sizeof(prof_cycles));
    extern uint64_t g_kv_cycles;
    g_kv_cycles = 0;
    uint64_t total = 0;
    int token = prompt_tokens[0], pos = 0, argmax_mismatch = 0;
    uint32_t t0 = LLM_CYCLES;
    for (; pos < steps; pos++) {
        float *logits = forward(m, s, token, pos);
        int next = pos < n_prompt - 1 ? prompt_tokens[pos + 1] : (int)LLM_ARGMAX;
        uint32_t t1 = LLM_CYCLES;  // tracing and printing don't count
        total += t1 - t0;
        if (trace) {
            if (pos >= n_prompt - 1 && next != sample_argmax(logits, m->c.vocab_size))
                argmax_mismatch++;
            trace_buf[2 * pos] = (uint32_t)next;
            trace_buf[2 * pos + 1] = trace_hash(logits, m->c.vocab_size);
        }
        if (next == 1 && !trace) {  // BOS: the story is over (not printed, like runq.c)
            pos++;
            break;
        }
        fputs(printable(decode(tok, token, next)), stdout);
        fflush(stdout);
        token = next;
        t0 = LLM_CYCLES;
    }
    double secs = (double)total / CLK_HZ;
    extern uint64_t g_kv_cycles;
    double per = 1e3 / CLK_HZ / pos;
    uint64_t parts = g_mm_cycles + prof_cycles[PROF_ATT] + prof_cycles[PROF_NORM] +
                     prof_cycles[PROF_QUANT] + prof_cycles[PROF_ROPE] + prof_cycles[PROF_FFN];
    printf("\n\n[%d tokens in %.2f s: %.1f tok/s; ms per token: matmul %.1f, attention %.1f, "
           "(kv store %.2f), quantize %.1f, swiglu %.1f, rmsnorm %.1f, rope %.1f, other %.1f]\n",
           pos, secs,
           pos / secs, g_mm_cycles * per, prof_cycles[PROF_ATT] * per, g_kv_cycles * per,
           prof_cycles[PROF_QUANT] * per, prof_cycles[PROF_FFN] * per,
           prof_cycles[PROF_NORM] * per, prof_cycles[PROF_ROPE] * per, (total - parts) * per);
    if (trace) {
        if (argmax_mismatch) printf("LLM FAIL: hardware argmax differs %d times\n", argmax_mismatch);
        for (int p = 0; p < pos; p++)
            printf("T %d %d %08lx\n", p, (int)trace_buf[2 * p], (unsigned long)trace_buf[2 * p + 1]);
    }
}

int main(void) {
    printf("\n=== top_soc: llama2 on the MicroBlaze V + int8 matmul engine ===\n");
    // straight from the bitstream the CPU starts before DDR3 is calibrated
    // and the model image is loaded: wait for both
    uint32_t st;
    do st = LLM_STATUS;
    while (!(st & ST_CALIB) || (st & ST_SD_BUSY) || !(st & (ST_SD_OK | ST_SD_ERR)));

    Model m;
    Tokenizer tok;
    RunState s;
    const char *err = model_open(&m, (const uint8_t *)DDR_BASE);
    if (err) fail(err);
    printf("model: dim %d, hidden %d, %d layers, %d heads, vocab %d, seq_len %d\n", m.c.dim,
           m.c.hidden_dim, m.c.n_layers, m.c.n_heads, m.c.vocab_size, m.c.seq_len);
    if ((err = tokenizer_open(&tok, &m))) fail(err);
    // the RMSNorm weights into local memory (read every layer)
    {
        int nw = (2 * m.c.n_layers + 1) * m.c.dim;
        float *w = fast_calloc(nw, sizeof(float));
        if (w) {
            int L = m.c.n_layers * m.c.dim;
            memcpy(w, m.rms_att, L * sizeof(float));
            memcpy(w + L, m.rms_ffn, L * sizeof(float));
            memcpy(w + 2 * L, m.rms_final, m.c.dim * sizeof(float));
            m.rms_att = w;
            m.rms_ffn = w + L;
            m.rms_final = w + 2 * L;
        }
    }
    if (!run_state_alloc(&s, &m.c)) fail("out of memory");
    g_vocab = m.c.vocab_size;

    int steps = m.c.seq_len;
    static char line[1024];
    for (;;) {
        printf("READY\n");
        read_line(line, sizeof(line));
        if (strncmp(line, "/steps ", 7) == 0) {
            steps = atoi(line + 7);
            if (steps < 1 || steps > m.c.seq_len) steps = m.c.seq_len;
            printf("steps: %d\nLLM DONE\n", steps);
            continue;
        }
        int trace = strncmp(line, "/trace ", 7) == 0;
        generate(&m, &tok, &s, trace ? line + 7 : line, steps, trace);
        printf("LLM DONE\n");
    }
}
