// Llama 2 inference on top_soc: runq.c's forward() (fw/model) on the
// MicroBlaze V, every matmul on the int8 engine, the model image in DDR3
// (from the microSD card at boot, or uploaded with host/image_tool.py).
//
// Greedy decoding from a fixed prompt. Prints the story as it is generated,
// then the speed, then one trace line per step ("T pos token logits-hash")
// that must equal fw/build/golden's output for the same image and prompt.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llm_hw.h"
#include "model.h"
#include "trace.h"

#ifndef PROMPT
#define PROMPT "Once upon a time"
#endif
#ifndef STEPS
#define STEPS 256
#endif

static uint64_t g_mm_cycles;  // fabric cycles spent in qmatmul
uint32_t prof_now(void) { return LLM_CYCLES; }

// the run state's small buffers in local memory (single-cycle, unlike DDR
// behind the write-through data cache); calloc once it is full
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

void qmatmul(float *y, const QTensor *x, uint32_t w_off, int n, int d) {
    uint32_t t0 = LLM_CYCLES;
    const uint32_t *xq = (const uint32_t *)x->q;  // calloc'd: word aligned
    const uint32_t *xs = (const uint32_t *)x->s;
    for (int i = 0; i < n / 4; i++) LLM_XQ[i] = xq[i];
    for (int g = 0; g < n / 32; g++) LLM_XS[g] = xs[g];
    LLM_MM_ADDR = w_off;
    LLM_MM_ND = (uint32_t)n | ((uint32_t)d << 16);
    LLM_MM_START = 1;
    while (LLM_STATUS & ST_MM_BUSY) {}
    uint32_t *yw = (uint32_t *)y;
    for (int i = 0; i < d; i++) yw[i] = LLM_Y[i];
    g_mm_cycles += LLM_CYCLES - t0;
}

static void fail(const char *msg) {
    printf("LLM FAIL: %s\n", msg);
    for (;;) {}
}

int main(void) {
    printf("\n=== top_soc: llama2 on the MicroBlaze V + int8 matmul engine ===\n");
    uint32_t st;
    while ((st = LLM_STATUS) & ST_SD_BUSY) {}
    if (!(st & ST_CALIB)) fail("DDR3 not calibrated");

    Model m;
    Tokenizer tok;
    RunState s;
    const char *err = model_open(&m, (const uint8_t *)DDR_BASE);
    if (err) fail(err);
    printf("model: dim %d, hidden %d, %d layers, %d heads, vocab %d, seq_len %d\n", m.c.dim,
           m.c.hidden_dim, m.c.n_layers, m.c.n_heads, m.c.vocab_size, m.c.seq_len);
    if ((err = tokenizer_open(&tok, &m))) fail(err);
    if (!run_state_alloc(&s, &m.c)) fail("out of memory");

    int steps = STEPS < m.c.seq_len ? STEPS : m.c.seq_len;
    static int prompt_tokens[sizeof(PROMPT) + 3];
    int n_prompt = 0;
    encode(&tok, PROMPT, 1, 0, prompt_tokens, &n_prompt);
    uint32_t *trace = malloc(steps * 2 * sizeof(uint32_t));
    static uint32_t step_cycles[64];
    uint64_t argmax_cycles = 0;
    if (!trace) fail("out of memory");

    printf("prompt \"%s\", %d steps\n\n", PROMPT, steps);
    uint32_t t0 = LLM_CYCLES;
    uint64_t total = 0;
    int token = prompt_tokens[0];
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(&m, &s, token, pos);
        uint32_t ta = LLM_CYCLES;
        int next = pos < n_prompt - 1 ? prompt_tokens[pos + 1] : sample_argmax(logits, m.c.vocab_size);
        argmax_cycles += LLM_CYCLES - ta;
        uint32_t t1 = LLM_CYCLES;  // the trace hash and printing don't count
        total += t1 - t0;
        trace[2 * pos] = (uint32_t)next;
        trace[2 * pos + 1] = trace_hash(logits, m.c.vocab_size);
        if (pos % 32 == 0 || pos == steps - 1) step_cycles[pos / 32] = t1 - t0;
        fputs(printable(decode(&tok, token, next)), stdout);
        fflush(stdout);
        token = next;
        t0 = LLM_CYCLES;
    }
    double secs = (double)total / CLK_HZ;
    printf("\n\n%d tokens in %.2f s: %.1f tok/s (matmuls %.0f%% of the time)\n", steps, secs,
           steps / secs, 100.0 * (double)g_mm_cycles / (double)total);
    static const char *names[PROF_N] = {"rmsnorm", "quantize", "matmul", "rope", "attention",
                                        "swiglu"};
    uint64_t acc = argmax_cycles;
    printf("ms per token:");
    for (int i = 0; i < PROF_N; i++) {
        printf(" %s %.1f", names[i], prof_cycles[i] * 1e3 / CLK_HZ / steps);
        acc += prof_cycles[i];
    }
    printf(" argmax %.1f other %.1f\n", argmax_cycles * 1e3 / CLK_HZ / steps,
           (total - acc) * 1e3 / CLK_HZ / steps);
    printf("ms per token at pos 0, 32, 64, ...:");
    for (int i = 0; i <= (steps - 1) / 32; i++) printf(" %.1f", step_cycles[i] * 1e3 / CLK_HZ);
    printf("\n");
    for (int pos = 0; pos < steps; pos++)
        printf("T %d %d %08lx\n", pos, (int)trace[2 * pos], (unsigned long)trace[2 * pos + 1]);
    printf("LLM DONE\n");
    for (;;) {}
}
