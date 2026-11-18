// Host (x86) build of the firmware's model code (fw/model), the golden for
// the board: the same forward() over the same model image, with the matmul
// done in software exactly the way the engine does it.
//
// Greedy decoding. Writes <outdir>/tokens.txt and <outdir>/logits.bin in the
// format of ref/runq_dump, and prints the same per-step trace lines as the
// board ("T pos token logits-hash"), so a console capture can be diffed.
//
// Built twice (fw/Makefile golden): build/golden uses fw/model/mathf.c and
// must match the board bit for bit; build/golden_libm uses the C library's
// expf/sinf/cosf/powf like runq.c and must match runq_dump bit for bit.
//
// usage: golden <model.img> <prompt> <steps> <outdir>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "trace.h"

static const uint8_t *g_image;

// the engine's matmul, from the qmv DDR format (see fw/bringup/main.c)
void qmatmul(float *y, const QTensor *x, uint32_t w_off, int n, int d) {
    const uint8_t *w = g_image + w_off;
    int G = n / 32;
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int g = 0; g < G; g++) {
            uint32_t b = ((uint32_t)(i / 16) * G + g) * 16 + i % 16;
            const uint8_t *chunk = w + (b / 8) * 9 * 32;
            float ws;
            memcpy(&ws, chunk + (b % 8) * 4, 4);
            const int8_t *q = (const int8_t *)(chunk + 32 + (b % 8) * 32);
            int32_t ival = 0;
            for (int k = 0; k < 32; k++) ival += (int32_t)x->q[g * 32 + k] * (int32_t)q[k];
            val += ((float)ival * ws) * x->s[g];
        }
        y[i] = val;
    }
}

static void die(const char *msg) {
    fprintf(stderr, "golden: %s\n", msg);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc != 5) die("usage: golden <model.img> <prompt> <steps> <outdir>");
    FILE *f = fopen(argv[1], "rb");
    if (!f) die("can't open the image");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *image = malloc(size);
    if (!image || fread(image, 1, size, f) != (size_t)size) die("can't read the image");
    fclose(f);
    g_image = image;

    Model m;
    Tokenizer tok;
    RunState s;
    const char *err = model_open(&m, image);
    if (!err) err = tokenizer_open(&tok, &m);
    if (err) die(err);
    if (!run_state_alloc(&s, &m.c)) die("out of memory");

    const char *prompt = argv[2];
    int steps = atoi(argv[3]);
    if (steps > m.c.seq_len) steps = m.c.seq_len;
    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int n_prompt = 0;
    encode(&tok, prompt, 1, 0, prompt_tokens, &n_prompt);

    char path[1024];
    snprintf(path, sizeof(path), "%s/logits.bin", argv[4]);
    FILE *logits_f = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/tokens.txt", argv[4]);
    FILE *tokens_f = fopen(path, "w");
    if (!logits_f || !tokens_f) die("can't write to outdir");

    int token = prompt_tokens[0];
    fprintf(tokens_f, "%d\n", token);
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(&m, &s, token, pos);
        fwrite(logits, sizeof(float), m.c.vocab_size, logits_f);
        int next = pos < n_prompt - 1 ? prompt_tokens[pos + 1] : sample_argmax(logits, m.c.vocab_size);
        fprintf(tokens_f, "%d\n", next);
        printf("T %d %d %08x\n", pos, next, trace_hash(logits, m.c.vocab_size));
        fprintf(stderr, "%s", printable(decode(&tok, token, next)));
        token = next;
    }
    fprintf(stderr, "\n");
    fclose(logits_f);
    fclose(tokens_f);
    return 0;
}
