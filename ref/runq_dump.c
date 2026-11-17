// Golden reference dump from llama2.c's runq.c (int8 Q8_0 inference).
//
// Runs greedy decoding (argmax, no sampling) and writes, to <outdir>:
//   tokens.txt   one token id per line: the prompt, then generated tokens
//   logits.bin   float32 [steps][vocab_size]: logits after each forward()
//   mm_NNN.bin   every matmul() call made during forward() at position
//                <dump_pos>, in call order (see ref/q80.py read_matmul)
//
// runq.c is compiled into this file with its matmul() renamed to
// matmul_orig() (see ref/build_ref.sh), so the matmul() below sees every call.
//
// usage: runq_dump <checkpoint> <tokenizer> <prompt> <steps> <dump_pos> <outdir>

#define main runq_main
#include "runq_patched.c"
#undef main

static int g_dump = 0;
static int g_mm_idx = 0;
static const char *g_outdir;

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    matmul_orig(xout, x, w, n, d);
    if (!g_dump) return;

    char path[1024];
    snprintf(path, sizeof(path), "%s/mm_%03d.bin", g_outdir, g_mm_idx++);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "can't write %s\n", path); exit(EXIT_FAILURE); }
    // header: n (inputs), d (outputs), group size, reserved
    int32_t hdr[4] = {n, d, GS, 0};
    fwrite(hdr, sizeof(int32_t), 4, f);
    fwrite(x->q, 1, n, f);
    fwrite(x->s, sizeof(float), n / GS, f);
    fwrite(w->q, 1, (size_t)n * d, f);
    fwrite(w->s, sizeof(float), (size_t)n * d / GS, f);
    fwrite(xout, sizeof(float), d, f);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <checkpoint> <tokenizer> <prompt> <steps> <dump_pos> <outdir>\n",
                argv[0]);
        return EXIT_FAILURE;
    }
    char *checkpoint = argv[1], *tokenizer_path = argv[2], *prompt = argv[3];
    int steps = atoi(argv[4]);
    int dump_pos = atoi(argv[5]);
    g_outdir = argv[6];

    Transformer transformer;
    build_transformer(&transformer, checkpoint);
    int vocab = transformer.config.vocab_size;
    if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab);

    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    char path[1024];
    snprintf(path, sizeof(path), "%s/logits.bin", g_outdir);
    FILE *logits_f = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/tokens.txt", g_outdir);
    FILE *tokens_f = fopen(path, "w");
    if (!logits_f || !tokens_f) { fprintf(stderr, "can't write to %s\n", g_outdir); return EXIT_FAILURE; }

    int token = prompt_tokens[0];
    fprintf(tokens_f, "%d\n", token);
    for (int pos = 0; pos < steps; pos++) {
        g_dump = (pos == dump_pos);
        float *logits = forward(&transformer, token, pos);
        fwrite(logits, sizeof(float), vocab, logits_f);

        int next = pos < num_prompt_tokens - 1 ? prompt_tokens[pos + 1]
                                               : sample_argmax(logits, vocab);
        fprintf(tokens_f, "%d\n", next);
        safe_printf(decode(&tokenizer, token, next));
        fflush(stdout);
        token = next;
    }
    printf("\n");

    fclose(logits_f);
    fclose(tokens_f);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
