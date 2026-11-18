// llama2.c int8 inference with every matmul on the FPGA (top_llm).
//
// This is runq.c (compiled in, see host/build_host.sh) with matmul() replaced:
// the weights live in the board's DDR3 in qmv DDR format, and each matmul
// sends only x over the UART and reads back the results. Everything else
// (RMSNorm, RoPE, attention, SwiGLU, sampling) is runq.c's own code, so the
// logits must come out bit-identical to the CPU run (ref/runq_dump.c).
//
// Greedy decoding. Writes <outdir>/tokens.txt and <outdir>/logits.bin in the
// same format as runq_dump, so the two can be compared with cmp.
//
// usage: llm_host <checkpoint> <tokenizer> <prompt> <steps> <outdir> [options]
//   -p port   serial port (default /dev/ttyUSB0)
//   -u        always upload the weights over the UART
//   -n        never upload: the weights are already in DDR
//             (default: upload unless DDR holds a model image of this model,
//             host/build_image.py, or the FPGA loaded bare weights of exactly
//             the right size from the microSD card at boot)
//   -s        after uploading, store the weights on the microSD card, so the
//             FPGA loads them by itself at power-on
//   -v        also run each matmul on the CPU and compare

#define main runq_main
#include "runq_patched.c"
#undef main

#include <errno.h>
#include <poll.h>
#include <termios.h>
#include <time.h>

#define BEAT 32
#define P 32
#define R 16

static int g_fd = -1;
static int g_verify = 0;
static long g_mismatches = 0;
static long g_matmuls = 0;
static double g_fpga_seconds = 0;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Serial port
// ---------------------------------------------------------------------------
static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static void ser_open(const char *dev) {
    g_fd = open(dev, O_RDWR | O_NOCTTY);
    if (g_fd < 0) { perror(dev); exit(EXIT_FAILURE); }
    struct termios t;
    if (tcgetattr(g_fd, &t) != 0) { perror("tcgetattr"); exit(EXIT_FAILURE); }
    cfmakeraw(&t);
    cfsetispeed(&t, B3000000);
    cfsetospeed(&t, B3000000);
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cc[VMIN] = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(g_fd, TCSANOW, &t) != 0) { perror("tcsetattr"); exit(EXIT_FAILURE); }
    // let any half-sent command from an earlier run time out on the FPGA
    usleep(200000);
    tcflush(g_fd, TCIOFLUSH);
}

static void ser_write(const void *buf, size_t n) {
    const uint8_t *p = buf;
    while (n > 0) {
        ssize_t k = write(g_fd, p, n);
        if (k < 0) {
            if (errno == EINTR) continue;
            perror("serial write");
            exit(EXIT_FAILURE);
        }
        p += k;
        n -= (size_t)k;
    }
}

// read exactly n bytes; timeout_ms is the longest allowed gap between bytes
static void ser_read(void *buf, size_t n, int timeout_ms) {
    uint8_t *p = buf;
    while (n > 0) {
        struct pollfd pfd = {.fd = g_fd, .events = POLLIN};
        int r = poll(&pfd, 1, timeout_ms);
        if (r == 0) {
            fprintf(stderr, "serial read timed out with %zu bytes still to come\n", n);
            exit(EXIT_FAILURE);
        }
        if (r < 0) {
            if (errno == EINTR) continue;
            perror("poll");
            exit(EXIT_FAILURE);
        }
        ssize_t k = read(g_fd, p, n);
        if (k < 0) {
            if (errno == EINTR) continue;
            perror("serial read");
            exit(EXIT_FAILURE);
        }
        p += k;
        n -= (size_t)k;
    }
}

static void put_u32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i));
}

// ---------------------------------------------------------------------------
// Commands (src/sources/llm_core.sv)
// ---------------------------------------------------------------------------
typedef struct {
    int calibrated, sd_busy, sd_ok, sd_err, sd_err_code;
    uint32_t sd_bytes;
} Status;

static Status fpga_status(void) {
    uint8_t op = 0x00, r[12];
    ser_write(&op, 1);
    ser_read(r, 12, 2000);
    if (memcmp(r, "LLM", 3) != 0) die("bad status reply: is top_llm loaded?");
    Status st = {r[3] & 1, (r[3] >> 1) & 1, (r[3] >> 2) & 1, (r[3] >> 3) & 1, r[4],
                 (uint32_t)r[8] | (uint32_t)r[9] << 8 | (uint32_t)r[10] << 16 | (uint32_t)r[11] << 24};
    return st;
}

static Status fpga_sd_wait(void) {
    Status st;
    while ((st = fpga_status()).sd_busy) usleep(200000);
    return st;
}

// copy DDR [0, nbytes) to the microSD card
static void fpga_sd_store(uint32_t nbytes) {
    uint8_t cmd[9], ack;
    cmd[0] = 0x05;
    put_u32(cmd + 1, 0);
    put_u32(cmd + 5, nbytes);
    ser_write(cmd, sizeof(cmd));
    ser_read(&ack, 1, 2000);
    if (ack != 0x05) die("bad SD_STORE ack");
    double t0 = now();
    fprintf(stderr, "storing %.1f MB to the microSD card...\n", nbytes / 1e6);
    Status st = fpga_sd_wait();
    if (!st.sd_ok) {
        fprintf(stderr, "SD store failed, error %d\n", st.sd_err_code);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "stored on the card in %.0f s\n", now() - t0);
}

static void fpga_write(uint32_t addr, const uint8_t *data, size_t len) {
    uint8_t hdr[9];
    hdr[0] = 0x01;
    put_u32(hdr + 1, addr);
    put_u32(hdr + 5, (uint32_t)(len / BEAT));
    ser_write(hdr, sizeof(hdr));
    ser_write(data, len);
    uint8_t ack;
    ser_read(&ack, 1, 5000);
    if (ack != 0x01) die("bad write ack");
}

static void fpga_read(uint32_t addr, uint8_t *data, size_t len) {
    uint8_t hdr[9];
    hdr[0] = 0x02;
    put_u32(hdr + 1, addr);
    put_u32(hdr + 5, (uint32_t)(len / BEAT));
    ser_write(hdr, sizeof(hdr));
    ser_read(data, len, 5000);
}

// ---------------------------------------------------------------------------
// Weights in DDR
// ---------------------------------------------------------------------------
// qmv DDR format (wstream_reader.sv): the qmv beat stream (qmv.sv: per block
// of R rows, per group, per row in the block, per beat of the group) in chunks
// of 8 beats, each chunk one beat of the 8 group scales then the 8 weight
// beats.
static size_t ddr_bytes(int n, int d) {
    return (size_t)d * n / P / 8 * 9 * BEAT;
}

static void pack_ddr(const QuantizedTensor *w, int n, int d, uint8_t *out) {
    int G = n / GS, bpg = GS / P;
    size_t beat = 0;
    for (int blk = 0; blk < d / R; blk++) {
        for (int g = 0; g < G; g++) {
            for (int r = 0; r < R; r++) {
                int row = blk * R + r;
                for (int k = 0; k < bpg; k++, beat++) {
                    uint8_t *chunk = out + beat / 8 * 9 * BEAT;
                    memcpy(chunk + (beat % 8) * 4, &w->s[(size_t)row * G + g], 4);
                    memcpy(chunk + (1 + beat % 8) * BEAT,
                           &w->q[(size_t)row * n + (size_t)g * GS + (size_t)k * P], P);
                }
            }
        }
    }
}

typedef struct {
    const int8_t *q;  // identifies the tensor
    uint32_t addr;
} WeightAddr;

static WeightAddr g_addrs[256];
static int g_naddrs = 0;

static uint32_t g_next_addr = 0;

static void place(const QuantizedTensor *w, int n, int d, int upload, size_t *uploaded) {
    if (n % GS || GS % P || d % R) die("matrix shape not supported by the FPGA engine");
    size_t bytes = ddr_bytes(n, d);
    g_addrs[g_naddrs].q = w->q;
    g_addrs[g_naddrs].addr = g_next_addr;
    g_naddrs++;
    if (upload) {
        uint8_t *buf = malloc(bytes);
        pack_ddr(w, n, d, buf);
        fpga_write(g_next_addr, buf, bytes);
        free(buf);
        *uploaded += bytes;
        fprintf(stderr, "\ruploaded %.1f MB", *uploaded / 1e6);
    }
    g_next_addr += (uint32_t)((bytes + 4095) / 4096 * 4096);
}

static void place_all(Transformer *t, int upload) {
    Config *p = &t->config;
    TransformerWeights *w = &t->weights;
    int dim = p->dim, hid = p->hidden_dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    size_t uploaded = 0;
    double t0 = now();
    for (int l = 0; l < p->n_layers; l++) {
        place(w->wq + l, dim, dim, upload, &uploaded);
        place(w->wk + l, dim, kv_dim, upload, &uploaded);
        place(w->wv + l, dim, kv_dim, upload, &uploaded);
        place(w->wo + l, dim, dim, upload, &uploaded);
        place(w->w1 + l, dim, hid, upload, &uploaded);
        place(w->w2 + l, hid, dim, upload, &uploaded);
        place(w->w3 + l, dim, hid, upload, &uploaded);
    }
    place(w->wcls, dim, p->vocab_size, upload, &uploaded);
    if (upload)
        fprintf(stderr, "\ruploaded %.1f MB of weights to DDR3 in %.0f s\n", uploaded / 1e6, now() - t0);
}

// A model image in DDR (host/build_image.py, fw/model/model.h) lists its
// matrices in place_all()'s order: use those addresses if it is this model.
static int use_model_image(Transformer *t) {
    uint8_t h[4096];
    fpga_read(0, h, sizeof(h));
    if (memcmp(h, "LLMMODEL", 8) != 0) return 0;
    int32_t c[8];
    uint32_t n_mats;
    memcpy(c, h + 16, sizeof(c));
    memcpy(&n_mats, h + 76, 4);
    Config *p = &t->config;
    if (c[0] != p->dim || c[1] != p->hidden_dim || c[2] != p->n_layers || c[3] != p->n_heads ||
        c[4] != p->n_kv_heads || c[5] != p->vocab_size || c[7] != GS ||
        n_mats != (uint32_t)p->n_layers * 7 + 1 || 84 + 4 * n_mats > sizeof(h))
        return 0;
    TransformerWeights *w = &t->weights;
    const QuantizedTensor *order[256];
    int k = 0;
    for (int l = 0; l < p->n_layers; l++) {
        order[k++] = w->wq + l;
        order[k++] = w->wk + l;
        order[k++] = w->wv + l;
        order[k++] = w->wo + l;
        order[k++] = w->w1 + l;
        order[k++] = w->w2 + l;
        order[k++] = w->w3 + l;
    }
    order[k++] = w->wcls;
    g_naddrs = 0;
    for (int i = 0; i < k; i++) {
        g_addrs[i].q = order[i]->q;
        memcpy(&g_addrs[i].addr, h + 84 + 4 * i, 4);
    }
    g_naddrs = k;
    return 1;
}

static uint32_t addr_of(const QuantizedTensor *w) {
    for (int i = 0; i < g_naddrs; i++)
        if (g_addrs[i].q == w->q) return g_addrs[i].addr;
    die("matmul with a weight that isn't in DDR");
    return 0;
}

// ---------------------------------------------------------------------------
// The matmul runq.c's forward() calls
// ---------------------------------------------------------------------------
void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    double t0 = now();
    size_t len = 9 + (size_t)n + (size_t)n / GS * 4;
    uint8_t *cmd = malloc(len);
    cmd[0] = 0x03;
    put_u32(cmd + 1, addr_of(w));
    put_u32(cmd + 5, (uint32_t)n | ((uint32_t)d << 16));
    memcpy(cmd + 9, x->q, (size_t)n);
    memcpy(cmd + 9 + n, x->s, (size_t)n / GS * 4);
    ser_write(cmd, len);
    free(cmd);
    ser_read(xout, (size_t)d * 4, 2000);
    g_fpga_seconds += now() - t0;
    g_matmuls++;

    if (g_verify) {
        float *ref = malloc((size_t)d * sizeof(float));
        matmul_orig(ref, x, w, n, d);
        for (int i = 0; i < d; i++)
            if (memcmp(&ref[i], &xout[i], 4) != 0) g_mismatches++;
        free(ref);
    }
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: %s <checkpoint> <tokenizer> <prompt> <steps> <outdir> "
                "[-p port] [-n] [-v]\n", argv[0]);
        return EXIT_FAILURE;
    }
    char *checkpoint = argv[1], *tokenizer_path = argv[2], *prompt = argv[3];
    int steps = atoi(argv[4]);
    char *outdir = argv[5];
    char *port = "/dev/ttyUSB0";
    int upload = -1;  // -1: decide from the SD status
    int store = 0;
    for (int i = 6; i < argc; i++) {
        if (!strcmp(argv[i], "-p") && i + 1 < argc) port = argv[++i];
        else if (!strcmp(argv[i], "-u")) upload = 1;
        else if (!strcmp(argv[i], "-n")) upload = 0;
        else if (!strcmp(argv[i], "-s")) store = 1;
        else if (!strcmp(argv[i], "-v")) g_verify = 1;
        else die("unknown option");
    }

    Transformer transformer;
    build_transformer(&transformer, checkpoint);
    int vocab = transformer.config.vocab_size;
    if (steps > transformer.config.seq_len) steps = transformer.config.seq_len;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, vocab);

    ser_open(port);
    Status st = fpga_sd_wait();  // a boot-time SD load may still be running
    if (!st.calibrated) die("DDR3 not calibrated");
    place_all(&transformer, 0);  // lay out the addresses only
    uint32_t image_bytes = g_next_addr;
    if (upload <= 0 && use_model_image(&transformer)) {
        fprintf(stderr, "using the model image in DDR\n");
        upload = 0;
        store = 0;
    }
    if (upload < 0) {
        upload = !(st.sd_ok && st.sd_bytes == image_bytes);
        fprintf(stderr, upload ? "no matching image loaded from the microSD card: uploading\n"
                               : "weights loaded from the microSD card at boot\n");
    }
    if (upload) {
        g_naddrs = 0;
        g_next_addr = 0;
        place_all(&transformer, 1);
    }
    if (store) fpga_sd_store(image_bytes);

    int *prompt_tokens = malloc((strlen(prompt) + 3) * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

    char path[1024];
    snprintf(path, sizeof(path), "%s/logits.bin", outdir);
    FILE *logits_f = fopen(path, "wb");
    snprintf(path, sizeof(path), "%s/tokens.txt", outdir);
    FILE *tokens_f = fopen(path, "w");
    if (!logits_f || !tokens_f) die("can't write to outdir");

    double t0 = now();
    int token = prompt_tokens[0];
    fprintf(tokens_f, "%d\n", token);
    for (int pos = 0; pos < steps; pos++) {
        float *logits = forward(&transformer, token, pos);
        fwrite(logits, sizeof(float), vocab, logits_f);
        int next = pos < num_prompt_tokens - 1 ? prompt_tokens[pos + 1]
                                               : sample_argmax(logits, vocab);
        fprintf(tokens_f, "%d\n", next);
        safe_printf(decode(&tokenizer, token, next));
        fflush(stdout);
        token = next;
    }
    double dt = now() - t0;
    printf("\n");
    fprintf(stderr, "%d tokens in %.1f s (%.2f tok/s); %ld matmuls on the FPGA, %.1f s of that in "
            "matmul round trips\n", steps, dt, steps / dt, g_matmuls, g_fpga_seconds);
    if (g_verify)
        fprintf(stderr, "verify: %ld matmul outputs differ from the CPU\n", g_mismatches);

    fclose(logits_f);
    fclose(tokens_f);
    free(prompt_tokens);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    close(g_fd);
    return (g_verify && g_mismatches) ? EXIT_FAILURE : EXIT_SUCCESS;
}
