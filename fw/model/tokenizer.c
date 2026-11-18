// runq.c's BPE tokenizer, reading tokenizer.bin from the model image.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((const TokenIndex *)a)->str, ((const TokenIndex *)b)->str);
}

const char *tokenizer_open(Tokenizer *t, const Model *m) {
    const ImageHeader *h = (const ImageHeader *)m->image;
    const uint8_t *p = m->image + h->off_tok, *end = p + h->tok_bytes;
    int vocab_size = m->c.vocab_size;
    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_scores = malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    if (!t->vocab || !t->vocab_scores) return "out of memory";
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    memcpy(&t->max_token_length, p, 4);
    p += 4;
    for (int i = 0; i < vocab_size; i++) {
        int len;
        if (p + 8 > end) return "tokenizer truncated";
        memcpy(&t->vocab_scores[i], p, 4);
        memcpy(&len, p + 4, 4);
        p += 8;
        if (len < 0 || p + len > end) return "tokenizer truncated";
        t->vocab[i] = malloc(len + 1);
        if (!t->vocab[i]) return "out of memory";
        memcpy(t->vocab[i], p, len);
        t->vocab[i][len] = '\0';
        p += len;
    }
    return NULL;
}

const char *decode(const Tokenizer *t, int prev_token, int token) {
    const char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') piece++;
    // raw byte tokens look like "<0x01>"
    unsigned int byte_val;
    if (sscanf(piece, "<0x%02X>", &byte_val) == 1)
        piece = (const char *)t->byte_pieces + byte_val * 2;
    return piece;
}

const char *printable(const char *piece) {
    if (piece[0] != '\0' && piece[1] == '\0') {
        unsigned char b = (unsigned char)piece[0];
        if (!(isprint(b) || isspace(b))) return "";
    }
    return piece;
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, const char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // UTF-8 code points, byte fallback for those not in the vocabulary
    for (const char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }

    // merge the best-scoring consecutive pair until none is in the vocabulary
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}
