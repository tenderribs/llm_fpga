// FNV-1a over the bits of the logits: one line per step on the console is
// enough to check the board against the host golden bit for bit.
#pragma once
#include <stdint.h>
#include <string.h>

static inline uint32_t trace_hash(const float *v, int n) {
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &v[i], 4);
        for (int b = 0; b < 4; b++) {
            h ^= (u >> (8 * b)) & 0xFF;
            h *= 16777619u;
        }
    }
    return h;
}
