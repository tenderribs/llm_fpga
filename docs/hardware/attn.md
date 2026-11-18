# Attention unit (`attn`)

Source: `src/sources/attn.sv`. Firmware: `fw/llm/attn_hw.c`.

## What it computes

These are the two float loops of `runq.c`'s attention for one head, with
the same float operations in the same order:

| Mode | Computes | Output |
|---|---|---|
| 0, scores | `y[t] = Σ_i q[i] · k_t[i]` for t < nt | `ceil(nt/128) × 128` scores (the padding timesteps' are garbage) |
| 1, weighted sum | `y[i] = Σ_t a[t] · v_t[i]` for i < hs | hs sums |

Each sum starts from +0 and adds its terms in index order. The scaling by
1/√hs and the softmax are done in between by the vector unit
([vpu.md](vpu.md)).

## Data

**The A RAM** (`MaxA` = 4096 words) holds q (mode 0) or the softmax
weights a (mode 1). The vector unit writes it: COPY for q, SOFTMAX for a.
The CPU can also write it through `AT_A`.

**K or V** stream in from DDR3 through the weight reader in raw mode, one
32-byte beat (8 floats) per cycle, into 8 lanes:

| Mode | DDR3 layout | Lane j of beat c holds | Beats |
|---|---|---|---|
| 0 | K in blocks of 128 timesteps, each block `[i][t % 128]` | timestep `c × 8 + j` (c = 0..15 within a row i) | `ceil(nt/128) × hs × 16` |
| 1 | V as rows `[t][i]` | element `c × 8 + j` (c = 0..hs/8−1) | `nt × hs / 8` |

**Why K is transposed.** Each lane's sum is a chain of dependent
multiply-adds, about 8 cycles from issue to issue. With K stored by element
`i`, each lane has 16 independent sums in flight: 16 beats per row, for 128
timesteps per block. A beat therefore never waits for an earlier add, and
no sum's order changes.

**The hazard check in mode 1.** A beat waits when its sums were used in the
last 7 beats. That only happens in mode 1 when hs < 64 (hs/8 < 8 beats per
row). For stories15M (hs = 48), a V row is 6 beats, so there is a short
wait per row. `hs` must be a multiple of 8.

## Pipeline

1. **Issue:** a beat is accepted when the job runs, a beat is valid, beats
   remain and there is no hazard.
2. **S1:** the A RAM read (`a[u]`, the same word for all 8 lanes) returns.
3. **S2:** the multipliers' inputs are registered.
4. **Per lane:** `fp32_mul` (5 cycles), then `fp32_add` onto the lane's sum
   (7 cycles). The sums live in a 32-entry distributed RAM per lane. They
   are read as a term enters the adder and written back when it leaves. A
   row's first term adds to +0.
5. **Output:** when a beat's terms are each sum's last, the 8 sums go into
   a 32-entry FIFO together. They then come out one per cycle on
   `y_valid_o`, `y_idx_o` and `y_o`, and `llm_core` writes them into Y.

`done_o` pulses after the last result.

## How the firmware uses it

KV cache layout (`fw/llm/attn_hw.c`): see
[data_formats.md](../data_formats.md#kv-cache). The cache starts at DDR3
offset `0x2000_0000`. Each layer and KV head has a K slot and a V slot, 4 KB
aligned.

**Per layer, `att_store_kv`:**
1. The CPU writes k, v and q into the V RAM over the local bus.
2. Two DMA jobs per KV head store them:
   - k with stride 512 bytes (one word per 128-timestep row);
   - v with stride 4 (whole beats).

**Per head, `att_head`:**
1. COPY q into the A RAM (vector unit).
2. Scores (mode 0).
3. SOFTMAX from Y into the A RAM (vector unit).
4. Weighted sum (mode 1).
5. The CPU reads hs words from Y over the local bus.

## Performance

**Scores** at position p read `ceil((p+1)/128) × hs × 16` beats per head.
For p < 128 and stories15M that is 768 beats (24 KB), about 3.4 µs at one
beat per cycle.

**Weighted sum** reads `(p+1) × 6` beats.

Over a 168-token story, attention, softmax and the head round trips average
0.84 ms per token. The KV cache writes add another 0.36 ms.

## Verification

`tb_attn` (`sim/run_attn.sh`) runs vectors from `sim/gen_attn.py`, with the
expected sums computed in float32 in `runq.c`'s order. The cases cover:
- head sizes 8, 16, 48 and 64;
- 1 to 200 timesteps;
- one and two K blocks;
- the hazard path (hs = 8, 16).

`tb_llm_core` sections 7 and 8 run the unit through the registers, the
second in a full scores → softmax → weighted-sum chain.

## Resources

About 4.2k LUTs, 4.3k FFs, 4 RAMB36 (the A RAM) and 16 DSPs (2 per lane's
multiplier).
