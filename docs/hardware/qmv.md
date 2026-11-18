# Matmul engine (`qmv`) and weight reader (`wstream_reader`)

Sources: `src/sources/qmv.sv`, `src/sources/wstream_reader.sv`.

## What it computes

`y = W x`. W is d × n and x has n elements, both in llama2.c's Q8_0
format: int8 values with one float32 scale per group of `GS` consecutive
values along n. The engine is bit-exact with `runq.c`'s `matmul()`:

```c
for each row:  val = 0.0f
               for each group g:  ival = sum of GS int8 * int8 products  // exact, int32
                                  val  = val + ((float)ival * w_s) * x_s[g]
```

The integer dot product is exact. What has to match is the float part: the
conversion, two multiplies and an add per group, each rounded to nearest
even, in group order. The engine uses the same operations on
`int_to_fp32`, `fp32_mul` and `fp32_add`
([float_units.md](float_units.md)), in the same order.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `P` | 32 | int8 lanes: weight bytes per beat (= one 256-bit DDR3 beat) |
| `GS` | 32 | group size (64 also works, and is tested) |
| `R` | 16 | rows worked on round-robin; must exceed the add latency |
| `MaxN` | 2048 | longest x |

The requirements are `n % GS == 0`, `GS % P == 0`, `d % R == 0` and
`n ≤ MaxN`. Every stories model meets them: 32,000 rows is a multiple of
16.

## Pipeline

One weight beat (32 int8) per cycle:

1. **c1:** the beat is registered, and the x RAM read returns the matching
   32 x values.
2. **c2:** 32 int8 × int8 products.
3. **Adder tree:** registered after every two levels; gives the beat's
   int32 sum.
4. **Group sum:** for `GS > P`, the group's consecutive beats are added.
5. **`int_to_fp32`** (3 cycles), then **× w_s** (`fp32_mul`, 5), then
   **× x_s** (5).
6. **Accumulate:** `fp32_add` (7 cycles) onto the row's running sum. The
   sum is read from a small RAM when the term enters and written back when
   the add finishes. The first group adds to +0.0, as `val = 0.0f` does,
   which matters for the sign of a zero result.

**Why rows go round-robin.** Each row's sum is a chain of dependent adds
(7 cycles each), so one row alone could only take a beat every 7 cycles.
With R = 16 rows round-robin, the same row comes back 16 beats later, after
its previous add has finished. The engine therefore takes a beat every
cycle, and every row still adds its groups in order.

**Results** come out one float per cycle, rows in order, with `y_valid_o`.
The consumer can't stall them:
- in `top_qmv`, the UART result FIFO takes them;
- in `llm_core`, the result FIFO for UART matmuls, or the Y RAM for CPU
  matmuls.

`done_o` pulses after the last row. `abort_i` drops the job.

**Argmax** (in `llm_core`) watches the results of CPU matmuls in row order.
It compares ordered integer keys of the floats, where −0 equals +0, and
keeps the first maximum, like C's `if (p[i] > max)`.

**Throughput:** 32 MACs per cycle at 225 MHz = 7.2 G int8 MAC/s, which
consumes exactly the DDR3 peak of 7.2 GB/s. In practice the matmuls run at
about 5.5 GB/s effective, including job start, DDR refresh and page
misses ([performance.md](../performance.md)).

## Weight stream order

The host lays W out in the order the engine consumes it:

```
for each block of R rows:
  for each group g:
    for each row r in the block:
      for each of the GS/P beats k of the group:
        W[row, g*GS + k*P +: P]        (one beat, with that row's scale for group g)
```

`qmv_stream()` in `ref/q80.py` and `ddr_format()` in `host/llm_link.py`
produce this order.

## Weight reader (`wstream_reader`)

The reader streams a matrix from DDR3 over AXI4 into the engine's weight
port.

**qmv DDR format** (`raw_i = 0`) groups the qmv beat stream in chunks of 8
beats. Each chunk is 9 AXI beats:
- AXI beat 0: the 8 beats' group scales, float32, scale j at bytes 4j;
- AXI beats 1-8: the 8 beats' 32 int8 weights.

The reader splits off each scale beat and hands every weight beat its scale.
A matrix of B qmv beats takes B/8 × 9 AXI beats: 12.5 % overhead, the same
as Q8_0's own scales. See [data_formats.md](../data_formats.md#qmv-ddr-format).

**Raw mode** (`raw_i = 1`) passes the beats on unchanged. The attention
unit uses it for K and V.

**Bursts:**
- `BurstBeats` = 64 (2 KB). `base_i` must be 4 KB aligned so that no burst
  crosses a 4 KB boundary.
- An AR burst is issued only when the FIFO is sure to have room for all of
  it, so R is always ready.
- `llm_core` sets `FifoDepth` to 512 beats (16 KB, LUT RAM), which allows 8
  bursts in flight. Going from 128 to 512 beats raised the rate from 128 to
  132 tokens/s (256-token runs).
- `abort_i` stops new bursts. Bursts already issued are received and
  dropped, and `busy_o` stays high until they are done.

## Verification

- `tb_qmv` (`sim/run_qmv.sh`) feeds real `runq.c` matmul dumps with random
  stalls in the weight stream, and checks every output bit for bit:
  - layer 0 wq (288 × 288);
  - w1 (768 × 288);
  - w2 (288 × 768);
  - the 32,000-row classifier;
  - a random group-size-64 matmul.
- `tb_top_qmv` runs the engine through `top_qmv`'s UART pins.
- `tb_llm_core` runs it from the AXI memory model through the reader.

## Resources

About 3.2k LUTs, 2k FFs, half a RAMB36 and 4 DSPs. The reader adds 3.6k
LUTs, mostly the FIFO as LUT RAM.
