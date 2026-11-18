# Float units

Sources:
- `src/sources/fp32_add.sv`
- `src/sources/fp32_mul.sv`
- `src/sources/int_to_fp32.sv`
- `src/sources/fp32_div.sv`
- `src/sources/fp32_exp.sv`

These IEEE 754 binary32 pipelines make the design's results bit-exact.
Each one rounds exactly like the CPU (x86 SSE and the MicroBlaze V's FPU):
round to nearest even.

## Common interface

```
clk_i, rst_ni
valid_i, tag_i[TagW], a_i, b_i     ->   valid_o, tag_o, y_o   (Latency cycles later)
```

- Each unit is fully pipelined and takes one operation per cycle.
- `tag_i` travels alongside, so the user can carry an index or other
  metadata through the pipeline.
- There is no stall input. Users size their pipelines and FIFOs to the
  latency.

| Unit | Operation | Latency (cycles) | Notes |
|---|---|---|---|
| `fp32_add` | a + b | 7 | guard, round and sticky bits for correct rounding |
| `fp32_mul` | a × b | 5 | 2 DSP48s |
| `int_to_fp32` | (float) int32 | 3 | as C's cast |
| `fp32_div` | a / b | 28 | long division, one quotient bit per stage (24 bits + round bit), the remainder as sticky; correctly rounded |
| `fp32_exp` | eˣ | ~130 | `m_expf`'s algorithm, step by step, on `fp32_mul` / `fp32_add` |

## Simplifications

All of the units share these:
- Subnormal inputs are read as zero, and subnormal results are flushed to
  zero.
- Inf and NaN inputs are not handled. Overflow saturates to inf.
- `fp32_div` follows RISC-V's results for two special cases: x / 0 is inf,
  and 0 / 0 is the default NaN `0x7FC00000`. The vector unit relies on the
  0/0 case for an all-zero quantize group.

Q8_0 matmuls, attention and softmax on these models never produce
subnormals, infs or NaNs, so these simplifications never change a result.
The whole-model trace against the golden confirms it on every step.

## `fp32_exp`

This unit implements `m_expf` from `fw/model/mathf.c`. Each operation
below runs on its own fp32 unit, in the same order as the C:

```
x > 88.7228394  -> inf;   x < -103.972084 -> 0
t = x * log2(e);  n = (int)(t < 0 ? t - 0.5 : t + 0.5)
r = (x - n * LN2_HI) - n * LN2_LO
p = 1 + r (1 + r (1/2 + r (1/6 + r (1/24 + r (1/120 + r (1/720 + r / 5040))))))
n > 127: p *= 2, n -= 1;   n < -126: p *= 2^-100, n += 100
result = p * 2^n
```

`m_expf` is not correctly rounded (it is accurate to about 1-2 ulp). It is
plain float arithmetic in a fixed order, so it gives the same bits on any
IEEE single-precision FPU without FMA, and in this pipeline. The only
difference from the C function is where the result is subnormal
(x < −87.3), which is flushed to zero.

## Verification

`tb_fp32` (`sim/run_qmv.sh`, step 1) checks:
- add, multiply and convert on 100,000 vectors from `sim/gen_qmv.py fp`
  against numpy float32, including exact rounding ties;
- divide and exp against numpy and against `sim/fp_ref.py`'s port of
  `m_expf`.

Cases whose inputs or results would be subnormal, inf or NaN are left out,
because the units flush or saturate those by design.
