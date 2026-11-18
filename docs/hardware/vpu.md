# Vector unit (`vpu`)

Source: `src/sources/vpu.sv`. Firmware: `fw/llm/vpu_hw.c` and
`fw/llm/attn_hw.c`.

The vector unit does the float work between the matmuls and the attention
unit. It uses the firmware's float operations (`fw/model/model.c`, which are
`runq.c`'s) in the same order, so the results are the same bits.

## Operations

Started by a write to `VPU_OP` (see [llm_core.md](llm_core.md#cpu-register-map)).

| Op | Name | Does |
|---|---|---|
| 0 | COPY | `A[i] = V[src + i]`: q into the attention unit's A RAM |
| 1 | QUANT | x[0..n) from Y, or from V when bit 8 is set, into the matmul's x RAM as Q8_0: per group of 32, `wmax = max \|x\|`, `scale = wmax / 127`, `q = round_half_away(x / scale)`, `xs = scale` |
| 2 | SILU | `V[dst + i] = (a · (1 / (1 + e^−a))) · b`, with `a = Y[src + i]` and `b = Y[src2 + i]` (SwiGLU) |
| 3 | SOFTMAX | `s_i = Y[src + i] / f` into `V[dst + i]`; the max; `e_i = e^(s_i − max)` into `Y[i]`; `sum = 0 + e_0 + e_1 + …` in order; `A[i] = e_i / sum` |

Each formula is exactly the C expression's order of operations. For
example, SILU computes `1 / (1 + e)` and then multiplies, as
`val *= (1.0f / (1.0f + expf(-val)))` does. It does not divide `a` by
`(1 + e)`.

## How it works

Each operation runs as one or more **passes**. A pass streams elements
through shared units: one divider (`fp32_div`), one exp (`fp32_exp`), one
adder and two multipliers.

| Op | Passes |
|---|---|
| COPY | `P_COPY` |
| QUANT | per group of 32: `P_QMAX` (max \|x\|), `P_QSCALE` (one divide), `P_QDIV` (32 divides, round, pack into a 32-byte line) |
| SILU | `P_SILU` (reads a and b, then exp → add → divide → multiply → multiply) |
| SOFTMAX | `P_SM1` (divide by f, track the max), `P_SM2` (subtract the max, exp), `P_SM3` (sequential sum), `P_SM4` (divide by the sum into A) |

**Reads.** Read data is used `RdLat` = 3 cycles after the address goes
out: the RAM's address register, the read, then the data register here.
Results are written as they come out of the units.

**Throughput.** Every pass issues an element per cycle, except `P_SM3`. The
softmax sum is a chain of dependent adds, so it takes one element per add
plus read latency, about 11 cycles each. The softmax over nt timesteps
therefore costs roughly `4·nt + 11·nt` cycles plus the pipeline fills. The
exp pipeline alone is about 130 cycles deep.

**Rounding.**
- QUANT rounds half away from zero, like `roundf`. The firmware uses RISC-V
  `fcvt.w.s` with `rmm`.
- A NaN (from 0/0 when a group is all zero) gives −1, like `fcvt.w.s`
  followed by truncation on the CPU.
- Exp results below 2⁻¹²⁶ are flushed to zero, the one difference from
  `m_expf`. Softmax inputs here never reach that range.

## Ports

| Port group | Connects to |
|---|---|
| `y_*` (addr, we, wdata, rdata) | Y RAM port A |
| `v_*` | V RAM port A |
| `xq_*`, `xs_*` | the matmul engine's x RAM write ports |
| `a_*` | the attention unit's A RAM write port |
| `start_i`, `op_i`, `n_i`, `src_v_i`, `src_i`, `src2_i`, `dst_i`, `fparam_i` | the job registers |
| `busy_o`, `done_o` | job status |

| Parameter | Default | Meaning |
|---|---|---|
| `YAddrW` | 15 | Y RAM words (32K) |
| `VAddrW` | 13 | V RAM words (8K) |
| `XLineW` | 6 | x RAM lines (MaxN / 32) |
| `AAddrW` | 12 | A RAM words (4K) |
| `RdLat` | 3 | read latency |

## How the firmware uses it

These functions override weak hooks in `fw/model/model.c`:

| Hook | Steps |
|---|---|
| `mm_quant(qx, x, n)` | The CPU writes x into V[3072..] over the local bus, then QUANT from V. The matmul that follows skips loading x. |
| `ffn_hidden` | w1 into Y[0], w3 into Y[8192], SILU into V[0], QUANT from V. The data never touches the CPU. |
| `att_head` | COPY q (V[6144 + h·hs]) into A, scores, SOFTMAX (`f = √hs`, scratch V[2048]) into A, weighted sum |

## Verification

- `tb_vpu` (`sim/run_vpu.sh`) runs each op on vectors from
  `sim/gen_vpu.py`. The generator computes the expected results in numpy
  float32 in the firmware's order, using `sim/fp_ref.py` for `m_expf`. The
  testbench models the RAMs with registered addresses, like `llm_core`.
  Every word is checked bit for bit.
- `tb_llm_core` section 8 runs QUANT and the softmax chain through the
  registers.

## Resources

About 8.1k LUTs, 8.2k FFs and 28 DSPs:
- `fp32_exp`: 4.8k LUTs, 24 DSPs (its multiplies);
- `fp32_div`: 1.9k LUTs.
