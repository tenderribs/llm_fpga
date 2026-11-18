# Firmware (`fw/`)

The firmware runs on `top_soc`'s MicroBlaze V (RV32IMF) and is built with
the RISC-V GCC that ships with Vivado. The model code also builds for x86 as
the golden reference.

## Layout

| Directory | What | Builds |
|---|---|---|
| `fw/model/` | the model: `runq.c`'s `forward()`, tokenizer and greedy sampler over a model image; float helpers | linked into `llm.elf` and the goldens |
| `fw/llm/` | the board program: hardware hooks, the interactive loop | `build/llm.elf` |
| `fw/golden/` | the x86 host program around `fw/model/` | `build/golden`, `build/golden_libm` |
| `fw/bringup/` | bring-up test: CPU, FPU, DDR3, one engine matmul | `build/bringup.elf` |
| `fw/membench/` | memory latency probe | `build/membench.elf` |
| `fw/fpubench/` | FPU latency and throughput probe | `build/fpubench.elf` |
| `fw/common/` | startup, linker script, newlib hooks, register header | (shared) |

## Building

```sh
make -C fw            # every board program -> fw/build/<name>.elf
make -C fw llm        # one program
make -C fw golden     # x86: fw/build/golden and fw/build/golden_libm
```

| Setting | Value |
|---|---|
| Toolchain | `CROSS ?= /tools/Xilinx/2026.1/gnu/riscv/lin/bin/riscv64-unknown-elf-` |
| Architecture | `-march=rv32imf_zicsr -mabi=ilp32f` |
| Flags | `-O2 -ffp-contract=off -fno-math-errno`, sections garbage-collected |

**The two flags that matter:**
- `-ffp-contract=off` forbids fused multiply-add. Every float multiply and
  add is rounded separately, as in `runq.c` on a CPU without FMA, which is
  required for bit-exact results.
- `-fno-math-errno` lets `sqrtf` compile to the FPU instruction, which is
  correctly rounded like the library's.

## Memory layout

| Region | Address | Holds |
|---|---|---|
| Local memory, 256 KB (`fw/common/link.ld`) | `0x0000_0000` | `_start` at 0 (the reset address), `.text`, `.rodata`, `.data`, `.bss`, a 16 KB stack. `llm.elf` must fit: the trace buffer was moved to the heap to make room |
| Heap (`fw/common/syscalls.c`) | `0x9000_0000`, 256 MB, cached DDR3 (DDR offset `0x1000_0000`) | `malloc` |
| Fast arena (`fw/llm/main.c`) | 48 KB in local memory | the run state's small buffers (x, xb, q, k, v, …) via the weak hook `fast_calloc`; plain `calloc` once it is full |
| RMSNorm weights | copied into local memory at startup | read every layer |
| KV cache | DDR offset `0x2000_0000` (CPU `0xA000_0000`, uncached) | written by DMA, read by the attention unit; the CPU never touches it |

The linker discards `.note.gnu.build-id` (`-Wl,--build-id=none`), which
would otherwise sit at address 0 ahead of `_start`.

**Startup** (`fw/common/start.S`):
1. Set the global pointer and the stack.
2. Enable the FPU: `mstatus.FS` = Dirty, `fcsr` = 0 (round to nearest even).
3. Clear `.bss`.
4. Run the constructors, then `main`.

**Newlib hooks** (`syscalls.c`):
- `_write` sends bytes to `llm_core`'s CONSOLE register, waiting while the
  console FIFO is full.
- `_sbrk` grows the heap in DDR3.
- Input is not done through stdio. `fw/llm/main.c` reads CONSOLE_IN
  directly.

## The model code (`fw/model/`)

| File | Contents |
|---|---|
| `model.h` | the image header, Config, RunState, the hook declarations |
| `model.c` | `model_open`, `run_state_alloc`, `forward()`, `sample_argmax`, and the weak CPU defaults of every hook |
| `tokenizer.c` | `runq.c`'s BPE tokenizer, reading `tokenizer.bin` from the image |
| `mathf.c` | `m_expf`, `m_expf4`, `m_sincosf`, `m_roundf` |
| `trace.h` | FNV-1a hash over the bits of the logits |

**How `forward()` stays bit-exact.**
- It performs `runq.c`'s float operations in `runq.c`'s order.
- The loops are unrolled to work on four independent elements at a time,
  so the pipelined FPU overlaps them. Each element still gets `runq.c`'s
  operations, and every sum still adds in `runq.c`'s order.
- The RoPE sin/cos tables are computed once per token.
- Quantize keeps four divisions in flight and rounds with `fcvt.w.s` and
  `rmm` (`ROUNDI`).

**Math functions.**
- By default `expf`, `sinf`, `cosf` and `powf` come from `mathf.c`. These
  are float-only, fixed-order and about 1-2 ulp accurate. They give the
  same bits on x86 and on the board. The C libraries' versions (glibc vs
  newlib) differ from each other, which would break the golden.
- With `-DMODEL_LIBM`, the C library's functions are used instead. The x86
  build must then reproduce `runq.c` exactly. `golden_libm` checks that
  the port itself is faithful.

### Hooks

`model.c` defines each hook as a weak function with a CPU implementation. A
platform overrides it with the same results:

| Hook | CPU default | Board override |
|---|---|---|
| `qmatmul(y, x, w_off, n, d)` | (none; the platform provides it) | `fw/llm/main.c`: loads x unless the x RAM already holds it, runs the matmul, reads results over the LMB (skipped for the classifier unless tracing) |
| `mm_quant(qx, x, n, gs)` | `quantize()` | `vpu_hw.c`: x into the V RAM, vector unit QUANT |
| `ffn_hidden(m, s, l)` | w1, w3, SwiGLU, quantize | `vpu_hw.c`: w1 → Y[0], w3 → Y[8192], SILU, QUANT |
| `att_store_kv(s, p, l, pos)` | copy into the RunState KV cache | `attn_hw.c`: k/v/q into the V RAM, DMA into the DDR3 KV cache |
| `att_head(s, p, l, h, kvh, pos, sqrt_hs)` | scores, scale, softmax, weighted sum | `attn_hw.c`: COPY, scores, SOFTMAX, weighted sum, read xb |
| `att_scores`, `att_wsum` | the two loops | (subsumed by `att_head`) |
| `fast_calloc(n, size)` | `calloc` | `main.c`: the 48 KB local-memory arena |

The same `fw/model/` sources build for the board and the host. Any hook can
fall back to the CPU by removing its override, which is how each hardware
unit was brought up and checked.

## The board program (`fw/llm/`)

| File | Contents |
|---|---|
| `main.c` | startup, `qmatmul` / `mm_run`, the interactive loop, profiling |
| `vpu_hw.c` | `vpu_run`, `mm_quant`, `ffn_hidden` |
| `attn_hw.c` | KV cache addressing, `dma`, `att_store_kv`, `att_head` |

### Startup

1. Print a banner.
2. Poll STATUS until DDR3 is calibrated and the SD load is done (SD_OK or
   SD_ERR). Straight from configuration, the CPU starts before either.
3. `model_open` at DDR3 offset 0 checks the header. It requires:
   - magic `"LLMMODEL"`;
   - version 1;
   - group size 32 (the engine's);
   - `n_layers × 7 + 1` matrices.

   If a check fails, the firmware prints `LLM FAIL: <reason>` and stops.
4. Open the tokenizer, copy the RMSNorm weights to local memory, allocate
   the run state.

### Console protocol

The firmware prints `READY` and waits for a line on CONSOLE_IN
(`host/chat.py` sends lines with the UART CONSOLE_IN command):

| Line | Does |
|---|---|
| any text | encode it as the prompt (with BOS), generate greedily until BOS or `steps`, print the text as it comes |
| `/steps N` | set the maximum number of tokens (1..seq_len). **This persists until reset**; `chat.py --steps` sets it |
| `/trace PROMPT` | generate exactly `steps` tokens without stopping at BOS. Copy all logits every step, check the hardware argmax against `sample_argmax`, and print `T pos token hash` lines at the end |

After each run the firmware prints a statistics line, then `LLM DONE`, and
then `READY` again:

```
[168 tokens in 1.02 s: 164.9 tok/s; ms per token: matmul 3.1, attention 1.2, (kv store 0.36),
 quantize 0.3, swiglu 0.1, rmsnorm 0.6, rope 0.4, other 0.3]
```

**What the statistics count:**
- Times come from `llm_core`'s CYCLES counter at 225 MHz, summed per part
  (`prof_cycles[]`, `MODEL_PROFILE`).
- Printing and tracing are excluded.
- "matmul" includes copying the logits out when tracing (~1.3 ms per
  token).
- "kv store" is part of "attention".

## The x86 golden (`fw/golden/golden.c`)

```sh
fw/build/golden models/stories15M.img "Once upon a time" 256 outdir > golden.txt
```

The golden is the same `forward()` over the same image. It does the matmul
in software exactly as the engine does: integer dot products, then the
float steps in group order. It writes `tokens.txt` and `logits.bin` in
`ref/runq_dump`'s format, and prints the same `T` lines as the board.

| Build | Math | Must equal |
|---|---|---|
| `golden` | `mathf.c` | the board's trace, bit for bit |
| `golden_libm` | glibc | `ref/runq_dump`'s logits, bit for bit |

## Other programs

They are loaded over JTAG onto a running `top_soc`:

```sh
make TOP=top_soc program
xsdb scripts/run_fw.tcl fw/build/bringup.elf
python3 host/console.py --until 'PASS|FAIL'
```

| Program | Checks or measures | Ends with |
|---|---|---|
| `bringup` | `llm_core` ID; CPU vs fabric cycle counts; FPU bit patterns (1/3, √2); cached DDR3 (256 KB) and uncached (4 KB) write/read; SD status; layer 0 wq (288 × 288) on the engine vs the CPU, bit for bit (needs an image on the card) | `BRINGUP PASS` / `BRINGUP FAIL` |
| `membench` | CPU cycles per access: D-cache with various strides, uncached, local memory | `MEMBENCH DONE` |
| `fpubench` | FPU latency (dependent chains) and throughput (independent streams), plus the model's float helpers | `FPUBENCH DONE` |

`scripts/run_fw.tcl` looks for the CPU on both JTAG targets
([troubleshooting.md](troubleshooting.md#jtag)), then resets it, downloads
the ELF and runs it.

## Register header (`fw/common/llm_hw.h`)

The header defines the address map, every `llm_core` register (`LLM_*`),
the windows (`LLM_XQ`, `LLM_XS`, `LLM_AT_A`, `LLM_Y`), the local-bus views
(`LLM_YL` for Y, `LLM_VL` for V), the STATUS bits (`ST_*`), the vector unit
ops (`VPU_*`) and the clock rates.
