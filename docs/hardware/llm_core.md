# `llm_core`

Source: `src/sources/llm_core.sv` (the header comment is the authoritative
register and command list). Diagram:
[llm_core.drawio.svg](../diagrams/llm_core.drawio.svg).

`llm_core` is everything except the DDR3 controller and the CPU. It is kept
separate from both so that it can be simulated with an AXI memory model
(`src/sim/tb_llm_core.sv`).

It contains:
- the UART receiver and transmitter, and the command state machine;
- the CPU register interface: an AXI4-Lite slave;
- the matmul engine (`qmv`), the attention unit (`attn`) and the vector
  unit (`vpu`);
- the weight reader (`wstream_reader`), which streams weights, K and V
  from DDR3;
- the Y RAM, the V RAM, the console FIFOs and the result FIFO;
- the microSD controller (`sd_ctrl`);
- the AXI4 master (256-bit) to DDR3, shared by the SD controller, the
  command logic, the reader and the DMA;
- the LEDs.

It all runs on one clock: `clk_i`, the DDR controller's 225 MHz user clock.
The only other clock is `yb_clk_i`, port B of the Y/V RAMs, which runs on
the CPU's local memory bus.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `ClkHz` | 225 000 000 | Clock frequency, for the UART and the timeouts |
| `Baud` | 3 000 000 | UART rate |
| `IdleTimeoutClks` | `ClkHz / 10` | A command stalled this long (100 ms) is dropped |
| `GS` | 32 | Quantization group size |
| `MaxN` | 2048 | Longest matmul input (x RAM size) |
| `AddrW` | 30 | AXI address width (1 GB) |
| `SdSlowHalf`, `SdFastHalf`, `SdPowerClks` | | microSD timing, see [sd_ctrl.md](sd_ctrl.md); shortened in simulation |

## UART command protocol

8N1 at `Baud`, little-endian. Each command starts with an opcode byte:

| Op | Command | Arguments | Reply |
|---|---|---|---|
| `0x00` | STATUS | none | 12 bytes, see below |
| `0x01` | WRITE | `addr:u32 beats:u32`, then `beats × 32` bytes | `0x01` once everything is written |
| `0x02` | READ | `addr:u32 beats:u32` | `beats × 32` bytes |
| `0x03` | MATMUL | `addr:u32 n:u16 d:u16`, then x.q (n bytes) and x.s (n/32 float32) | d float32 results, rows in order, sent as they are computed |
| `0x04` | SD_LOAD | none | `0x04`, then copies the card image into DDR3 |
| `0x05` | SD_STORE | `addr:u32 nbytes:u32` | `0x05`, then copies DDR3 to the card |
| `0x06` | CONSOLE_IN | `n:u8`, then n bytes | none; the bytes go to the CPU's CONSOLE_IN FIFO |

Requirements on the arguments:
- WRITE and READ addresses are 32-byte aligned.
- MATMUL's `addr` is 4 KB aligned, and W (d × n) is stored there in qmv DDR
  format ([data_formats.md](../data_formats.md#qmv-ddr-format)).

The **STATUS reply** is 12 bytes:

| Bytes | Contents |
|---|---|
| 0-2 | `"LLM"` |
| 3 | flags: bit 0 DDR calibrated, 1 SD busy, 2 SD ok, 3 SD error, 4 card-detect pin |
| 4 | SD error code |
| 5 | SD's last R1 response |
| 6-7 | 0 |
| 8-11 | `sd_image_bytes:u32` |

Rules the command logic follows:
- **Boot load.** SD_LOAD runs by itself once DDR3 has calibrated after
  reset.
- **During SD operations** only STATUS is answered. Other commands are
  dropped.
- **Idle timeout.** If the host stops partway through a command for
  `IdleTimeoutClks`, the command is dropped and the next byte is read as an
  opcode. A crashed host script therefore can't leave the board out of
  step. `host/llm_link.py` waits 0.2 s after opening the port for this
  reason.
- **Console output** (the CPU's `CONSOLE` writes) goes out only *between*
  commands, so it never interleaves with a reply.
- **CPU jobs and UART commands exclude each other.** A CPU job starts only
  when no UART command is running, and it holds off new commands until it
  is done.

## CPU register map

AXI4-Lite slave, byte offsets from `0x4000_0000` in the CPU's map
(`fw/common/llm_hw.h` has the C macros). Reads take 2 cycles.

| Offset | Name | Access | Meaning |
|---|---|---|---|
| `0x00000` | ID | R | `"LLM1"` = `0x314D4C4C` |
| `0x00004` | STATUS | R | 0 DDR calibrated, 1 SD busy, 2 SD ok, 3 SD error, 7:4 SD error code, **8 busy** (any CPU job), 9 console full, 10 CONSOLE_IN has a byte |
| `0x00008` | CONSOLE | W | a byte for the UART (dropped when full: poll bit 9) |
| `0x0000C` | CONSOLE_IN | R | bit 8 set: bits 7:0 are the next byte from the host, and the read takes it; clear: nothing waiting |
| `0x00010` | MM_ADDR | W | DDR3 offset of W (qmv DDR format, 4 KB aligned) |
| `0x00014` | MM_ND | W | `n \| d << 16` |
| `0x00018` | MM_START | W | any write: run y = W x into Y from `MM_YOFF` |
| `0x0001C` | SD_BYTES | R | size of the image the card loaded |
| `0x00020` | SD_ADDR | R | where it was loaded |
| `0x00024` | CYCLES | R | free-running 225 MHz cycle counter |
| `0x00030` | AT_ADDR | W | DDR3 offset of the head's K or V (2 KB aligned) |
| `0x00034` | AT_CFG | W | `head_size \| timesteps << 16` |
| `0x00038` | AT_START | W | 0: scores q·k_t into Y[t]; 1: weighted sum Σ a_t v_t into Y[i] |
| `0x0003C` | ARGMAX | R | index of the largest result of the last CPU matmul (the first if tied; −0 == +0) |
| `0x00044` | MM_YOFF | W | first Y word for a CPU matmul's results |
| `0x00050` | VPU_OP | W | starts a vector unit job: bits 1:0 op (0 COPY, 1 QUANT, 2 SILU, 3 SOFTMAX); bit 8: QUANT reads V (else Y) |
| `0x00054` | VPU_N | W | elements |
| `0x00058` | VPU_SRC | W | source word (Y or V) |
| `0x0005C` | VPU_SRC2 | W | SILU's second source (Y) |
| `0x00060` | VPU_DST | W | destination or scratch (V) |
| `0x00064` | VPU_F | W | SOFTMAX's divisor, as float32 bits (√head_size) |
| `0x00068` | DMA_SRC | W | first V word |
| `0x0006C` | DMA_N | W | words (a multiple of 8 for stride 4) |
| `0x00070` | DMA_DST | W | DDR3 byte address (32-byte aligned for stride 4) |
| `0x00074` | DMA_STRIDE | W | DDR3 bytes between consecutive words |
| `0x00078` | DMA_START | W | any write: run it |
| `0x01000` | XQ | W | matmul x.q: word i = bytes 4i..4i+3; write in order (every 8th word completes a 32-byte line) |
| `0x01800` | XS | W | matmul x.s: word g = group g's scale |
| `0x04000` | AT_A | W | attention A RAM, up to 4096 words |
| `0x20000` | Y | R | Y RAM, up to 32768 words (slow; use the local-bus port) |

**Job protocol.**
1. Write the parameters.
2. Write the start register (`MM_START`, `AT_START`, `VPU_OP` or
   `DMA_START`).
3. Poll `STATUS` until bit 8 clears.

Only one job runs at a time. There are no interrupts.

## Y and V RAMs: the CPU's local-bus port

The `yb_*` port is a block RAM port on the CPU's clock, wired to a second
data-side LMB slave in the block design. Its data arrives one cycle after
`yb_en_i`.

| CPU address | Byte address bit 17 | RAM | Size | CPU access |
|---|---|---|---|---|
| `0x0010_0000` | clear | Y RAM | 32K words | read only |
| `0x0012_0000` | set | V RAM | 8K words | read and write |

These are true dual-port block RAMs:
- **Port A** (225 MHz) belongs to the units:
  - the matmul engine and attention unit write results into Y;
  - the vector unit reads Y and V and writes V;
  - the DMA reads V.
- **Port B** is the CPU's.

The address and data of port A are registered before the RAM, so the vector
unit sees read data 3 cycles after the address (`RdLat = 3`).

There is no handshake between the ports. The CPU only touches results after
it has seen busy drop through the register path, which is slower than the
RAM path.

The firmware's V RAM layout (`fw/llm/vpu_hw.c`, `attn_hw.c`):

| V words | Use |
|---|---|
| 0-2047 | SwiGLU output (hidden_dim) |
| 2048-3071 | softmax scratch (scaled scores) |
| 3072-4095 | x for quantize |
| 4096 / 5120 / 6144 | k / v / q staging for the KV DMA and COPY |

In Y, w1's results go at 0 and w3's at 8192.

## Jobs in detail

**Matmul** (`S_CPU_PREP` → `S_CPU_RUN`)
- The weight reader streams W from `MM_ADDR` in qmv DDR format into
  `qmv`.
- Each result goes into Y at `MM_YOFF + row`.
- Argmax is tracked as results come out.
- x comes from the XQ/XS windows or from the vector unit's QUANT. The x RAM
  keeps its contents between jobs, so two matmuls with the same input (q, k,
  v or w1, w3) load it once.

**Attention** (`S_CPU_PREP` → `S_CPU_ATT`)
- The reader runs in raw mode (no scale beats).
- The beat count is computed from `AT_CFG`:
  - scores: `ceil(nt/128) × hs × 16`;
  - weighted sum: `nt × hs / 8`.
- Results go into Y by index.
- See [attn.md](attn.md).

**Vector unit** (`S_CPU_VPU`): one job per `VPU_OP` write. See
[vpu.md](vpu.md).

**DMA** (`S_DMA_RD` → `S_DMA_AXI` → `S_DMA_DRAIN`) copies V RAM words to
DDR3:
- **Stride 4:** 8 consecutive words make one whole 32-byte beat. Used for
  V rows.
- **Other strides:** one word per AXI write, at `DMA_DST + k × stride`,
  with a 4-byte strobe. Used for K, which is stored transposed (stride 512
  bytes).
- Up to 15 writes are in flight. The job ends when every B response is
  back.

## AXI master and arbitration

| Who owns the port | When |
|---|---|
| SD controller | while an SD operation runs (single-beat bursts, all strobes set) |
| Command logic | the rest of the time; the reader, UART READ/WRITE and DMA go through it |

- Ownership changes only while both sides are idle.
- AR goes through a skid buffer (`u_ar_slice`) for timing. A request waiting
  in it always belongs to the current owner.
- The reader issues 2 KB bursts (64 beats) and keeps up to 8 in flight.
  Its 512-beat FIFO always has room for every beat it has asked for, so R
  is never stalled.

## Reset, LEDs

`rst_ni` comes from the DDR controller's synchronized reset, so the CPU
RESET button resets `llm_core` too.

| LED | Meaning |
|---|---|
| 0 | heartbeat |
| 1 | DDR3 calibrated |
| 2 | a matmul or attention job is running |
| 3 | SD: blinks while busy, on when an image loaded or stored fine |
| 7:4 | the SD error code after an SD error, else completed matmuls (mod 16) |

## Verification

`src/sim/tb_llm_core.sv` (run by `sim/run_qmv.sh`) drives `llm_core`
through its UART pins and its CPU ports. An AXI memory model with random
stalls stands in for DDR3, and an SD card model for the card. The test
sequence:

1. Idle timeout.
2. STATUS.
3. WRITE / READ.
4. MATMULs on real `runq.c` matmuls.
5. An SD store / load round trip.
6. CPU matmul through XQ/XS, Y read through both ports, ARGMAX,
   CONSOLE_IN.
7. Attention scores and weighted sum.
8. Vector unit QUANT, matmul at MM_YOFF, then the chain scores →
   SOFTMAX → weighted sum.
9. DMA, whole-beat and scattered.

Every result is checked bit for bit.
