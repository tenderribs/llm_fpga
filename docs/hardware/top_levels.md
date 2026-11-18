# Top levels, the systolic array, helper modules and pins

`make TOP=<name>` picks the design ([build.md](../build.md)).

| Top | Contains | Clock | Host side |
|---|---|---|---|
| `top_soc` | MicroBlaze V + `llm_core` + DDR3 controller | 225 MHz (core), 150 MHz (CPU) | `host/chat.py`, `host/image_tool.py` |
| `top_llm` (default) | `llm_core` + DDR3 controller, no CPU | 225 MHz | `host/llm_host`, `host/ddr_test.py`, `host/sd_tool.py`, `host/image_tool.py` |
| `top_qmv` | matmul engine with weights over the UART, no DDR3 | 200 MHz | `host/qmv_uart.py` |
| `top_gemm` | int8 systolic array GEMM | 200 MHz | `host/gemm_uart.py` |

## `top_soc` (`src/sources/top_soc.sv`)

The full system. It instantiates:
- `u_mig`: the DDR3 controller. It takes the 200 MHz board clock and
  produces `ui_clk` at 225 MHz.
- `u_cpu`: the CPU block design ([cpu_subsystem.md](cpu_subsystem.md)).
- `u_core`: `llm_core` ([llm_core.md](llm_core.md)).

```
llm_core's DDR master  ---> cpu/S_AXI_EXT  \
CPU caches, uncached port ------------------> SmartConnect --> DDR3 controller
CPU peripheral port -------------------------> llm_core registers
CPU second data LMB  ------------------------> llm_core Y / V RAMs (port B)
```

**Address offset.** `llm_core` addresses DDR3 from 0, and the SmartConnect
sees DDR3 at `0x8000_0000`. `top_soc` prepends `2'b10` to `llm_core`'s
30-bit addresses.

**Reset.** The CPU RESET button drives the DDR controller's `sys_rst`. The
controller's `ui_clk_sync_rst`, registered once, resets `llm_core` and the
CPU block design.

**Firmware.** It is built into the bitstream (`make TOP=top_soc fwbit`,
using `updatemem` with `build/top_soc/top_soc.mmi`), or loaded over JTAG
with `xsdb scripts/run_fw.tcl prog.elf`.

## `top_llm` (`src/sources/top_llm.sv`)

The DDR3 controller and `llm_core`, without the CPU. The CPU ports are tied
off. The PC runs the model (`host/llm_host.c`) and sends each matmul's x
over the UART; the weights stay in DDR3. This was the first full-model
design: bit-exact, but limited to 0.7 tokens/s by USB round trips.

The same UART command set works on `top_soc`, so model upload and SD tools
work on both.

## `top_qmv` (`src/sources/top_qmv.sv`)

The matmul engine alone, with W streamed over the UART. It has no DDR3 and
runs on the 200 MHz board clock (`src/constrs/sysclk.xdc`).

The protocol is 8N1 at 1 Mbaud, little-endian.

| Direction | Contents |
|---|---|
| host → FPGA | `n:u16 d:u16`, x.q (n int8), x.s (n/GS float32), then d·n/32 weight beats in qmv stream order, each 32 int8 followed by that row's float32 group scale |
| FPGA → host | d float32 results, rows in order, sent as they are computed |

| LED | Meaning |
|---|---|
| 0 | heartbeat |
| 1 | receiving header / x |
| 2 | streaming W |
| 3 | sending the last results |
| 7:4 | completed jobs (mod 16) |

## `top_gemm` and the systolic array

Sources: `top_gemm.sv`, `systolic_array.sv`, `pe.sv`, `delay_line.sv`.

`top_gemm` computes `Y = A @ B + C` on an N × N weight-stationary systolic
array of processing elements:
1. B is loaded into the PEs and stays there.
2. A flows in from the left (skewed by `delay_line`s), and partial sums
   flow down from the top, starting at C.
3. Rows of Y come out at the bottom, one per cycle.

This design isn't part of the LLM path. It is kept for batched prompt
processing later.

The protocol is 8N1 at 115200 baud.

| Direction | Contents |
|---|---|
| host → FPGA | A, B, C, each N·N signed bytes, row-major |
| FPGA → host | Y, N·N little-endian int32, row-major |

Bytes that arrive while a job is computing or sending are dropped.

| LED | Meaning |
|---|---|
| 0 | heartbeat |
| 1 | receiving |
| 2 | computing |
| 3 | sending |
| 7:4 | completed jobs |

## Behaviour shared by all tops

- **Idle timeout.** A job or command that stalls for 100 ms partway
  through is dropped, so a crashed host script can't leave the board out of
  step.
- **Reset.** The CPU RESET button resets the design.

## Helper modules

| Module | What |
|---|---|
| `fifo.sv` | synchronous FIFO, first-word fall-through; power-of-two depth |
| `skid_buffer.sv` | valid/ready register slice with full throughput (two entries); cuts combinational paths both ways |
| `uart_rx.sv` | 8N1 receiver; samples mid-bit; drops bytes with a bad stop bit |
| `uart_tx.sv` | 8N1 transmitter with valid/ready |
| `delay_line.sv` | shift-register delay (systolic array skew) |
| `pe.sv` | systolic array processing element |

## Pins and constraints (`src/constrs/`)

| File | Contains |
|---|---|
| `genesys2.xdc` | the CPU RESET button (R19), the USB-UART (Y23 TX out, Y20 RX in), 8 LEDs, false paths on these asynchronous I/Os, QSPI configuration settings (×4, 33 MHz, compressed) |
| `sysclk.xdc` | the 200 MHz differential board clock, for designs without the DDR3 controller |
| `top_llm.xdc` | the microSD slot pins (for `top_llm` and `top_soc`; the DDR3 controller constrains its own clock and DDR3 pins) |

The pin assignments are copied from
`board_support/digilent-xdc/Genesys-2-Master.xdc`. Digilent names the UART
pins from the USB bridge's side: the FPGA transmits on `uart_rx_out`.

`src/ip/mig_ddr3.prj` is Digilent's Genesys 2 MIG preset, with the module
name changed. It sets 1800 MT/s and a 256-bit AXI interface at 225 MHz.
