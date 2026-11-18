# Documentation

llama2.c's TinyStories models (int8 "Q8_0") on a Digilent Genesys 2,
bit-exact with `runq.c`, at about 165 tokens/s for stories15M.

## Where to start

| If you want to... | Read |
|---|---|
| Build it and run a story on the board | [getting_started.md](getting_started.md) |
| Understand the system as a whole | [system_design.md](system_design.md) |
| Know why it's as fast as it is, and no faster | [performance.md](performance.md) |
| Fix something that doesn't work | [troubleshooting.md](troubleshooting.md) |

## All pages

**System**
- [system_design.md](system_design.md) covers the whole system:
  - the hardware / software split;
  - clocks and memories;
  - the job model;
  - numerics;
  - boot;
  - verification;
  - the time budget;
  - resources;
  - design decisions.
- [performance.md](performance.md) has the measurements, how they're taken,
  the speed history and the comparison with a desktop CPU.
- [data_formats.md](data_formats.md) gives the byte layouts:
  - the model image;
  - the microSD card;
  - the qmv DDR weight format;
  - the KV cache;
  - the trace lines.

**Hardware** (`src/sources/`)
- [hardware/llm_core.md](hardware/llm_core.md): the core. It covers the
  UART command protocol, the CPU register map, the RAMs, the DMA and the
  LEDs.
- [hardware/qmv.md](hardware/qmv.md): the int8 matmul engine and the weight
  reader.
- [hardware/attn.md](hardware/attn.md): the fp32 attention unit.
- [hardware/vpu.md](hardware/vpu.md): the fp32 vector unit (quantize,
  SwiGLU, softmax, copy).
- [hardware/float_units.md](hardware/float_units.md): the add, multiply,
  int-to-float, divide and exp pipelines.
- [hardware/sd_ctrl.md](hardware/sd_ctrl.md): the microSD controller.
- [hardware/cpu_subsystem.md](hardware/cpu_subsystem.md): the MicroBlaze V
  block design, its caches and address map.
- [hardware/top_levels.md](hardware/top_levels.md) covers:
  - the four top levels: `top_soc`, `top_llm`, `top_qmv`, `top_gemm`;
  - the systolic array;
  - the small helper modules;
  - the pins.

**Software**
- [firmware.md](firmware.md): `fw/`. It covers the model code and its
  hooks, the board overrides, the console commands, the other programs and
  the memory layout.
- [host_tools.md](host_tools.md): every script and program in `host/`.
- [reference.md](reference.md): `ref/`, the llama2.c golden: quantizing,
  dumping, and the numpy reference.

**Flows**
- [build.md](build.md): Vivado builds, make targets, IP and block design
  generation, programming and flashing.
- [simulation.md](simulation.md): Verilator testbenches, vector generators,
  simulation models, and the fake board.

## Diagrams

All diagrams are in [diagrams/](diagrams/). They are `.drawio.svg` files:
GitHub shows them as images, and draw.io (or the VS Code draw.io extension)
opens them for editing. `diagrams/make_diagrams.py` generates them. After
editing a diagram in draw.io, keep editing it there, because rerunning the
script overwrites it.

| Diagram | Shows |
|---|---|
| [system](diagrams/system.drawio.svg) | Host, the CPU subsystem, `llm_core`, DDR3, flash, microSD |
| [llm_core](diagrams/llm_core.drawio.svg) | `llm_core`'s blocks and data paths |
| [token_flow](diagrams/token_flow.drawio.svg) | One token's forward pass, coloured by which unit does each step |
| [memory](diagrams/memory.drawio.svg) | The CPU's address map and DDR3's contents |
| [clocks](diagrams/clocks.drawio.svg) | Clocks, resets, clock-domain crossings |
| [budget](diagrams/budget.drawio.svg) | Where a token's 6.1 ms go, and the DDR3 floor |
| [boot](diagrams/boot.drawio.svg) | Power-on to a story |
| [verification](diagrams/verification.drawio.svg) | The chain of bit-exact goldens, and the testbenches |

## Repository layout

| Path | Contents |
|---|---|
| `src/sources/` | RTL (SystemVerilog) |
| `src/sim/` | Testbenches and simulation models |
| `src/constrs/` | Pin and timing constraints |
| `src/ip/` | DDR3 controller configuration (`mig_ddr3.prj`) |
| `scripts/` | Vivado Tcl: build, IP, CPU block design, firmware loading |
| `fw/` | Firmware for the MicroBlaze V, and its x86 golden build |
| `host/` | PC-side tools: chat, model upload, tests |
| `ref/` | llama2.c golden reference and the numpy reference |
| `sim/` | Simulation scripts and test vector generators |
| `docs/` | This documentation |
| `third_party/llama2.c` | Karpathy's llama2.c (submodule) |
| `board_support/` | Digilent board files and master XDC (submodules) |
| `models/` | Downloaded and generated model files (not in git) |
| `build/` | Vivado output (not in git) |
