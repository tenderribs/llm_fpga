# LLM on FPGA

Goal: run a Llama 2-architecture model (Karpathy's llama2.c TinyStories
models, starting with stories15M) on a Digilent Genesys 2 board (Kintex-7
`xc7k325tffg900-2`), using llama2.c's int8 "Q8_0" format.

Current state:

- `top_qmv`: the int8 matrix-vector engine that token generation needs, fed
  over the USB-UART. Every per-layer matmul of stories15M run on the board
  matches llama2.c's `runq.c` bit-for-bit.
- `top_gemm`: `Y = A @ B + C` on an N×N int8 systolic array (default N = 16),
  also over the UART. The array is kept for prompt processing later, where
  the work is a real matrix-matrix product.

## Architecture

### Quantized matrix-vector engine (`qmv.sv`)

Computes `y = W @ x` exactly as `runq.c`'s `matmul()` does: for each row, per
group of 32 (or 64) inputs, an int32 dot product of int8 values, then
`val += ((float)ival * w_scale) * x_scale` in float32, in group order. The
float units (`fp32_mul`, `fp32_add`, `int_to_fp32`) round to nearest even
like the CPU, which is what makes the results bit-exact. They flush
subnormals to zero, a range these values never reach.

`x` sits in on-chip RAM; `W` streams in at 32 bytes per beat (the DDR3
interface width, for later) with each beat's group scale. Each row's float
sum is a chain of dependent adds, so the engine works on 16 rows
round-robin to hide the adder pipeline and takes a beat every cycle. The host
lays `W` out in that order: `qmv_stream()` in `ref/q80.py`.

`top_qmv` protocol (1 Mbaud): the host sends `n`, `d` (u16), `x.q`, `x.s`,
then the weight beats (32 int8 + float32 scale each); the FPGA streams back
`d` float32 results as they are computed. See `src/sources/top_qmv.sv`.

### Systolic array (`systolic_array.sv`)

A weight-stationary systolic array of N×N processing elements (PEs). `B` is loaded into the PEs first and stays there. Then `A` flows in from the left and partial sums flow down from the top, starting at `C`. Rows of `Y` come out at the bottom, one per cycle.

`top_gemm` protocol (115200 baud): the host sends `A`, `B`, `C` (N·N bytes
each, row-major); the FPGA sends back `Y` (N·N little-endian int32,
row-major).

### Both tops

If the host stops partway through a job for 100 ms, the job is dropped, so a
crashed host script can't leave the board out of step. LEDs: 0 heartbeat,
1-3 job phase, 7:4 completed jobs (mod 16). CPU RESET resets the design.

## Required software

| Tool                             | Version                          | Used for                             |
| -------------------------------- | -------------------------------- | ------------------------------------ |
| Vivado (Basic license is enough) | 2026.1                           | synthesis, implementation, bitstream |
| Verilator                        | 5.x (tested with 5.020, 5.052)   | simulation; needs `--timing` support |
| zlib headers (`zlib1g-dev`)      | any                              | Verilator FST waveform traces        |
| Python                           | 3.x + `numpy`, `pyserial`        | test vectors, host script            |
| GNU Make, Bash                   | any recent                       | build and sim scripts                |

Hardware: Digilent Genesys 2 FPGA board.

## Setup

```sh
git submodule update --init

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`board_support/` holds Digilent's board files and master XDC as submodules.
The constraints this design uses are in `src/constrs/genesys2.xdc`.
`third_party/llama2.c` is Karpathy's llama2.c, used as the golden reference.

## Golden reference (`ref/`)

```sh
mkdir -p models && curl -L -o models/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
ref/build_ref.sh                 # quantize, build runq_dump, dump golden data
python3 ref/check_ref.py         # check the numpy reference against it
```

- `export_q80.py`: numpy port of llama2.c's Q8_0 export (no PyTorch needed);
  writes `models/stories15M_q80.bin` (group size 32, since 288 isn't a
  multiple of 64).
- `runq_dump.c`: runs `runq.c` greedily and dumps the logits of every step
  plus every `matmul()` call at one position (43 for stories15M: 6 layers ×
  7, plus the classifier) to `ref/build/stories15M/`.
- `q80.py`: numpy reference. `matmul()` and `quantize()` are bit-exact with
  `runq.c`; `forward()` is close but not exact (numpy's float sums and
  `exp`/`sqrt` differ by an ulp now and then, and once one tips an
  activation across a rounding boundary the KV cache carries it), so
  `runq.c` itself stays the full-model golden.

## Simulation (Verilator)

```sh
sim/run_qmv.sh                          # matmul engine (needs ref/build_ref.sh)
sim/run_verilator.sh 2 4 16             # systolic array alone
TB=tb_top_gemm sim/run_verilator.sh 2 4 16   # top_gemm through the UART pins
```

`run_qmv.sh` runs:

1. `tb_fp32`: the float units against numpy float32, on 60k vectors
   including exact rounding ties (`sim/gen_qmv.py fp`)
2. `tb_qmv`: the engine on real `runq.c` matmuls (layer 0 wq, w1, w2 and the
   32000-row classifier) plus a random group-size-64 one, every output
   checked bit-exact, with random stalls in the weight stream
3. `tb_top_qmv`: `top_qmv` through its UART pins, including the idle timeout

`run_verilator.sh` regenerates the GEMM vectors with `sim/gen_stim.py` (`iden`:
`A` = identity, `C` = 0, so `Y = B`; `rand`: all random) and runs the chosen
testbench per size. `sim/run_tb.sh <tb> <tag> [-G...]` builds and runs any
single testbench. The top-level testbenches run the UART at 8 clocks per bit
to keep the sims short.

## FPGA build (Vivado)

```sh
source /tools/Xilinx/2026.1/Vivado/settings64.sh

make                 # top_qmv: synth -> impl -> bitstream
make TOP=top_gemm    # the GEMM design instead
make impl            # stop after place and route
make synth           # stop after synthesis
make program         # load the bitstream over the JTAG USB port (not to flash)
```

Each stage reruns only when its inputs have changed. Output goes to
`build/<TOP>/`: checkpoints, logs, and the utilization and timing reports.
Both designs meet timing at 200 MHz.

## Running on the board

Plug in both micro-USB ports: **PROG** (JTAG) and **UART**. Then:

```sh
make program
python3 host/qmv_uart.py                 # layer 0 matmuls; or pass dump indices
make TOP=top_gemm program
python3 host/gemm_uart.py --n 16
```

If `make program` reports "No devices detected" although the board is on,
power-cycle the board. If the UART shows up under another name than
`/dev/ttyUSB0`, check `dmesg` after plugging it in. Your user needs to be in
the `dialout` group.
