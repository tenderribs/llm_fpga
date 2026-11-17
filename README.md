# LLM on FPGA

Goal: run a Llama 2-architecture model on a Digilent Genesys 2 board
(Kintex-7 `xc7k325tffg900-2`), built up from a hand-written int8 GEMM engine.

Current state: `Y = A @ B + C` on an N×N systolic array (default N = 16), fed
over the board's USB-UART from a host script. `A`, `B` and `C` are signed
8-bit, `Y` is signed 32-bit.

## Architecture

The core is a weight-stationary systolic array of N×N processing elements (PEs). `B` is loaded into the PEs first and stays there. Then `A` flows in from the left and partial sums flow down from the top, starting at `C`. Rows of `Y` come out at the bottom, one per cycle.

`top` wraps the array with a UART receiver and transmitter. The host sends
`A`, `B`, `C` (N·N bytes each, row-major); the FPGA computes and sends back `Y`
(N·N little-endian int32, row-major). If the host stops partway through a job
for 100 ms, the partial job is dropped, so a crashed host script can't leave the
board out of step.

LEDs: 0 heartbeat, 1 receiving, 2 computing, 3 sending, 7:4 completed jobs
(mod 16). The CPU RESET button resets the design.

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

## Simulation (Verilator)

```sh
sim/run_verilator.sh 2 4 16            # systolic array alone
TB=tb_top sim/run_verilator.sh 2 4 16  # whole design, through the UART pins
```

For each size the script regenerates the test vectors with `sim/gen_stim.py`,
builds the testbench once per mode in `sim/build/verilator/<TB>/N<size>_<mode>/`,
runs it and prints `TEST PASSED` / `MISMATCH` / `FAIL` / `TIMEOUT` lines.

The vector modes are `iden` (`A` = identity, `C` = 0, so `Y = B`) and `rand`.
The seed is fixed, so the vectors are the same on every run.

`tb_top` runs the UART at 8 clocks per bit to keep the sim short. It first
sends a few stray bytes to check the idle timeout, then two full jobs.

## FPGA build (Vivado)

```sh
source /tools/Xilinx/2026.1/Vivado/settings64.sh

make            # synth -> impl -> bitstream (build/top.bit)
make impl       # stop after place and route
make synth      # stop after synthesis
make program    # load build/top.bit over the JTAG USB port (not to flash)
```

Each stage reruns only when its inputs have changed. All output goes to `build/`: checkpoints, logs, and the utilization and timing reports.

## Running on the board

Plug in both micro-USB ports: **JTAG** (programming) and **UART**. Then:

```sh
make program
python3 host/gemm_uart.py --port /dev/ttyUSB0 --n 16
```

The first job uses `A` = identity, the rest are random; each is checked
against numpy. If the UART shows up under another name, check `dmesg` after
plugging it in. Your user needs to be in the `dialout` group.
