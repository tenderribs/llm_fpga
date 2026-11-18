# Simulation (Verilator)

Every hardware unit is simulated against reference results and checked bit
for bit; nothing uses tolerances. The tests need Verilator 5.x with
`--timing`, Python with numpy, and the golden dumps from
`ref/build_ref.sh` for the matmul tests.

## Running

```sh
sim/run_qmv.sh                          # float units, engine, top_qmv, llm_core
sim/run_attn.sh                         # attention unit
sim/run_vpu.sh                          # vector unit
sim/run_tb.sh tb_sd_ctrl sdhc -GSdhc=1  # microSD controller, SDHC card
sim/run_tb.sh tb_sd_ctrl sdsc -GSdhc=0  # ... SDSC card
sim/run_verilator.sh 2 4 16             # systolic array, N = 2, 4, 16
TB=tb_top_gemm sim/run_verilator.sh 2 4 16   # top_gemm through its UART pins
```

Each test prints `TEST PASSED`, or `MISMATCH` lines and `FAIL: n
mismatches`. The scripts exit non-zero on any failure.

**`sim/run_tb.sh <tb> <tag> [-G...]`** builds and runs one testbench:
- It compiles `src/sim/<tb>.sv` with every file in `src/sources/` and the
  simulation models in `src/sim/` (the files not named `tb_*`).
- Output goes to `sim/build/verilator/<tb>/<tag>/`.
- `VERBOSE=1` prints the whole log.
- `TRACE=1` writes `out.fst` for GTKWave.

`top_llm.sv` and `top_soc.sv` are left out because they need vendor IP.
`tb_llm_core` covers everything in them except the DDR3 controller and the
CPU. `sim/BUFG.v` and `sim/IBUFDS.v` are stand-ins for the Xilinx
primitives the other tops use.

## What `run_qmv.sh` runs

```sh
sim/run_qmv.sh [dump index ...]     # default: 000 004 006 042
```

| Step | Testbench | Checks |
|---|---|---|
| 1 | `tb_fp32` | add, multiply, int-to-float on 100k vectors (including rounding ties), divide, and exp against the firmware's `m_expf` (`sim/gen_qmv.py fp`) |
| 2 | `tb_qmv` | the engine on `runq.c` matmul dumps (000 = layer 0 wq 288 × 288, 004 = w1 768 × 288, 006 = w2 288 × 768, 042 = classifier 32000 × 288) plus a random 256 × 512 matmul with group size 64; random stalls in the weight stream |
| 3 | `tb_top_qmv` | `top_qmv` through its UART pins, including the idle timeout |
| 4 | `tb_llm_core` | `llm_core` against the AXI memory and SD card models; see [hardware/llm_core.md](hardware/llm_core.md#verification) for its nine sections |

## Testbenches (`src/sim/`)

| Testbench | Device under test | Vectors from |
|---|---|---|
| `tb_fp32.sv` | `fp32_add`, `fp32_mul`, `int_to_fp32`, `fp32_div`, `fp32_exp` | `sim/gen_qmv.py fp` |
| `tb_qmv.sv` | `qmv` | `sim/gen_qmv.py mm` / `rand-dump` |
| `tb_top_qmv.sv` | `top_qmv` | the same |
| `tb_attn.sv` | `attn` | `sim/gen_attn.py` |
| `tb_vpu.sv` | `vpu` (with Y/V RAM models that register the address, like `llm_core`) | `sim/gen_vpu.py` |
| `tb_llm_core.sv` | `llm_core` | `gen_qmv.py`, `gen_attn.py`, `gen_vpu.py xf` |
| `tb_sd_ctrl.sv` | `sd_ctrl` | (self-contained) |
| `tb_systolic_array.sv` | `systolic_array` | `sim/gen_stim.py` |
| `tb_top_gemm.sv` | `top_gemm` | `sim/gen_stim.py` |

The top-level testbenches run the UART at 8 clocks per bit to keep the
simulations short.

## Simulation models (`src/sim/`)

| Model | What |
|---|---|
| `axi_mem_model.sv` | AXI4 memory with random stalls on every channel; honours write strobes; the testbenches load it directly (e.g. weights in qmv DDR format) |
| `sd_card_model.sv` | SPI-mode SD card (SDHC or SDSC) with random delays; flags protocol violations |

## Vector generators (`sim/`)

| Script | Writes |
|---|---|
| `gen_qmv.py fp OUT [--count N]` | float unit vectors: `add.hex`, `mul.hex`, `i2f.hex`, plus divide and exp; subnormal, inf and NaN cases left out |
| `gen_qmv.py mm OUT dump.bin` | a `runq.c` matmul laid out for the engine: `cfg.hex`, `xq.hex`, `xs.hex`, `w.hex` (beats in stream order), `y.hex` (expected, from `runq.c`) |
| `gen_qmv.py rand-dump PATH n d gs` | a random matmul in `runq_dump` format, for group sizes the model doesn't use |
| `gen_attn.py OUT` | one directory per attention case (`a.hex`, `beats.hex`, `y.hex`); prints each case's testbench parameters |
| `gen_vpu.py OUT` | one directory per vector unit case (`in.hex`, `out.hex`) |
| `gen_vpu.py xf DIR` | float x for a matmul vector set (`tb_llm_core`'s QUANT test) |
| `gen_stim.py` | GEMM inputs and outputs (`iden`: A = identity, C = 0; `rand`) |
| `fp_ref.py` | numpy float32 ports of `fw/model/mathf.c`, bit-exact |

The expected results come from:
- `runq.c` itself (the matmul dumps);
- `ref/q80.py`, which is bit-exact for matmul and quantize;
- numpy float32 with the operations in `runq.c`'s / the firmware's order.

## Fake board

```sh
python3 sim/fake_board.py      # prints a pty path; pass it as --port / -p
```

This is the `top_llm` command set in software on a pseudo-terminal:
STATUS, WRITE, READ, MATMUL, SD_LOAD and SD_STORE. The "card" lives in
memory. MATMUL decodes the qmv DDR format the way the hardware consumes it,
so a host program that lays out or addresses weights wrongly also gets
wrong results here. Use it to test `host/llm_host` or `host/llm_link.py`
without the board.

## Verilator notes

These came up while writing the testbenches:
- **`$bitstoshortreal` loses the sign.** Compare floats as ordered integers
  instead.
- **`2.0 ** n`** caused an internal error. Use integer shifts.
- **Assignment patterns** (`'{...}`) are rejected in port connections. Use
  concatenations.
- **`small` is a SystemVerilog keyword.** Don't use it as a name.
- **Mixing blocking and non-blocking writes** to the same array is an
  error.
- **Outputs before reset** can be random. Gate checks on reset.
