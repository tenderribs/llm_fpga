# Getting started

This page takes you from a fresh clone to a story generated on the board,
plus the check that its output is bit-exact.

## 1. What you need

**Hardware**
- A Digilent Genesys 2 (Kintex-7 `xc7k325tffg900-2`).
- A microSD card. Any size works; the stories15M image is 28 MB.
  **Everything on the card is overwritten.**
- Two micro-USB cables: one for **PROG** (JTAG) and one for **UART**.

**Software**

| Tool | Version | Used for |
|---|---|---|
| Vivado (the Basic license is enough) | 2026.1 | synthesis, implementation, bitstream, programming; its RISC-V GCC builds the firmware |
| Verilator | 5.x (tested with 5.020 and 5.052) | simulation; needs `--timing` |
| zlib headers (`zlib1g-dev`) | any | Verilator FST waveforms |
| Python | 3.x with `numpy` and `pyserial` | reference, test vectors, host tools |
| GCC, GNU Make, Bash | any recent | reference build, x86 golden, scripts |

Your user must be in the `dialout` group to open `/dev/ttyUSB*`.

The paths assume Vivado is installed in `/tools/Xilinx/2026.1`. If yours is
elsewhere, override `CROSS` in `fw/Makefile`, and source your own
`settings64.sh`.

## 2. Set up the repository

```sh
git submodule update --init          # llama2.c, Digilent board files
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Build the reference (golden) data

```sh
mkdir -p models && curl -L -o models/stories15M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
ref/build_ref.sh                 # quantize to Q8_0, build runq_dump, dump golden data
python3 ref/check_ref.py         # optional: check the numpy reference against it
```

This writes:
- `models/stories15M_q80.bin`, the int8 model;
- `ref/build/stories15M/`, with every step's logits and the 43 matmuls at
  one position.

See [reference.md](reference.md).

## 4. Build the model image

```sh
python3 host/build_image.py models/stories15M_q80.bin \
    third_party/llama2.c/tokenizer.bin models/stories15M.img
```

The image is everything the firmware needs, laid out as it sits in DDR3.
See [data_formats.md](data_formats.md#model-image).

## 5. Build the bitstream with the firmware

```sh
source /tools/Xilinx/2026.1/Vivado/settings64.sh
make TOP=top_soc fwbit
```

This first generates the DDR3 controller IP and the CPU block design. It
then runs synthesis, implementation and the bitstream. Finally it builds
`fw/build/llm.elf` and writes it into the CPU's local memory in the
bitstream (`build/top_soc/top_soc_fw.bit`). A full build takes a while;
later builds only rerun the stages whose inputs changed. See
[build.md](build.md).

## 6. Program the board and upload the model

Plug in both USB cables, put the microSD card in its slot and power on.
Then:

```sh
make TOP=top_soc program_fw                                       # over JTAG; lost at power-off
python3 host/image_tool.py upload models/stories15M.img --store   # ~2 min, once per model
```

The upload sends the image over the UART into DDR3 at 3 Mbaud. `--store`
then copies it to the microSD card. From then on, the board loads the model
from the card by itself after every configuration (~7 s).

## 7. Generate stories

```sh
python3 host/chat.py                          # interactive: one prompt per line
python3 host/chat.py --steps 100 "Tom and his dog"
```

The firmware prints a speed line after each story:

```
[168 tokens in 1.02 s: 164.9 tok/s; ms per token: matmul 3.1, attention 1.2, ...]
```

## 8. Check bit-exactness

```sh
make -C fw golden                                   # x86 build of the same firmware
python3 host/chat.py --trace "Once upon a time" > run.txt
fw/build/golden models/stories15M.img "Once upon a time" 256 /tmp > golden.txt
grep '^T ' run.txt | diff - golden.txt              # no output: identical
```

Each `T pos token hash` line holds a hash of all 32,000 logits of that
step. A single changed bit anywhere shows up as a difference.

## 9. Boot standalone from flash (optional)

```sh
make TOP=top_soc flash      # erases the QSPI flash and writes top_soc_fw.bit
```

Set the board's mode jumper to **QSPI**. At power-on the board then:
1. configures itself;
2. loads the model from the card;
3. prints `READY`.

`host/chat.py` works as before. Bring-up so far: rebooting from the flash
over JTAG worked; a boot from a real power cycle hasn't been tried yet.

## Other designs

| Design | What it is | How to use it |
|---|---|---|
| `top_llm` (default `TOP`) | No CPU. The PC runs the model, and the board does each matmul over the UART | `host/llm_host` (0.7 tokens/s); see [host_tools.md](host_tools.md#llm_host) |
| `top_qmv` | Just the matmul engine, with the weights sent over the UART | `host/qmv_uart.py` |
| `top_gemm` | An int8 systolic array GEMM | `host/gemm_uart.py` |

See [hardware/top_levels.md](hardware/top_levels.md).
