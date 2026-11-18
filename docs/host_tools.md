# Host tools (`host/`)

Everything here runs on the PC. All serial tools take `--port` (default
`/dev/ttyUSB0`) and talk 8N1 at 3 Mbaud to `llm_core`, except the
`top_qmv` and `top_gemm` scripts.

## Using `top_soc`

### `chat.py`

Talks to the firmware: sends a prompt and prints the story as the board
generates it.

```sh
python3 host/chat.py                                # interactive, one prompt per line
python3 host/chat.py "Once upon a time"             # one prompt, then exit
python3 host/chat.py --steps 64 "A cat"             # at most 64 tokens (persists on the board)
python3 host/chat.py --trace "Once upon a time" > run.txt   # full length + T lines
```

How it works:
- It sends each line with UART command `0x06` (CONSOLE_IN), in chunks of
  up to 255 bytes.
- It prints the console output until a line `LLM DONE`, then waits for
  `READY`.
- `--timeout` (default 30 s) is how long it waits without output.
- `--steps N` sends `/steps N` first. The setting persists on the board
  until reset, so pass `--steps` again (e.g. `--steps 256`) to change it.

### `build_image.py`

```sh
python3 host/build_image.py models/stories15M_q80.bin third_party/llama2.c/tokenizer.bin models/stories15M.img
```

Builds the model image from a `runq.c` Q8_0 checkpoint and `tokenizer.bin`.
The format is in [data_formats.md](data_formats.md#model-image).

### `image_tool.py`

```sh
python3 host/image_tool.py upload models/stories15M.img [--store] [--verify]
python3 host/image_tool.py check models/stories15M.img
```

- `upload` writes the image to DDR3 offset 0 over the UART.
- `--store` then has the FPGA copy it to the microSD card (SD_STORE), so it
  loads at every boot.
- `--verify` reads all of it back.
- `check` compares DDR3 with the file: the header plus samples.

It works with `top_llm` and `top_soc`.

### `console.py`

```sh
python3 host/console.py [--until 'PASS|FAIL'] [--timeout 30]
```

Prints the firmware's console. It returns 0 if a line matching `--until`
was seen and doesn't contain `FAIL`. Use it with the JTAG-loaded programs
(bringup, membench, fpubench).

## Using `top_llm` (no CPU)

### `llm_host`

`host/llm_host.c`, built by `host/build_host.sh` into `ref/build/llm_host`.

This is `runq.c` with `matmul()` replaced:
- the weights live in the board's DDR3 in qmv DDR format;
- each matmul sends only x over the UART and reads back the results.

Everything else is `runq.c`'s own code, so the logits must be bit-identical
to `ref/runq_dump`.

```sh
host/build_host.sh          # needs ref/build/runq_patched.c from ref/build_ref.sh
ref/build/llm_host models/stories15M_q80.bin third_party/llama2.c/tokenizer.bin \
    "Once upon a time" 64 ref/build/run [options]
cmp ref/build/run/logits.bin ref/build/stories15M/logits.bin
```

| Option | Meaning |
|---|---|
| `-p port` | serial port |
| `-u` | always upload the weights |
| `-n` | never upload |
| `-s` | after uploading, store the weights on the card |
| `-v` | also run each matmul on the CPU and compare |

By default it uploads unless DDR3 already holds this model, either as a
model image or as bare weights loaded from the card. It runs at 0.7
tokens/s, limited by USB round trips.

### `llm_link.py`

The Python library for the UART command set:
- `Link(port, baud)` with `status()`, `write(addr, data)`,
  `read(addr, n)`, `matmul(addr, x, n, d)`, `sd_load()`,
  `sd_store(addr, n)` and `sd_wait()`;
- `ddr_format(w, n, d, gs)`, which lays out a matrix in qmv DDR format.

Opening the port waits 0.2 s, so a half-sent command from an earlier run
times out on the board first.

### `ddr_test.py`

```sh
python3 host/ddr_test.py [dump index ...]     # default 000 004 006
```

`top_llm` bring-up:
1. Checks STATUS for DDR3 calibration.
2. Writes random data at addresses spread over the 1 GB and reads it back.
3. Runs real `runq.c` matmuls from DDR3, every result checked bit for bit.

### `sd_tool.py`

```sh
python3 host/sd_tool.py status | load | store ADDR NBYTES
```

| Command | Does |
|---|---|
| `status` | the STATUS reply, decoded |
| `load` | SD_LOAD |
| `store ADDR NBYTES` | SD_STORE (wipes the card) |

## The standalone engines

| Tool | For | What it does |
|---|---|---|
| `qmv_uart.py [dump index ...]` | `top_qmv`, 1 Mbaud | Sends real `runq.c` matmul dumps (`ref/build/stories15M/mm_NNN.bin`; default 000, 004, 006 = layer 0 wq, w1, w2) and checks every output bit for bit |
| `gemm_uart.py --n 16 --iters 10` | `top_gemm`, 115200 baud | Random `Y = A @ B + C`, checked against numpy |

## Testing without the board

`sim/fake_board.py` implements the `top_llm` command set on a
pseudo-terminal. See [simulation.md](simulation.md#fake-board).
