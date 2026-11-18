# Troubleshooting and known issues

## Board

### JTAG

**"No devices detected", or no CPU target in `xsdb`.**
- The Genesys 2's JTAG chip shows up as two hardware-server targets (the
  serial number, and the same with `B` appended), in varying order. Only one
  has the FPGA on its chain.
- Opening the wrong one gives "No devices detected". That is not a power
  problem, and no power cycle is needed.
- `scripts/build.tcl` (`open_fpga`) and `scripts/run_fw.tcl` try each
  target.
- `run_fw.tcl` also retries for a few seconds, because the target list
  fills in after `connect`.

### UART

**Nothing happens on the UART.**
- Check that the **UART** micro-USB port is plugged in (not only PROG).
- Check that your user is in the `dialout` group.
- The port is normally `/dev/ttyUSB0` (FT232R); pass `--port` / `-p`
  otherwise.

**A host script hangs, or the board seems out of step.** A command
interrupted partway is dropped after 100 ms of silence. `llm_link.py` waits
0.2 s after opening the port for this reason. Rerunning the script is
enough.

**Garbled text in the middle of a reply.** This shouldn't happen: firmware
console output is only sent between UART commands. If it does, two programs
probably have the port open.

### `chat.py`

**`[no answer from the board: is top_soc running fw/llm?]`**
- The design may not be `top_soc` with the firmware (`make TOP=top_soc
  program_fw`).
- The model may still be loading: about 7 s after configuration.
- The firmware may have stopped with `LLM FAIL: <reason>`. Run
  `python3 host/console.py` and press CPU RESET to see the startup output.

**Stories stop early.** `--steps N` (or a `/steps N` line) persists on the
board until reset. Pass `--steps 256` to lift it.

### Model loading

**`LLM FAIL: no model image (bad magic)`** means no valid model image is
in DDR3:
- Check `python3 host/sd_tool.py status`. Error 8 is a blank card or one
  without an image.
- Upload the model with `python3 host/image_tool.py upload
  models/stories15M.img --store`.

**SD errors in STATUS** (LEDs 7:4 show the code): see the table in
[hardware/sd_ctrl.md](hardware/sd_ctrl.md#status-and-errors).
- Codes 1-5 are card init problems: no card, or an unusual card. Reseat it
  and press CPU RESET, which retries the load.
- Code 9 is a checksum mismatch: store the image again.

**`LLM FAIL: group size must be 32 (the engine's)`.** The model was exported
with group size 64 (llama2.c does this when dim is a multiple of 64, as for
stories42M and stories110M). Re-export with
`python3 ref/export_q80.py --group-size 32 src.bin dst_q80.bin` and rebuild
the image.

**Other `model_open` errors** (`unsupported model image version`,
`bad matrix count`) mean the image doesn't match the firmware. Rebuild it
with `host/build_image.py`.

### Boot from flash

- The mode jumper must be on **QSPI**.
- `make TOP=top_soc flash` writes `top_soc_fw.bit`, the bitstream *with*
  the firmware. A plain bitstream would boot a CPU with empty memory.
- If the flash part isn't found, set `FLASH_PART` to the part name Vivado
  lists for your board revision.
- Booting from a real power cycle hasn't been tested yet. Rebooting from
  the flash over JTAG works.

## Build

### Unexpected full rebuilds

Synthesis depends on every source file's modification time. Checking out
another branch, or fast-forwarding an old one, touches files and triggers a
full rebuild. So does editing `scripts/gen_cpu.tcl`, which regenerates the
block design. Avoid switching branches while a build runs.

To program a saved bitstream without make:

```sh
vivado -mode batch -source scripts/build.tcl -tclargs program top_soc path/to.bit
```

### Timing

`top_soc` closes with +0.008 ns of slack. A small RTL change can tip it
negative. The build prints `WARNING: timing not met` but still writes a
bitstream, which may misbehave on the board. Check
`build/top_soc/post_route_timing.rpt`.

### Firmware

**`llm.elf` doesn't fit in 256 KB.** Code, data, the 48 KB arena and the
16 KB stack all share local memory. Move large buffers to the heap, which
is in DDR3 (`malloc`).

**The CPU doesn't start from the bitstream.** `_start` must be at address
0. The toolchain puts a `.note.gnu.build-id` there unless it is discarded:
`link.ld` and `--build-id=none` take care of that.

**Results differ from the golden after a firmware change.** Look for:
- reordered float operations. Only *independent* operations may be
  reordered or overlapped;
- an FMA: keep `-ffp-contract=off`;
- a C library math call instead of `mathf.c`.

Rebuild `fw/build/golden` and compare the `T` lines to find the first step
that differs.

## Simulation

See the Verilator notes in [simulation.md](simulation.md#verilator-notes).
Also:
- `run_qmv.sh` needs the dumps from `ref/build_ref.sh`.
- Generated vectors go to `sim/build/`. Delete it to force regeneration.
