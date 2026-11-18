# Building and programming (Vivado)

```sh
source /tools/Xilinx/2026.1/Vivado/settings64.sh
```

The build is non-project mode Vivado, driven by the `Makefile` and
`scripts/build.tcl`. Output goes to `build/<TOP>/`.

## Make targets

| Target | Does |
|---|---|
| `make` / `make bit` | synthesis → implementation → bitstream for `TOP` (default `top_llm`) |
| `make synth` | stop after synthesis |
| `make impl` | stop after place and route |
| `make program` | load `build/<TOP>/<TOP>.bit` over JTAG (lost at power-off) |
| `make ip` | generate the DDR3 controller IP (automatic for `top_llm` and `top_soc`) |
| `make cpu` | generate the CPU block design (automatic for `top_soc`) |
| `make TOP=top_soc fwbit` | build `fw/build/llm.elf` and put it into the CPU's local memory: `build/top_soc/top_soc_fw.bit` |
| `make TOP=top_soc program_fw` | the same, then load it over JTAG |
| `make TOP=top_soc flash` | the same, then write it to the QSPI flash (erases the flash) |

`TOP` is one of `top_soc`, `top_llm`, `top_qmv` or `top_gemm`.

**Only changed stages rerun:**
- Synthesis depends on every `src/sources/*.sv`, `src/constrs/*.xdc`,
  `scripts/build.tcl`, and the generated IP and block design.
- Each stage writes its checkpoint last. A stage that fails partway
  therefore runs again next time.
- `fwbit` always rebuilds the firmware (`make -C fw llm`), but reruns
  `updatemem` only if the ELF changed.

## Stages (`scripts/build.tcl`)

| Stage | Runs | Writes |
|---|---|---|
| `synth` | reads the RTL, `genesys2.xdc`, and either the MIG IP + `top_llm.xdc` (+ the CPU block design) or `sysclk.xdc`; `synth_design` | `post_synth.dcp`, utilization and timing reports |
| `impl` | `opt_design`; `place_design -directive ExtraTimingOpt`; `phys_opt_design -directive AggressiveExplore`; `route_design -directive AggressiveExplore`; one more `phys_opt_design` if setup slack is negative | `post_route.dcp`, utilization, timing, DRC reports. Prints `Timing met, WNS = …` or a warning; a timing failure doesn't stop the build, so check the message |
| `bit` | `write_bitstream`; for `top_soc` also `write_mem_info` (`top_soc.mmi`, where the local memory sits, for `updatemem`) | `<TOP>.bit` |
| `program` | finds the FPGA on the JTAG targets, programs it | |
| `flash` | `write_cfgmem` (SPI ×4, 32 MB) → `.mcs`; programs the S25FL256S (`FLASH_PART` overrides the part name); erases, programs, verifies; then boots the FPGA from the flash | |

To run a stage by hand:

```sh
vivado -mode batch -nojournal -log build/top_soc/impl.log -source scripts/build.tcl -tclargs impl top_soc
```

## Generated IP and block design

- **`scripts/gen_ip.tcl`** generates `mig_ddr3` into `build/ip/` from
  `src/ip/mig_ddr3.prj`. That file is Digilent's Genesys 2 preset with the
  module name changed: 1800 MT/s, 256-bit AXI, 225 MHz user clock. The IP is
  synthesized out of context once.
- **`scripts/gen_cpu.tcl`** generates the CPU block design into
  `build/cpu_bd/`. See [hardware/cpu_subsystem.md](hardware/cpu_subsystem.md).
  Editing this script rebuilds everything downstream.

## Timing

| Design | Clock(s) | Result |
|---|---|---|
| `top_soc` | 225 MHz (`llm_core`, DDR3 side), 150 MHz (CPU) | met, WNS +0.008 ns: very little margin. The first vector unit build missed by 0.8 ns, before its RAM paths were pipelined and these directives were added |
| `top_llm` | 225 MHz | met |
| `top_qmv`, `top_gemm` | 200 MHz | met |

The timing-critical paths are in the 225 MHz domain. Past trouble spots
were the vector unit's RAM accesses and the weight reader's burst logic;
both are now pipelined (`RdLat` = 3, the AR skid buffer). If a change breaks timing,
look at `build/<TOP>/post_route_timing.rpt` first.

**Utilization** (`top_soc`): about 26 % of LUTs, 12 % of FFs, 29 % of
block RAM and 7 % of DSPs. See
[system_design.md](system_design.md#12-resources) for the per-block
breakdown. Use `report_utilization -hierarchical` on `post_route.dcp` for
the details.

## Firmware in the bitstream

```sh
make TOP=top_soc fwbit
```

This runs:

```sh
updatemem -force -meminfo build/top_soc/top_soc.mmi -bit build/top_soc/top_soc.bit \
    -data fw/build/llm.elf -proc u_cpu/mbv -out build/top_soc/top_soc_fw.bit
```

`updatemem` writes the ELF into the local memory's block RAM contents in
the bitstream. It takes seconds, with no resynthesis. The CPU runs it from
configuration on, with `_start` at address 0.

## Programming

Plug in the **PROG** micro-USB port.
- `make program` and `make program_fw` load the bitstream into the FPGA
  (volatile).
- `make flash` writes the QSPI flash. With the mode jumper on QSPI, the
  board then configures from the flash at power-on: ×4 SPI at 33 MHz,
  compressed bitstream. The flash holds only the bitstream; the model comes
  from the microSD card.

Other firmware can be loaded onto a running `top_soc` without rebuilding the
bitstream:

```sh
xsdb scripts/run_fw.tcl fw/build/bringup.elf
```

The Genesys 2's JTAG shows up as **two** targets, and only one has the FPGA
on its chain. `build.tcl` and `run_fw.tcl` try each. See
[troubleshooting.md](troubleshooting.md#jtag).
