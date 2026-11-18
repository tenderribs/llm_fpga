# Vivado build. Each target runs only the stages whose inputs have changed.
#   make        build through to the bitstream
#   make impl   stop after place and route
#   make synth  stop after synthesis
#   make program  load the bitstream onto the board over JTAG
#   make TOP=top_soc program_fw   ... with the llama2 firmware built in
#   make TOP=top_soc flash        write that to the QSPI flash (boots from it)
# TOP picks the design (default top_llm; also top_soc, top_qmv, top_gemm).
# Output goes to build/<TOP>/.
#   make ip     generate the DDR controller IP (top_llm / top_soc; automatic)
#   make cpu    generate the CPU block design (top_soc; automatic)
# Source /tools/Xilinx/2026.1/Vivado/settings64.sh first so vivado is on PATH.

VIVADO ?= vivado
TOP    ?= top_llm
OUT    := build/$(TOP)

SRCS := $(wildcard src/sources/*.sv)
XDC  := $(wildcard src/constrs/*.xdc)

run = $(VIVADO) -mode batch -nojournal -log $(OUT)/$(1).log -source scripts/build.tcl -tclargs $(1) $(TOP)

.PHONY: bit impl synth program ip cpu fwbit program_fw flash FORCE

MIG_XCI := build/ip/mig_ddr3/mig_ddr3.xci
CPU_BD  := build/cpu_bd/cpu_bd.srcs/sources_1/bd/cpu/cpu.bd
IP_DEPS := $(if $(filter top_llm top_soc,$(TOP)),$(MIG_XCI)) \
           $(if $(filter top_soc,$(TOP)),$(CPU_BD))

bit:   $(OUT)/$(TOP).bit
impl:  $(OUT)/post_route.dcp
synth: $(OUT)/post_synth.dcp
ip:    $(MIG_XCI)
cpu:   $(CPU_BD)

$(MIG_XCI): src/ip/mig_ddr3.prj scripts/gen_ip.tcl
	rm -rf build/ip/mig_ddr3
	mkdir -p build/ip
	$(VIVADO) -mode batch -nojournal -log build/ip/gen_ip.log -source scripts/gen_ip.tcl

$(CPU_BD): scripts/gen_cpu.tcl
	mkdir -p build
	$(VIVADO) -mode batch -nojournal -log build/gen_cpu.log -source scripts/gen_cpu.tcl

$(OUT)/post_synth.dcp: $(SRCS) $(XDC) scripts/build.tcl $(IP_DEPS)
	mkdir -p $(OUT)
	$(call run,synth)

$(OUT)/post_route.dcp: $(OUT)/post_synth.dcp
	$(call run,impl)

$(OUT)/$(TOP).bit: $(OUT)/post_route.dcp
	$(call run,bit)

program: $(OUT)/$(TOP).bit
	$(call run,program)

# top_soc with the llama2 firmware (fw/build/llm.elf) in the CPU's local
# memory, so it runs from configuration on: no JTAG loading
FW_ELF := fw/build/llm.elf
FW_BIT := $(OUT)/$(TOP)_fw.bit
$(FW_BIT): $(OUT)/$(TOP).bit $(FW_ELF)
	updatemem -force -meminfo $(OUT)/$(TOP).mmi -bit $(OUT)/$(TOP).bit \
	    -data $(FW_ELF) -proc u_cpu/mbv -out $@
$(FW_ELF): FORCE
	$(MAKE) -C fw llm
FORCE:

fwbit: $(FW_BIT)
program_fw: $(FW_BIT)
	$(VIVADO) -mode batch -nojournal -log $(OUT)/program.log -source scripts/build.tcl \
	    -tclargs program $(TOP) $(FW_BIT)
# writes the board's QSPI flash (erases what was there)
flash: $(FW_BIT)
	$(VIVADO) -mode batch -nojournal -log $(OUT)/flash.log -source scripts/build.tcl \
	    -tclargs flash $(TOP) $(FW_BIT)
