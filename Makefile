# Vivado build. Each target runs only the stages whose inputs have changed.
#   make        build through to the bitstream
#   make impl   stop after place and route
#   make synth  stop after synthesis
#   make program  load the bitstream onto the board over JTAG
# TOP picks the design (default top_qmv; also top_gemm). Output goes to
# build/<TOP>/.
# Source /tools/Xilinx/2026.1/Vivado/settings64.sh first so vivado is on PATH.

VIVADO ?= vivado
TOP    ?= top_qmv
OUT    := build/$(TOP)

SRCS := $(wildcard src/sources/*.sv)
XDC  := $(wildcard src/constrs/*.xdc)

run = $(VIVADO) -mode batch -nojournal -log $(OUT)/$(1).log -source scripts/build.tcl -tclargs $(1) $(TOP)

.PHONY: bit impl synth program
bit:   $(OUT)/$(TOP).bit
impl:  $(OUT)/post_route.dcp
synth: $(OUT)/post_synth.dcp

$(OUT)/post_synth.dcp: $(SRCS) $(XDC) scripts/build.tcl
	mkdir -p $(OUT)
	$(call run,synth)

$(OUT)/post_route.dcp: $(OUT)/post_synth.dcp
	$(call run,impl)

$(OUT)/$(TOP).bit: $(OUT)/post_route.dcp
	$(call run,bit)

program: $(OUT)/$(TOP).bit
	$(call run,program)
