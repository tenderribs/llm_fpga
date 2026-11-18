# Firmware for top_soc's MicroBlaze V (RV32IMF). Build with `make` here;
# each program in its own directory becomes build/<name>.elf.
#   make            all programs
#   make bringup    one program
#   make golden     host (x86) builds of the model code: build/golden (the
#                   board's bit-exact golden) and build/golden_libm (runq.c's)
# model/ holds the model code (linked into llm), golden/ the host program.

CROSS   ?= /tools/Xilinx/2026.1/gnu/riscv/lin/bin/riscv64-unknown-elf-
CC      := $(CROSS)gcc
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

ARCH    := -march=rv32imf_zicsr -mabi=ilp32f
# -ffp-contract=off: no fused multiply-add, so float results match a CPU
# without FMA (and the reference runs) bit for bit; -fno-math-errno: sqrtf is
# the FPU instruction (correctly rounded, like newlib's slow software one)
CFLAGS  := $(ARCH) -O2 -g -Wall -Wextra -ffp-contract=off -fno-math-errno -ffunction-sections -fdata-sections \
           -Icommon
LDFLAGS  = $(ARCH) -nostartfiles -T common/link.ld -Wl,--build-id=none \
           -Wl,--gc-sections -Wl,-Map,$@.map

COMMON  := common/start.S common/syscalls.c
PROGS   := $(patsubst %/,%,$(filter-out common/ build/ model/ golden/,$(wildcard */)))
MODEL   := $(wildcard model/*.c)
MODEL_H := $(wildcard model/*.h)
HOSTCC  ?= gcc

.PHONY: all $(PROGS) golden clean
all: $(PROGS)
$(PROGS): %: build/%.elf

build/%.elf: %/*.c $(COMMON) common/link.ld common/llm_hw.h
	mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.c %.S,$^) -lm
	$(SIZE) $@

# programs using the model code
build/llm.elf: $(MODEL) $(MODEL_H)
build/llm.elf: CFLAGS += -Imodel -DMODEL_PROFILE

golden: build/golden build/golden_libm
build/golden build/golden_libm: golden/golden.c $(MODEL) $(MODEL_H)
	mkdir -p build
	$(HOSTCC) -O2 -Wall -Wextra -ffp-contract=off -fno-builtin -Imodel \
	    $(if $(findstring libm,$@),-DMODEL_LIBM) -o $@ golden/golden.c $(MODEL) -lm

clean:
	rm -rf build
