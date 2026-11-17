# Firmware for top_soc's MicroBlaze V (RV32IMF). Build with `make` here;
# each program in its own directory becomes build/<name>.elf.
#   make            all programs
#   make bringup    one program

CROSS   ?= /tools/Xilinx/2026.1/gnu/riscv/lin/bin/riscv64-unknown-elf-
CC      := $(CROSS)gcc
OBJDUMP := $(CROSS)objdump
SIZE    := $(CROSS)size

ARCH    := -march=rv32imf_zicsr -mabi=ilp32f
# -ffp-contract=off: no fused multiply-add, so float results match a CPU
# without FMA (and the reference runs) bit for bit
CFLAGS  := $(ARCH) -O2 -g -Wall -Wextra -ffp-contract=off -ffunction-sections -fdata-sections \
           -Icommon
LDFLAGS := $(ARCH) -nostartfiles -T common/link.ld  \
           -Wl,--gc-sections -Wl,-Map,$@.map

COMMON  := common/start.S common/syscalls.c
PROGS   := $(patsubst %/,%,$(filter-out common/ build/,$(wildcard */)))

.PHONY: all $(PROGS) clean
all: $(PROGS)
$(PROGS): %: build/%.elf

build/%.elf: %/*.c $(COMMON) common/link.ld common/llm_hw.h
	mkdir -p build
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.c %.S,$^) -lm
	$(SIZE) $@

clean:
	rm -rf build
