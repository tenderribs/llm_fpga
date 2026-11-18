// Address map and llm_core registers (see src/sources/llm_core.sv and
// scripts/gen_cpu.tcl).
#pragma once
#include <stdint.h>

#define DDR_BASE        0x80000000u  // DDR3, cached: 0x8000_0000..0x9FFF_FFFF
#define DDR_UNCACHED    0xA0000000u  // DDR3 offset 512 MB, uncached
#define HEAP_BASE       0x90000000u  // malloc arena (cached), 256 MB
#define HEAP_SIZE       0x10000000u

#define LLM_BASE        0x40000000u
#define REG(off)        (*(volatile uint32_t *)(LLM_BASE + (off)))
#define LLM_ID          REG(0x00000)
#define LLM_STATUS      REG(0x00004)
#define LLM_CONSOLE     REG(0x00008)
#define LLM_MM_ADDR     REG(0x00010)
#define LLM_MM_ND       REG(0x00014)
#define LLM_MM_START    REG(0x00018)
#define LLM_SD_BYTES    REG(0x0001C)
#define LLM_SD_ADDR     REG(0x00020)
#define LLM_CYCLES      REG(0x00024)
#define LLM_XQ          ((volatile uint32_t *)(LLM_BASE + 0x01000))
#define LLM_XS          ((volatile uint32_t *)(LLM_BASE + 0x01800))
#define LLM_Y           ((volatile uint32_t *)(LLM_BASE + 0x20000))

#define ST_CALIB        (1u << 0)
#define ST_SD_BUSY      (1u << 1)
#define ST_SD_OK        (1u << 2)
#define ST_SD_ERR       (1u << 3)
#define ST_SD_CODE(s)   (((s) >> 4) & 0xF)
#define ST_MM_BUSY      (1u << 8)
#define ST_CONS_FULL    (1u << 9)

#define CLK_HZ          225000000u  // llm_core and its CYCLES counter (DDR user clock)
#define CPU_HZ          150000000u  // the CPU (rdcycle); scripts/gen_cpu.tcl
