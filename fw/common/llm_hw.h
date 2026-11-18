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
#define LLM_CONSOLE_IN  REG(0x0000C)   // bit 8: a byte from the host in 7:0 (read takes it)
#define LLM_MM_ADDR     REG(0x00010)
#define LLM_MM_ND       REG(0x00014)
#define LLM_MM_START    REG(0x00018)
#define LLM_SD_BYTES    REG(0x0001C)
#define LLM_SD_ADDR     REG(0x00020)
#define LLM_CYCLES      REG(0x00024)
#define LLM_AT_ADDR     REG(0x00030)   // attention: K / V address (DDR offset)
#define LLM_AT_CFG      REG(0x00034)   // head_size | timesteps << 16
#define LLM_AT_START    REG(0x00038)   // 0: scores, 1: weighted sum
#define LLM_ARGMAX      REG(0x0003C)   // largest result of the last CPU matmul
#define LLM_MM_YOFF     REG(0x00044)   // where a CPU matmul's results start in Y
#define LLM_VPU_OP      REG(0x00050)   // start a vector unit job (VPU_*)
#define LLM_VPU_N       REG(0x00054)
#define LLM_VPU_SRC     REG(0x00058)
#define LLM_VPU_SRC2    REG(0x0005C)
#define LLM_VPU_DST     REG(0x00060)
#define LLM_VPU_F       REG(0x00064)
#define LLM_DMA_SRC     REG(0x00068)   // DMA V -> DDR: first V word
#define LLM_DMA_N       REG(0x0006C)   // words
#define LLM_DMA_DST     REG(0x00070)   // DDR byte address
#define LLM_DMA_STRIDE  REG(0x00074)   // 4: whole beats, else one word per write
#define LLM_DMA_START   REG(0x00078)
#define VPU_COPY        0u            // V -> the attention unit's A RAM
#define VPU_QUANT       1u
#define VPU_SILU        2u
#define VPU_SOFTMAX     3u
#define VPU_SRC_V       (1u << 8)     // QUANT reads the V RAM
#define LLM_XQ          ((volatile uint32_t *)(LLM_BASE + 0x01000))
#define LLM_XS          ((volatile uint32_t *)(LLM_BASE + 0x01800))
#define LLM_AT_A        ((volatile uint32_t *)(LLM_BASE + 0x04000))
#define LLM_Y           ((volatile uint32_t *)(LLM_BASE + 0x20000))
// the same Y RAM on the local memory bus: a few cycles per read instead of
// a round trip through the interconnect
#define LLM_YL          ((const volatile uint32_t *)0x00100000u)
// the vector unit's V RAM (8192 words), read / write on the local memory bus
#define LLM_VL          ((volatile uint32_t *)0x00120000u)

#define ST_CALIB        (1u << 0)
#define ST_SD_BUSY      (1u << 1)
#define ST_SD_OK        (1u << 2)
#define ST_SD_ERR       (1u << 3)
#define ST_SD_CODE(s)   (((s) >> 4) & 0xF)
#define ST_MM_BUSY      (1u << 8)   // matmul or attention job
#define ST_CONS_FULL    (1u << 9)
#define ST_CONS_IN      (1u << 10)

#define CLK_HZ          225000000u  // llm_core and its CYCLES counter (DDR user clock)
#define CPU_HZ          150000000u  // the CPU (rdcycle); scripts/gen_cpu.tcl
