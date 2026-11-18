/* MicroBlaze V firmware: code, data and stack in the 256 KB local memory.
   The heap (malloc) is in cached DDR, see syscalls.c. */
OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    LMB (rwx) : ORIGIN = 0x00000000, LENGTH = 256K
}

STACK_SIZE = 16K;

SECTIONS
{
    .text : {
        KEEP(*(.text.start))
        *(.text .text.*)
        *(.rodata .rodata.* .srodata .srodata.*)
        . = ALIGN(4);
        __init_array_start = .;
        KEEP(*(.init_array .init_array.*))
        __init_array_end = .;
        __preinit_array_start = .;
        __preinit_array_end = .;
        __fini_array_start = .;
        __fini_array_end = .;
    } > LMB

    .data : {
        . = ALIGN(8);
        *(.data .data.*)
        __global_pointer$ = . + 0x800;
        *(.sdata .sdata.*)
    } > LMB

    .bss (NOLOAD) : {
        . = ALIGN(8);
        __bss_start = .;
        *(.sbss .sbss.* .bss .bss.* COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > LMB

    .stack (NOLOAD) : {
        . = ALIGN(16);
        . += STACK_SIZE;
        __stack_top = .;
    } > LMB

    /DISCARD/ : { *(.eh_frame .eh_frame_hdr) }
}
