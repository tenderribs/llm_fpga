# Reset entry for the MicroBlaze V firmware: stack, BSS, FPU, then main().
    .section .text.start
    .global _start
_start:
    .option push
    .option norelax
    la      gp, __global_pointer$
    .option pop
    la      sp, __stack_top

    # The FPU traps until mstatus.FS is non-zero; set it to Dirty (3).
    li      t0, (3 << 13)
    csrs    mstatus, t0
    fscsr   zero                    # round to nearest even, no flags

    # clear .bss
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sw      zero, 0(t0)
    addi    t0, t0, 4
    j       1b
2:
    call    __libc_init_array
    li      a0, 0
    li      a1, 0
    call    main
3:  j       3b                      # main returned: stop here
