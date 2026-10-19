# A masked vector op (vm=0) is outside the supported subset: core must halt with
# ILLEGAL (cause 2). Run with: vvp sim +hex=<file> +expect=2
        .section .text.init
        .globl _start
_start:
        li      t0, 4
        vsetvli a0, t0, e32, m1, ta, ma
        .word   0x00208157            # vadd.vv v2,v2,v1, vm=0
        li      a7, 93
        li      a0, 1
        ecall                         # must not be reached
