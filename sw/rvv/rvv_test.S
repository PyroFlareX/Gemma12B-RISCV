# Self-checking RVV-subset test for rtl/rv32i_core.sv.
# Convention (matches rtl/riscv_tests_tb.sv): pass = ecall with a7=93,a0=0;
# failure = ecall with a0 = test number (gp). Data at 0x4000.
# Build/run: see sw/rvv/run.sh
        .section .text.init
        .globl _start
_start:
        li      gp, 0
        li      s0, 0x4000            # data base

# CHECK macro: compare a0 with expected, else fail with test number in gp
        .macro CHECK reg, val
        addi    gp, gp, 1
        li      t6, \val
        bne     \reg, t6, fail
        .endm

# --- vsetvli: vl = min(AVL, VLMAX), VLMAX = 16/8/4 for e8/e16/e32 (VLEN=128, LMUL=1)
        li      t0, 100
        vsetvli a0, t0, e8, m1, ta, ma
        CHECK   a0, 16
        vsetvli a0, t0, e16, m1, ta, ma
        CHECK   a0, 8
        vsetvli a0, t0, e32, m1, ta, ma
        CHECK   a0, 4
        li      t0, 3
        vsetvli a0, t0, e32, m1, ta, ma
        CHECK   a0, 3
        vsetivli a0, 5, e16, m1, ta, ma
        CHECK   a0, 5
        csrr    a1, 0xC20             # vl
        CHECK   a1, 5
        csrr    a1, 0xC22             # vlenb
        CHECK   a1, 16
        li      t1, 7
        li      t0, 12
        vsetvl  a0, t0, t1             # vtype=7 -> e? sew field=0 lmul=7 -> vill? (lmul=7 is mf2: unsupported)
        CHECK   a0, 0                 # vill -> vl=0

# --- vle32 / vse32 round trip, vadd.vv, vsub.vv
        li      t0, 4
        vsetvli a0, t0, e32, m1, ta, ma
        vle32.v v1, (s0)              # data: 1,2,3,0xfffffffe
        addi    t1, s0, 32
        vle32.v v2, (t1)              # data: 10,20,30,40
        vadd.vv v3, v1, v2
        addi    t2, s0, 64
        vse32.v v3, (t2)
        lw      a0, 0(t2)
        CHECK   a0, 11
        lw      a0, 4(t2)
        CHECK   a0, 22
        lw      a0, 8(t2)
        CHECK   a0, 33
        lw      a0, 12(t2)
        CHECK   a0, 38
        vsub.vv v4, v2, v1
        vse32.v v4, (t2)
        lw      a0, 12(t2)
        CHECK   a0, 42
        lw      a0, 0(t2)
        CHECK   a0, 9

# --- .vx / .vi forms, logic and shifts
        vadd.vx v5, v1, s0            # + 0x4000
        vmv.x.s a0, v5
        CHECK   a0, 0x4001
        vrsub.vi v6, v1, 5            # 5 - v1
        vmv.x.s a0, v6
        CHECK   a0, 4
        vand.vx v7, v2, t0            # t0=4: 10&4=0, 20&4=4
        vse32.v v7, (t2)
        lw      a0, 4(t2)
        CHECK   a0, 4
        vsll.vi v8, v2, 2
        vse32.v v8, (t2)
        lw      a0, 8(t2)
        CHECK   a0, 120
        vsra.vi v9, v1, 1             # elem3 = 0xfffffffe >>> 1 = -1
        vse32.v v9, (t2)
        lw      a0, 12(t2)
        CHECK   a0, -1
        vsrl.vi v9, v1, 1
        vse32.v v9, (t2)
        lw      a0, 12(t2)
        CHECK   a0, 0x7fffffff
        vxor.vv v9, v1, v1
        vor.vi  v9, v9, 7
        vse32.v v9, (t2)
        lw      a0, 8(t2)
        CHECK   a0, 7

# --- min/max signed vs unsigned (elem3 = -2 vs 3)
        vmin.vv v10, v1, v2           # signed: min(-2,40) = -2
        vse32.v v10, (t2)
        lw      a0, 12(t2)
        CHECK   a0, -2
        vminu.vv v10, v1, v2          # unsigned: min(0xfffffffe,40) = 40
        vse32.v v10, (t2)
        lw      a0, 12(t2)
        CHECK   a0, 40
        vmax.vv v10, v1, v2           # signed: max(-2,40)=40
        vse32.v v10, (t2)
        lw      a0, 12(t2)
        CHECK   a0, 40
        vmaxu.vv v10, v1, v2
        vse32.v v10, (t2)
        lw      a0, 12(t2)
        CHECK   a0, -2

# --- vmul, vmacc, vredsum
        vmul.vv v11, v1, v2
        vse32.v v11, (t2)
        lw      a0, 8(t2)
        CHECK   a0, 90
        vmacc.vv v11, v1, v2          # v11 += v1*v2 -> 180 at elem 2
        vse32.v v11, (t2)
        lw      a0, 8(t2)
        CHECK   a0, 180
        vmv.v.i v12, 0
        vredsum.vs v12, v2, v12       # 10+20+30+40
        vmv.x.s a0, v12
        CHECK   a0, 100
        li      t3, 55
        vmv.s.x v13, t3
        vmv.x.s a0, v13
        CHECK   a0, 55
        vmv.v.x v14, t3
        vmv.v.v v15, v14
        vse32.v v15, (t2)
        lw      a0, 12(t2)
        CHECK   a0, 55

# --- SEW=8: wraparound, signed ops, vl < VLMAX tail undisturbed
        li      t0, 16
        vsetvli a0, t0, e8, m1, ta, ma
        vmv.v.i v16, 0
        li      t4, 127
        vadd.vx v16, v16, t4          # 127 each
        vadd.vi v16, v16, 1           # wraps to 0x80
        vmv.x.s a0, v16
        CHECK   a0, -128              # vmv.x.s sign-extends
        li      t0, 4
        vsetvli a0, t0, e8, m1, ta, ma
        vmv.v.i v16, 3                # only first 4 elements change
        li      t0, 16
        vsetvli a0, t0, e8, m1, ta, ma
        vse8.v  v16, (t2)
        lbu     a0, 3(t2)
        CHECK   a0, 3
        lbu     a0, 4(t2)
        CHECK   a0, 0x80              # tail undisturbed
        vsra.vi v17, v16, 1           # 0x80>>>1 = 0xc0 in elem 4
        vse8.v  v17, (t2)
        lbu     a0, 4(t2)
        CHECK   a0, 0xc0
        vmul.vv v18, v16, v16         # 0x80*0x80 low 8 bits = 0
        vse8.v  v18, (t2)
        lbu     a0, 4(t2)
        CHECK   a0, 0

# --- SEW=16: load/store, sum
        li      t0, 8
        vsetvli a0, t0, e16, m1, ta, ma
        addi    t1, s0, 96
        vle16.v v19, (t1)             # 1..8
        vmv.v.i v20, 0
        vredsum.vs v20, v19, v20
        vmv.x.s a0, v20
        CHECK   a0, 36
        vse16.v v19, (t2)
        lhu     a0, 14(t2)
        CHECK   a0, 8

        j       pass

pass:
        li      a7, 93
        li      a0, 0
        ecall
fail:
        li      a7, 93
        mv      a0, gp
        ecall

        .section .data
        .org    0
        .word   1, 2, 3, 0xfffffffe          # 0x4000
        .word   0, 0, 0, 0
        .word   10, 20, 30, 40               # 0x4020
        .word   0, 0, 0, 0
        .word   0, 0, 0, 0                   # 0x4040 scratch
        .word   0, 0, 0, 0
        .half   1, 2, 3, 4, 5, 6, 7, 8       # 0x4060
