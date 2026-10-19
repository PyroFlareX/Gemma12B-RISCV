#!/usr/bin/env bash
# Build RVV subset tests and run them on rtl/rv32i_core.sv under Icarus.
# Needs: riscv64-unknown-elf-gcc (>=13, -march=rv32iv) and iverilog/vvp on PATH (or set IVERILOG/VVP/IVL_B).
set -e
cd "$(dirname "$0")"; R=../../rtl; O=${OUT:-/tmp/rvv-out}; mkdir -p $O
IVERILOG=${IVERILOG:-iverilog}; VVP=${VVP:-vvp}
$IVERILOG ${IVL_B:+-B $IVL_B} -g2012 -s riscv_tests_tb -o $O/sim $R/rv32i_core.sv $R/riscv_tests_tb.sv 2>&1 | grep -E "error" || true
build() { riscv64-unknown-elf-gcc -march=rv32iv -mabi=ilp32 -mno-relax -static -nostdlib -nostartfiles \
  -Wl,--no-relax -Wl,-Ttext=0 -Wl,-Tdata=0x4000 -o $O/$1.elf $1.S
  riscv64-unknown-elf-objcopy -O binary $O/$1.elf $O/$1.bin
  python3 -c "import struct;d=open('$O/$1.bin','rb').read();d+=b'\0'*(-len(d)%4);open('$O/$1.hex','w').write(''.join('%08x\n'%w for w in struct.unpack('<%dI'%(len(d)//4),d)))"; }
build rvv_test; build illegal_masked
echo -n "rvv_test: ";        $VVP $O/sim +hex=$O/rvv_test.hex | grep -E "PASS|FAIL"
echo -n "illegal_masked: ";  $VVP $O/sim +hex=$O/illegal_masked.hex +expect=2 | grep -E "PASS|FAIL"
