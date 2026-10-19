#!/usr/bin/env bash
# Full RTL regression for rtl/rv32i_core.sv. Prints a PASS/FAIL line per suite and a summary.
# Tools (override via env): IVERILOG, VVP, IVL_B (iverilog -B dir), RVGCC_DIR (dir holding riscv64-unknown-elf-*),
# RISCV_TESTS (checkout of riscv-software-src/riscv-tests WITH submodule env/ initialised), OUT.
# Known/expected failures (optional features not implemented): rv32mi/breakpoint (Sdtrig), rv32mi/pmpaddr (PMP).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=$HERE/../rtl; O=${OUT:-/tmp/rv-regress}; mkdir -p "$O"
IVERILOG=${IVERILOG:-iverilog}; VVP=${VVP:-vvp}; T=${RISCV_TESTS:?set RISCV_TESTS}
[ -n "${RVGCC_DIR:-}" ] && PATH=$RVGCC_DIR:$PATH
IV="$IVERILOG ${IVL_B:+-B $IVL_B} -g2012"
fails=0; total=0; expected_fail="rv32mi/breakpoint rv32mi/pmpaddr"
report() { total=$((total+1)); if [ "$2" = PASS ]; then echo "PASS $1"; elif [[ " $expected_fail " == *" $1 "* ]]; then echo "XFAIL $1 (optional feature)"; else echo "FAIL $1"; fails=$((fails+1)); fi; }
hex() { riscv64-unknown-elf-objcopy -O binary "$1" "$1.bin"; python3 -c "
import struct,sys
d=open('$1.bin','rb').read(); d+=b'\0'*(-len(d)%4)
open('$1.hex','w').write(''.join('%08x\n'%w for w in struct.unpack('<%dI'%(len(d)//4),d)))"; }

$IV -s rv32i_tb -o $O/tb $R/rv32i_core.sv $R/q4_dot_product.sv $R/rv32i_tb.sv 2>&1 | grep error
report hand-written/rv32i_tb "$($VVP $O/tb | grep -o PASS | head -1)"
for cfg in "8 6 1234" "5 12 99"; do
  set -- $cfg; Rn=$1; Gn=$2; Sd=$3
  python3 $HERE/gen_q4_vectors.py --rows $Rn --groups $Gn --seed $Sd --out $O/q4_${Rn}_${Gn}
  printf 'module tb_wrap; q4_gemv_tb #(.ROWS(%d), .GROUPS(%d)) t(); endmodule\n' $Rn $Gn > $O/wrap_${Rn}_${Gn}.sv
  $IV -s tb_wrap -o $O/q4tb $R/q4_gemv_tile.sv $R/q4_gemv_tb.sv $O/wrap_${Rn}_${Gn}.sv 2>&1 | grep error
  report "q4-tile ${Rn}x${Gn}" "$($VVP $O/q4tb +mem=$O/q4_${Rn}_${Gn}.mem +act=$O/q4_${Rn}_${Gn}.act +exp=$O/q4_${Rn}_${Gn}.exp | grep -o PASS | head -1)"
done

$IV -s riscv_tests_tb -o $O/sim1 $R/rv32i_core.sv $R/riscv_tests_tb.sv 2>&1 | grep error
$IV -s riscv_tests_tb -Priscv_tests_tb.HALT=0 -o $O/sim0 $R/rv32i_core.sv $R/riscv_tests_tb.sv 2>&1 | grep error

# RVV subset tests (HALT_ON_TRAP images, ECALL exit)
for t in rvv_test illegal_masked; do
  riscv64-unknown-elf-gcc -march=rv32iv -mabi=ilp32 -mno-relax -static -nostdlib -nostartfiles -Wl,--no-relax \
    -Wl,-Ttext=0 -Wl,-Tdata=0x4000 -o $O/$t.elf $HERE/rvv/$t.S && hex $O/$t.elf
  x=""; [ $t = illegal_masked ] && x="+expect=2"
  report rvv/$t "$($VVP $O/sim1 +hex=$O/$t.elf.hex $x | grep -oE 'PASS|FAIL' | head -1)"
done

# Official riscv-tests, "p" environment (machine mode, tohost). Linker script places .tohost after .text.init.
printf 'SECTIONS { . = 0x0; .text.init : { *(.text.init) } . = ALIGN(0x1000); .tohost : { *(.tohost) } .text : { *(.text) } . = 0x4000; .data : { *(.data) } .bss : { *(.bss) } }\n' > $O/link.ld
FL="-march=rv32i_zicsr_zifencei -mabi=ilp32 -mno-relax -static -mcmodel=medany -nostdlib -nostartfiles -Wl,--no-relax -T $O/link.ld -I $T/env/p -I $T/env -I $T/isa/macros/scalar"
for s in rv32ui rv32mi; do
  for f in $T/isa/$s/*.S; do t=$(basename $f .S); [ $t = Makefrag ] && continue
    riscv64-unknown-elf-gcc $FL -o $O/$s-$t.elf $f 2>/dev/null || { report $s/$t BUILDFAIL; continue; }
    hex $O/$s-$t.elf
    th=$(riscv64-unknown-elf-nm $O/$s-$t.elf | awk '$3=="tohost"{print $1}')
    report $s/$t "$($VVP $O/sim0 +hex=$O/$s-$t.elf.hex +tohost=$th | grep -oE 'PASS|FAIL' | head -1)"
  done
done
echo "SUMMARY: $total suites, $fails unexpected failures"; [ $fails -eq 0 ]
