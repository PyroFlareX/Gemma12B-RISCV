#!/usr/bin/env bash
# riscv-arch-test (ACT) signature-comparison runner for rtl/rv32i_core.sv; not certification.
# Two-pass per test: (1) build the signature-dump ELF and run Sail to produce the
# reference signature; (2) run the SAME ELF on the RTL and compare the signature
# region against Sail's reference. The ACT verdict is the signature match, not the
# tohost value (HTIF writes the summary string to the tohost address).
#
# Env: ACT (riscv-arch-test checkout), SAIL (sail_riscv_sim binary), RVGCC_DIR,
#      IVERILOG, VVP, IVL_B, OUT.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); R=$HERE/../rtl; O=${OUT:-/tmp/rv-act}; mkdir -p "$O"
ACT=${ACT:?set ACT to a riscv-arch-test checkout}
SAIL=${SAIL:?set SAIL to sail_riscv_sim}
IVERILOG=${IVERILOG:-iverilog}; VVP=${VVP:-vvp}
[ -n "${RVGCC_DIR:-}" ] && PATH=$RVGCC_DIR:$PATH
IV="$IVERILOG ${IVL_B:+-B $IVL_B} -g2012"

hex() { riscv64-unknown-elf-objcopy -O binary "$1" "$1.bin"; python3 -c "
import struct
d=open('$1.bin','rb').read(); d+=b'\0'*(-len(d)%4)
open('$1.hex','w').write(''.join('%08x\n'%w for w in struct.unpack('<%dI'%(len(d)//4),d)))"; }

# Signature region + tohost address from the ELF symbols.
addrs() { riscv64-unknown-elf-nm "$1" | awk '$3=="begin_signature"{b=$1} $3=="end_signature"{e=$1} $3=="tohost"{t=$1} END{print b,e,t}'; }

$IV -s riscv_tests_tb -Priscv_tests_tb.HALT=0 -o $O/sim $R/rv32i_core.sv $R/riscv_tests_tb.sv 2>&1 | grep error

total=0; fails=0
for f in $ACT/tests/rv32i/I/*.S; do
  t=$(basename $f .S)
  riscv64-unknown-elf-gcc -march=rv32i_zicsr_zifencei -mabi=ilp32 -mno-relax -mcmodel=medany \
    -nostdlib -nostartfiles -static -Wl,--no-relax -Wl,-T$ACT/rtl-test/link.ld \
    -DTEST_FLEN=32 "-DTEST_FILE=\"$t\"" -DSAIL_CLINT_BASE_ADDRESS=0x2000000 \
    -DSAIL_SIMPLE_INTERRUPT_GENERATOR_BASE_ADDRESS=0x200000 \
    -I$ACT/tests/env -I$ACT/rtl-test \
    -o $O/$t.elf $f 2>/dev/null || { echo "BUILDFAIL $t"; fails=$((fails+1)); continue; }
  hex $O/$t.elf
  read sb se th <<< "$(addrs $O/$t.elf)"
  $SAIL --config $ACT/rtl-test/sail.json --test-signature=$O/$t.sig --signature-granularity 4 $O/$t.elf >/dev/null 2>&1 || { echo "SAILFAIL $t"; fails=$((fails+1)); continue; }
  total=$((total+1))
  verdict=$(timeout 120 $VVP $O/sim +hex=$O/$t.elf.hex +tohost=$th +sigbegin=$sb +sigend=$se +sigref=$O/$t.sig | grep -oE 'PASS|FAIL' | head -1)
  if [ "$verdict" = PASS ]; then echo "PASS $t"; else echo "FAIL $t"; fails=$((fails+1)); fi
done
echo "SUMMARY: $total ACT tests, $fails failures"
[ $fails -eq 0 ]
