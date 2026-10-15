# Gemma12B-RISCV

Exploration: a RISC-V based accelerator purpose-built for efficient **Gemma 3 12B** decode/prefill.

## Layout
- `model/gemma12b.py` — analytical roofline (weights/KV bytes, FLOPs, tok/s vs. memory BW & TOPS). Run: `python3 model/gemma12b.py`
- `rtl/` — hardware design (empty; start with an RVV-based GEMV tile + dequant unit)
- `sw/` — kernels / reference runtime (empty)
- `docs/` — design notes

## Premises
- Decode is memory-bandwidth bound: ~12B params, so tok/s ≈ BW / bytes-per-token. Quantization (4-bit weights) and DRAM bandwidth dominate over raw TOPS.
- Gemma 3 uses 5:1 local(1024 window):global attention, so KV traffic stays small at long context.
- Config values in `gemma12b.py` should be checked against the HF `config.json`.

## Next steps
1. Fix target: bandwidth (LPDDR5X/GDDR/HBM), area/process, quant format (e.g. Q4 group-wise).
2. Sweep roofline to size compute vs. bandwidth.
3. Pick ISA base: RV64GC + RVV 1.0 + custom dequant/GEMV ops; simulate with Spike/QEMU.
4. RTL for one tile (Chisel/Rocket-Chip/Chipyard or Verilog + Verilator).

Toolchain not installed yet (riscv gcc, verilator, spike).
