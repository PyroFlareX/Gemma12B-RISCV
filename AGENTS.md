# Gemma12B-RISCV Agent Instructions

## Goal
Design a RISC-V-based chip/accelerator built for efficient inference of **Gemma 3 12B** (initial target; may generalize later). Exploratory, early stage.

## Status
- Roofline model done: `model/gemma12b.py` (`python3 model/gemma12b.py`).
- Findings: ~11.8B params (attn 2.26B, MLP 8.49B, embed 1.01B). Decode is memory-bound in all swept cases (e.g. 8-bit weights @ 250 GB/s ≈ 21 tok/s). Bandwidth + weight quantization dominate over TOPS; choose memory system and quant format before compute array.
- Model config values were written from memory (48 layers, d=3840, 16 heads / 8 KV heads, head_dim 256, ffn 15360, vocab 262144, 5:1 local:global, window 1024). **Unverified: check against HF config.json.**
- `rtl/`, `sw/`, `docs/` are empty.

## Open decisions (ask user)
1. Target memory: LPDDR5X / GDDR / HBM.
2. Weight quant format (e.g. Q4 group-wise).
3. RTL flow: Chipyard/Chisel vs plain Verilog + Verilator.

## Toolchain
Not installed: riscv gcc, verilator, spike. User runs NixOS; add via flake rather than system installs.

## Conventions
Informal; no CI. Do not add docs/scaffolding beyond what's requested. Single repo root (OMP ACP has no multi-root support).
