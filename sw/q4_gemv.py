"""Reference block-Q4 signed GEMV for an accelerator DMA contract.

Each 32-element block: 16 packed bytes (low nibble first, signed two's
complement -8..7) and one BF16/FP32 software scale per block. The reference
uses Python floats, not a bit-exact hardware floating-point oracle. Software
packs separately from native RVV; this is a custom accelerator data format.
"""
from collections.abc import Sequence

BLOCK = 32


def pack_q4(weights: Sequence[int]) -> bytes:
    if len(weights) % BLOCK:
        raise ValueError("weight count must be a multiple of 32")
    if any(w < -8 or w > 7 for w in weights):
        raise ValueError("signed Q4 weights must be in [-8,7]")
    return bytes((weights[i] & 15) | ((weights[i + 1] & 15) << 4)
                 for i in range(0, len(weights), 2))


def gemv(packed: bytes, scales: Sequence[float], activations: Sequence[float],
         rows: int) -> list[float]:
    """Row-major blocks; each block's 16 bytes followed logically by one scale."""
    if rows < 1 or len(activations) % BLOCK:
        raise ValueError("rows must be positive and activations block-aligned")
    blocks_per_row = len(activations) // BLOCK
    if len(packed) != rows * len(activations) // 2 or len(scales) != rows * blocks_per_row:
        raise ValueError("weight/scales dimensions do not match rows and activations")
    result = [0.0] * rows
    for row in range(rows):
        acc = 0.0
        for block in range(blocks_per_row):
            base = (row * blocks_per_row + block) * (BLOCK // 2)
            a_base = block * BLOCK
            dot = 0.0
            for offset in range(BLOCK // 2):
                packed_byte = packed[base + offset]
                lo = packed_byte & 15
                hi = packed_byte >> 4
                dot += (lo - 16 if lo & 8 else lo) * activations[a_base + 2 * offset]
                dot += (hi - 16 if hi & 8 else hi) * activations[a_base + 2 * offset + 1]
            acc += dot * scales[row * blocks_per_row + block]
        result[row] = acc
    return result


if __name__ == "__main__":
    w = [(-1) ** i * (i % 8) for i in range(BLOCK)]
    x = [float(i % 3) for i in range(BLOCK)]
    print(gemv(pack_q4(w), [0.125], x, 1))
