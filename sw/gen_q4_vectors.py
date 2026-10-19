"""Generate Q4 GEMV test vectors for rtl/q4_gemv_tile.sv from the reference packing.

Emits two files:
  <out>.mem   one 32-bit word per line: per row, per group, four little-endian
              packed-Q4 words (low nibble first) then the signed Q16.16 scale word
  <out>.exp   one signed 64-bit expected accumulator per row

The RTL result is compared against integer arithmetic in Q16.16 fixed point, which
is what the tile implements; sw/q4_gemv.py defines the nibble packing.
"""
import argparse, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4_gemv import pack_q4, BLOCK


def vectors(rows: int, groups: int, seed: int):
    rng = random.Random(seed)
    packed = b""
    scales = []
    for _ in range(rows):
        for _ in range(groups):
            packed += pack_q4([rng.randint(-8, 7) for _ in range(BLOCK)])
            scales.append(rng.randint(-32768, 32767))
    acts = [rng.randint(-128, 127) for _ in range(groups * BLOCK)]

    words = []
    for r in range(rows):
        for g in range(groups):
            base = (r * groups + g) * (BLOCK // 2)
            for w in range(4):
                words.append(int.from_bytes(packed[base + w * 4:base + w * 4 + 4], "little"))
            words.append(scales[r * groups + g] & 0xFFFFFFFF)

    expected = []
    for r in range(rows):
        acc = 0
        for g in range(groups):
            base = (r * groups + g) * (BLOCK // 2)
            dot = 0
            for o in range(BLOCK // 2):
                b = packed[base + o]
                lo, hi = b & 15, b >> 4
                lo = lo - 16 if lo & 8 else lo
                hi = hi - 16 if hi & 8 else hi
                dot += lo * acts[g * BLOCK + 2 * o] + hi * acts[g * BLOCK + 2 * o + 1]
            acc += dot * scales[r * groups + g]
        expected.append(acc)
    return words, expected, acts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--groups", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/tmp/q4rand")
    a = ap.parse_args()
    words, expected, acts = vectors(a.rows, a.groups, a.seed)
    with open(f"{a.out}.mem", "w") as f:
        f.write("\n".join(f"{w:08x}" for w in words) + "\n")
    with open(f"{a.out}.exp", "w") as f:
        f.write("\n".join(f"{e & 0xFFFFFFFFFFFFFFFF:016x}" for e in expected) + "\n")
    with open(f"{a.out}.act", "w") as f:
        f.write("\n".join(f"{x & 0xFF:02x}" for x in acts) + "\n")
    print(f"rows={a.rows} groups={a.groups} words={len(words)}")
    print("expected:", expected)


if __name__ == "__main__":
    main()
