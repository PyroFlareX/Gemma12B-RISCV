"""Signed-nibble order, row stride and group scale regression checks."""
import unittest

from q4_gemv import gemv, pack_q4


class Q4KernelTest(unittest.TestCase):
    def test_negative_edges_and_nibble_order(self):
        weights = [-8, 7] * 16
        self.assertEqual(pack_q4(weights), bytes([0x78] * 16))
        self.assertEqual(gemv(pack_q4(weights), [0.5], [1.0] * 32, 1), [-8.0])

    def test_blocks_rows_and_distinct_scales(self):
        weights = [1] * 32 + [-2] * 32 + [3] * 32 + [-4] * 32
        result = gemv(pack_q4(weights), [0.5, 2.0, 1.0, 0.25], [1.0] * 32 + [2.0] * 32, 2)
        self.assertEqual(result, [16 - 256, 96 - 64])

    def test_invalid_dimensions_and_range(self):
        with self.assertRaises(ValueError):
            pack_q4([0] * 31)
        with self.assertRaises(ValueError):
            pack_q4([8] * 32)
        with self.assertRaises(ValueError):
            gemv(bytes(16), [], [1.0] * 32, 1)


if __name__ == "__main__":
    unittest.main()
