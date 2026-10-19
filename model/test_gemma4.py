"""Boundary checks for capacity and traffic decisions in the Gemma 4 sizing model."""
import unittest

from gemma4 import Config, concurrent_sequences, counts, decode, image_tokens, kv_bytes, prefill


class GemmaSizingTest(unittest.TestCase):
    def test_global_kv_is_one_head_and_local_window_saturates(self):
        c = Config()
        local_per_token = 2 * 40 * 8 * 256
        global_per_token = 2 * 8 * 1 * 512
        self.assertEqual(kv_bytes(c, 1024, 8), 1024 * (local_per_token + global_per_token))
        self.assertEqual(kv_bytes(c, 2048, 8) - kv_bytes(c, 1024, 8), 1024 * global_per_token)
        self.assertEqual(kv_bytes(c, 262144, 8), 2_315_255_808)

    def test_global_kv_tied_weights_but_cached_as_key_and_value(self):
        c = Config()
        q_o = 2 * c.hidden * c.heads * c.global_head_dim
        one_kv = c.hidden * c.global_kv_heads * c.global_head_dim
        self.assertEqual(counts(c)["global_attention"], 8 * (q_o + one_kv))

    def test_batch_reuses_weights_but_multiplies_kv_traffic(self):
        one = decode(262144, batch=1)
        eight = decode(262144, batch=8)
        self.assertEqual(one["weights_GB"], eight["weights_GB"])
        self.assertAlmostEqual(eight["kv_resident_GB"], 8 * one["kv_resident_GB"])
        self.assertLess(eight["per_sequence_tokens_s_ceiling"], one["per_sequence_tokens_s_ceiling"])
        self.assertGreater(eight["aggregate_tokens_s_ceiling"], one["aggregate_tokens_s_ceiling"])

    def test_nonphysical_inputs_rejected(self):
        for kwargs in ({"context": 262145}, {"context": 0}, {"context": 1024, "batch": 0},
                       {"context": 1024, "efficiency": 1.1}, {"context": 1024, "bandwidth_gbs": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                decode(**kwargs)

    def test_prefix_cache_split_matches_one_shot_attention_work(self):
        # Prefilling A then B (with A cached) must do the same attention work as A+B at once;
        # only the once-per-call logits matmul (2 * embed ops) is duplicated.
        c = Config()
        logits = 2 * counts(c)["embedding_output"]
        for a, b in ((500, 700), (1024, 1), (3000, 41512)):
            with self.subTest(a=a, b=b):
                whole = prefill(a + b)["ops_P"] * 1e15
                split = (prefill(a)["ops_P"] + prefill(b, cached_tokens=a)["ops_P"]) * 1e15
                self.assertAlmostEqual(split - logits, whole, delta=whole * 1e-12)

    def test_image_budget_and_capacity_boundaries(self):
        self.assertEqual(image_tokens(16, 280), 16 * 282)
        with self.assertRaises(ValueError):
            image_tokens(1, 300)
        self.assertEqual(concurrent_sequences(6, 44512), 0)  # 6 GB cannot hold weights + scales
        self.assertGreater(concurrent_sequences(16, 44512), concurrent_sequences(16, 262144))


if __name__ == "__main__":
    unittest.main()
