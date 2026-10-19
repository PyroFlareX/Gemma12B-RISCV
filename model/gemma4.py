"""Gemma 4 12B text-only inference sizing; ideal lower bounds, not a performance claim.

Source: https://huggingface.co/google/gemma-4-12B/resolve/main/config.json
Projection layout: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py
Decimal GB and GB/s throughout. Weight traffic assumes the entire model is streamed
once per decode step per batch; KV is read once per query and written once per new token.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    layers: int = 48
    hidden: int = 3840
    heads: int = 16
    local_kv_heads: int = 8
    global_kv_heads: int = 1
    local_head_dim: int = 256
    global_head_dim: int = 512
    intermediate: int = 15360
    vocab: int = 262144
    local_window: int = 1024
    global_every: int = 6
    max_context: int = 262144


def counts(c: Config = Config()) -> dict[str, int]:
    """Matrix weights only; tied embedding/output matrix counted once."""
    local = c.layers - c.layers // c.global_every
    glob = c.layers // c.global_every
    def attention(kv_heads: int, dim: int, tied_kv: bool) -> int:
        q_and_o = 2 * c.hidden * c.heads * dim
        kv = (1 if tied_kv else 2) * c.hidden * kv_heads * dim
        return q_and_o + kv
    return {
        "local_attention": local * attention(c.local_kv_heads, c.local_head_dim, False),
        "global_attention": glob * attention(c.global_kv_heads, c.global_head_dim, True),
        "mlp": c.layers * 3 * c.hidden * c.intermediate,
        "embedding_output": c.vocab * c.hidden,
    }


def kv_bytes(c: Config, context: int, kv_bits: int, batch: int = 1) -> float:
    """Resident K+V cache at a given context, excluding allocator/padding overhead."""
    local = c.layers - c.layers // c.global_every
    glob = c.layers // c.global_every
    return batch * 2 * kv_bits / 8 * (
        local * c.local_kv_heads * c.local_head_dim * min(context, c.local_window)
        + glob * c.global_kv_heads * c.global_head_dim * context
    )


def decode(context: int, weight_bits: int = 4, kv_bits: int = 8,
           bandwidth_gbs: float = 500, tops: float = 16, batch: int = 1,
           efficiency: float = 1.0, c: Config = Config()) -> dict[str, float]:
    """Optimistic aggregate throughput and per-step time for synchronous batching.

    `efficiency` scales *sustained* vs pin bandwidth; no claim about GDDR efficiency.
    Assumes one full weight stream per step and per-query KV read plus new KV write.
    For batch>1 output is aggregate tokens/s (per-sequence = aggregate/batch).
    """
    if not 1 <= context <= c.max_context or batch < 1:
        raise ValueError("context must fit model max_context; batch must be positive")
    if weight_bits <= 0 or kv_bits <= 0 or bandwidth_gbs <= 0 or tops <= 0 or not 0 < efficiency <= 1:
        raise ValueError("positive bits/bandwidth/TOPS and efficiency in (0,1] required")
    p = counts(c)
    weights = sum(p.values()) * weight_bits / 8
    # KV scan reads existing tokens; one appended K/V per layer per batch.
    kv_read = kv_bytes(c, context, kv_bits, batch)
    kv_write = kv_bytes(c, 1, kv_bits, batch)
    local = c.layers - c.layers // c.global_every
    glob = c.layers // c.global_every
    # Multiply-add counts as two ops; attention QK + AV is ~4 ops per head/dim/token.
    attn_ops = 4 * batch * c.heads * (
        local * c.local_head_dim * min(context, c.local_window)
        + glob * c.global_head_dim * context
    )
    ops = 2 * sum(p.values()) * batch + attn_ops
    memory_s = (weights + kv_read + kv_write) / (bandwidth_gbs * efficiency * 1e9)
    compute_s = ops / (tops * 1e12)
    step_s = max(memory_s, compute_s)
    return {
        "weights_GB": weights / 1e9,
        "kv_resident_GB": kv_read / 1e9,
        "step_traffic_GB": (weights + kv_read + kv_write) / 1e9,
        "step_memory_ms": memory_s * 1e3,
        "step_compute_ms": compute_s * 1e3,
        "aggregate_tokens_s_ceiling": batch / step_s,
        "per_sequence_tokens_s_ceiling": 1 / step_s,
    }


# 12B Unified is encoder-free: image patches are linearly projected into the decoder, so every
# image token costs a full decoder pass. Budgets from the model card; config default is 280.
IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)
IMAGE_DELIMITERS = 2  # boi + eoi per image


def image_tokens(images: int, budget: int = 280) -> int:
    if budget not in IMAGE_TOKEN_BUDGETS or images < 0:
        raise ValueError(f"budget must be one of {IMAGE_TOKEN_BUDGETS}; images >= 0")
    return images * (budget + IMAGE_DELIMITERS)


def _sum_min(lo: int, hi: int, cap: int | None) -> int:
    """sum over k in (lo, hi] of min(k, cap): keys attended by causal queries at positions lo..hi-1."""
    def s(n: int) -> int:
        if cap is None or n <= cap:
            return n * (n + 1) // 2
        return cap * (cap + 1) // 2 + (n - cap) * cap
    return s(hi) - s(lo)


def prefill(new_tokens: int, cached_tokens: int = 0, chunk: int = 2048, weight_bits: int = 4,
            kv_bits: int = 8, bandwidth_gbs: float = 500, tops: float = 16,
            efficiency: float = 1.0, c: Config = Config()) -> dict[str, float]:
    """Ideal latency for prefilling `new_tokens` after `cached_tokens` of reused prefix KV.

    Chunked prefill streams all non-embedding weights once per chunk; logits only for the last
    token; input embeddings are row lookups. Attention reads prior global KV (and the local
    window) from DRAM once per chunk; intra-chunk K/V stay on chip.
    """
    if new_tokens < 1 or cached_tokens < 0 or cached_tokens + new_tokens > c.max_context or chunk < 1:
        raise ValueError("token counts must fit max_context; new_tokens and chunk positive")
    if weight_bits <= 0 or kv_bits <= 0 or bandwidth_gbs <= 0 or tops <= 0 or not 0 < efficiency <= 1:
        raise ValueError("positive bits/bandwidth/TOPS and efficiency in (0,1] required")
    p = counts(c)
    embed = p["embedding_output"]
    body = sum(p.values()) - embed
    local = c.layers - c.layers // c.global_every
    glob = c.layers // c.global_every
    end = cached_tokens + new_tokens
    attn_ops = 4 * c.heads * (local * c.local_head_dim * _sum_min(cached_tokens, end, c.local_window)
                              + glob * c.global_head_dim * _sum_min(cached_tokens, end, None))
    ops = 2 * body * new_tokens + 2 * embed + attn_ops
    chunks = -(-new_tokens // chunk)
    wbyte = weight_bits / 8
    weight_traffic = chunks * body * wbyte + embed * wbyte + new_tokens * c.hidden * wbyte
    kv_local_tok = 2 * kv_bits / 8 * local * c.local_kv_heads * c.local_head_dim
    kv_glob_tok = 2 * kv_bits / 8 * glob * c.global_kv_heads * c.global_head_dim
    kv_write = new_tokens * (kv_local_tok + kv_glob_tok)
    starts = range(cached_tokens, end, chunk)
    kv_read = sum(s * kv_glob_tok + min(s, c.local_window) * kv_local_tok for s in starts)
    memory_s = (weight_traffic + kv_write + kv_read) / (bandwidth_gbs * efficiency * 1e9)
    compute_s = ops / (tops * 1e12)
    return {
        "ops_P": ops / 1e15,
        "attention_share": attn_ops / ops,
        "dram_GB": (weight_traffic + kv_write + kv_read) / 1e9,
        "compute_s": compute_s,
        "memory_s": memory_s,
        "latency_s": max(compute_s, memory_s),
        "chunk_activation_MB": chunk * c.hidden * 2 / 1e6,  # one BF16 residual row block
    }


def concurrent_sequences(capacity_gb: float, context: int, weight_bits: int = 4, kv_bits: int = 8,
                         scale_bytes_per_group: int = 2, group: int = 32,
                         reserve_gb: float = 0.5, c: Config = Config()) -> int:
    """Sequences whose full KV fits beside ONE shared weight copy (weights + group scales)."""
    if capacity_gb <= 0 or not 1 <= context <= c.max_context or reserve_gb < 0:
        raise ValueError("positive capacity, context within max_context, nonnegative reserve")
    n = sum(counts(c).values())
    resident = n * weight_bits / 8 + n / group * scale_bytes_per_group + reserve_gb * 1e9
    free = capacity_gb * 1e9 - resident
    return max(0, int(free // kv_bytes(c, context, kv_bits)))


if __name__ == "__main__":
    c = Config()
    p = counts(c)
    print("Gemma 4 12B text matrix parameters:", {k: round(v / 1e9, 3) for k, v in p.items()},
          "total", round(sum(p.values()) / 1e9, 3), "B")
    print("Ideal lower-bound decode ceilings; 16 TOPS effective; 100% bandwidth efficiency")
    print(f"{'bits':>4} {'ctx':>7} {'batch':>5} {'GB/s':>5} {'weights GB':>10} {'KV GB':>8} {'aggregate tok/s':>17} {'seq tok/s':>10}")
    for bits in (4, 8):
        for context in (4096, 32768, 262144):
            for batch in (1, 8):
                for bw in (250, 500, 700):
                    r = decode(context, bits, 8, bw, 16, batch, c=c)
                    print(f"{bits:>4} {context:>7} {batch:>5} {bw:>5} {r['weights_GB']:>10.2f} {r['kv_resident_GB']:>8.2f} {r['aggregate_tokens_s_ceiling']:>17.1f} {r['per_sequence_tokens_s_ceiling']:>10.1f}")

    text = 40_000
    print("\nPrefill: 16 images + 40,000 text tokens; Q4 weights, 8-bit KV, 500 GB/s, chunk 2048")
    print(f"{'scenario':<34} {'tokens':>6} {'PetaOps':>7} {'attn%':>5} " +
          " ".join(f"{t:>5}T s" for t in (16, 100, 400, 1000)))
    scenarios = [(f"images@{b} + text, no cache", image_tokens(16, b) + text, 0) for b in (70, 280, 1120)]
    scenarios.append(("text prefix cached, images@280", image_tokens(16, 280), text))
    for name, new, cached in scenarios:
        rs = [prefill(new, cached, tops=t) for t in (16, 100, 400, 1000)]
        print(f"{name:<34} {new:>6} {rs[0]['ops_P']:>7.3f} {100 * rs[0]['attention_share']:>5.1f} " +
              " ".join(f"{r['latency_s']:>7.2f}" for r in rs))
    print("Chunk size vs bound at 400 TOPS, 500 GB/s (16 images@280 + text):")
    for ch in (64, 128, 256, 512, 2048):
        r = prefill(image_tokens(16, 280) + text, chunk=ch, tops=400)
        print(f"  chunk {ch:>5}: compute {r['compute_s']:.2f} s, memory {r['memory_s']:.2f} s, "
              f"activation block {r['chunk_activation_MB']:.1f} MB")
    ctx = image_tokens(16, 280) + text
    print(f"Concurrent {ctx}-token sequences beside one Q4 weight copy (+FP16 group-32 scales, 0.5 GB reserve):",
          {f"{g} GB": concurrent_sequences(g, ctx) for g in (8, 12, 16, 24, 32)})
