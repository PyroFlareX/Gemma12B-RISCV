"""Gemma 3 12B decode roofline. Config from google/gemma-3-12b text_config (verify against HF config.json)."""
from dataclasses import dataclass

@dataclass
class Cfg:
    layers: int = 48
    d: int = 3840
    heads: int = 16
    kv_heads: int = 8
    head_dim: int = 256
    ffn: int = 15360
    vocab: int = 262144
    window: int = 1024      # sliding-window layers
    global_every: int = 6   # 5 local : 1 global

def params(c: Cfg):
    attn = c.d * c.heads * c.head_dim + 2 * c.d * c.kv_heads * c.head_dim + c.heads * c.head_dim * c.d
    mlp = 3 * c.d * c.ffn
    return {"attn": attn * c.layers, "mlp": mlp * c.layers, "embed": c.vocab * c.d}

def decode(c: Cfg, ctx: int, w_bits: float, kv_bits: float, bw_gbs: float, tops: float):
    p = params(c)
    w_bytes = (p["attn"] + p["mlp"] + p["embed"]) * w_bits / 8  # tied embed/lm_head read once per token
    n_glob = c.layers // c.global_every
    n_loc = c.layers - n_glob
    kv_tok = 2 * c.kv_heads * c.head_dim * kv_bits / 8
    kv_bytes = kv_tok * (n_glob * ctx + n_loc * min(ctx, c.window))
    flops = 2 * (p["attn"] + p["mlp"] + p["embed"]) + 4 * c.heads * c.head_dim * (n_glob * ctx + n_loc * min(ctx, c.window))
    t_mem = (w_bytes + kv_bytes) / (bw_gbs * 1e9)
    t_cmp = flops / (tops * 1e12)
    return dict(weights_GB=w_bytes / 1e9, kv_GB=kv_bytes / 1e9, t_mem_ms=t_mem * 1e3,
                t_cmp_ms=t_cmp * 1e3, tok_s=1 / max(t_mem, t_cmp),
                bound="memory" if t_mem > t_cmp else "compute")

if __name__ == "__main__":
    c = Cfg()
    print({k: f"{v/1e9:.2f}B" for k, v in params(c).items()})
    print(f"{'wbits':>5} {'ctx':>7} {'BW GB/s':>8} {'W GB':>6} {'KV GB':>6} {'tok/s':>7} bound")
    for wb in (16, 8, 4):
        for ctx in (4096, 32768, 131072):
            for bw in (60, 250, 800):
                r = decode(c, ctx, wb, 8, bw, tops=4)
                print(f"{wb:>5} {ctx:>7} {bw:>8} {r['weights_GB']:>6.1f} {r['kv_GB']:>6.2f} {r['tok_s']:>7.1f} {r['bound']}")
