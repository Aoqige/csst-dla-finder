"""Conv-stem Transformer with Rotary Position Embedding (RoPE), five-head.

v5 evolution from v3c (`transformer_conv_stem_5head.py`):
  - replaces the learned absolute positional embedding + nn.TransformerEncoder
    with a custom RoPE attention + pre-norm encoder layer.
  - RoPE encodes relative position directly in Q/K (no absolute pos_emb added
    to the input), the standard relative-position upgrade for 1D sequences.

Design goal (v5 - isolate the position-encoding variable):
  - conv stem L3 (+/-9px) + cosine + all v3c hyperparameters unchanged,
    ONLY swapping absolute-pos-emb nn.TransformerEncoder -> RoPE encoder.
  - RoPE adds zero learned parameters (rotations are deterministic), so v5
    has the same param budget as v3c (~2.5M) for a fair comparison.

Output contract identical to v3c:
  center_logits / region_logits / lognhi_raw / offset_raw / count_logits
"""
from __future__ import annotations

from typing import Any


PIPELINE_BIAS = 20.3


def _build_transformer_conv_stem_rope_5head(
    in_channels: int = 6,
    d_model: int = 192,
    nhead: int = 8,
    num_layers: int = 4,
    dim_ff: int = 768,
    dropout: float = 0.1,
    conv_kernel: int = 7,
    num_conv_layers: int = 1,
    n_count_classes: int = 3,
    use_offset: bool = True,
    max_len: int = 1024,
) -> Any:
    import torch
    nn = torch.nn

    if conv_kernel % 2 == 0:
        raise ValueError(f"conv_kernel must be odd (got {conv_kernel})")
    assert d_model % nhead == 0, "d_model must be divisible by nhead"
    pad = conv_kernel // 2
    d_head = d_model // nhead

    class ConvStem(nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = nn.ModuleList()
            for i in range(num_conv_layers):
                in_c = in_channels if i == 0 else d_model
                self.convs.append(nn.Conv1d(in_c, d_model, conv_kernel, padding=pad))
            self.act = nn.GELU()
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            h = x
            for conv in self.convs:
                h = self.act(conv(h))
            return self.norm(h.transpose(1, 2))

    class RoPEAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.nhead = nhead
            self.d_head = d_head
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.drop = nn.Dropout(dropout)
            half = d_head // 2
            freqs = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / d_head))
            self.register_buffer("rope_freqs", freqs, persistent=False)

        def _rope(self, x, L):
            pos = torch.arange(L, device=x.device, dtype=self.rope_freqs.dtype)
            angles = torch.outer(pos, self.rope_freqs)
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
            d = x.shape[-1]
            x1, x2 = x[..., : d // 2], x[..., d // 2 :]
            rot = torch.cat([-x2, x1], dim=-1)
            return x * cos + rot * sin

        def forward(self, x):
            B, L, _ = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.nhead, self.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q = self._rope(q, L)
            k = self._rope(k, L)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout if self.training else 0.0
            )
            out = out.transpose(1, 2).reshape(B, L, -1)
            return self.out(out)

    class RoPEEncoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn = RoPEAttention()
            self.norm2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

    class TransformerConvStemRoPE5Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = in_channels
            self.d_model = d_model
            self.use_offset = bool(use_offset)
            self.max_len = max_len
            self.conv_kernel = conv_kernel
            self.num_conv_layers = num_conv_layers

            self.conv_stem = ConvStem()
            self.encoder = nn.Sequential(*[RoPEEncoderLayer() for _ in range(num_layers)])
            self.final_norm = nn.LayerNorm(d_model)
            self.center_head = nn.Linear(d_model, 1)
            self.region_head = nn.Linear(d_model, 1)
            self.lognhi_head = nn.Linear(d_model, 1)
            self.count_head = nn.Linear(d_model, n_count_classes)
            if self.use_offset:
                self.offset_head = nn.Linear(d_model, 1)
            self._init_heads()

        def _init_heads(self):
            for h in (self.center_head, self.region_head, self.lognhi_head, self.count_head):
                nn.init.zeros_(h.bias)
                nn.init.xavier_uniform_(h.weight)
            if self.use_offset:
                nn.init.zeros_(self.offset_head.bias)
                nn.init.xavier_uniform_(self.offset_head.weight)

        def forward(self, x):
            if x.ndim != 3:
                raise ValueError(f"expected [B, C, L], got shape {tuple(x.shape)}")
            B, C, L = x.shape
            if L > self.max_len:
                raise ValueError(f"sequence length {L} exceeds buffer {self.max_len}")
            h = self.conv_stem(x)
            h = self.encoder(h)
            h = self.final_norm(h)
            center_logits = self.center_head(h).squeeze(-1)
            region_logits = self.region_head(h).squeeze(-1)
            lognhi_raw = self.lognhi_head(h).squeeze(-1)
            count_logits = self.count_head(h.mean(dim=1))
            out = {
                "center_logits": center_logits,
                "region_logits": region_logits,
                "lognhi_raw": lognhi_raw,
                "count_logits": count_logits,
            }
            if self.use_offset:
                out["offset_raw"] = torch.tanh(self.offset_head(h).squeeze(-1))
            return out

    return TransformerConvStemRoPE5Head()


def main() -> None:
    torch = __import__("torch")
    x = torch.zeros(4, 6, 681)
    m = _build_transformer_conv_stem_rope_5head(num_conv_layers=3)
    with torch.no_grad():
        out = m(x)
    n = sum(p.numel() for p in m.parameters())
    print(f"v5 RoPE L3 params={n:,}")
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)} range=[{v.min():.3f},{v.max():.3f}]")


if __name__ == "__main__":
    main()
