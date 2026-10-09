"""Per-pixel Transformer encoder with a small Conv1d stem (v1 hybrid).

Evolution from v0 (`transformer_5head.py`):
  - v0: per-pixel Linear projection (no local context)
  - v1: single Conv1d(kernel=7, stride=1) stem -> each token sees +/-3 pixels

Design goal (v1 - minimal local inductive bias):
  - one Conv1d stem replaces v0's Linear projection; everything else
    (positional embedding, pre-norm TransformerEncoder, five heads) is
    identical to v0 so the comparison directly isolates the effect of
    local context per token.
  - kernel=7, stride=1 keeps 1 pixel == 1 token (681 tokens), so the
    sequence length and attention cost are unchanged.
  - net parameter delta vs v0 is ~7k (Conv1d 6->192 k7 + LayerNorm 192
    minus Linear 6->192); total ~1.99M, still matching the CNN baseline
    (~2.0M) for a fair three-way comparison.

The builder also supports a deeper stem (`num_conv_layers` > 1) for later
experiments; v1 ships with a single layer.

Output contract (identical to v0 and the dilated CNN baseline):
  center_logits / region_logits / lognhi_raw / offset_raw / count_logits
"""
from __future__ import annotations

from typing import Any


# Pipeline convention: predicted logNHI = PIPELINE_BIAS + lognhi_raw.
# Keep in sync with hybrid_ensemble/model.py and train_hybrid.py.
PIPELINE_BIAS = 20.3


def _build_transformer_conv_stem_5head(
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
    """Construct the conv-stem Transformer five-head model.

    Parameters
    ----------
    in_channels:
        Input feature channels per pixel (6 for ``input_mode="flux"``).
    d_model / nhead / num_layers / dim_ff / dropout:
        Standard Transformer encoder hyper-parameters. Defaults match v0
        and the CNN baseline (~2.0M params total).
    conv_kernel:
        Conv1d kernel size for the stem. 7 gives +/-3 pixel context per
        token; must be odd so ``padding = conv_kernel // 2`` keeps length.
    num_conv_layers:
        Number of conv layers in the stem. 1 = minimal local bias (v1).
        Each added layer widens the receptive field by (kernel-1) pixels.
    max_len:
        Upper bound on sequence length for the learned positional embedding.
    """
    import torch
    nn = torch.nn

    if conv_kernel % 2 == 0:
        raise ValueError(f"conv_kernel must be odd (got {conv_kernel})")
    pad = conv_kernel // 2

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
            # x: [B, C, L]
            h = x
            for conv in self.convs:
                h = self.act(conv(h))          # [B, d_model, L]
            h = h.transpose(1, 2)             # [B, L, d_model]
            return self.norm(h)

    class TransformerConvStem5Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = in_channels
            self.d_model = d_model
            self.use_offset = bool(use_offset)
            self.max_len = max_len
            self.conv_kernel = conv_kernel
            self.num_conv_layers = num_conv_layers

            self.conv_stem = ConvStem()
            self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm: stable without warmup
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

            self.center_head = nn.Linear(d_model, 1)
            self.region_head = nn.Linear(d_model, 1)
            self.lognhi_head = nn.Linear(d_model, 1)
            self.count_head = nn.Linear(d_model, n_count_classes)
            if self.use_offset:
                self.offset_head = nn.Linear(d_model, 1)
            self._init_heads()

        def _init_heads(self) -> None:
            for head in (self.center_head, self.region_head, self.lognhi_head, self.count_head):
                nn.init.zeros_(head.bias)
                nn.init.xavier_uniform_(head.weight)
            if self.use_offset:
                nn.init.zeros_(self.offset_head.bias)
                nn.init.xavier_uniform_(self.offset_head.weight)

        def forward(self, x):
            # x: [B, C, L]
            if x.ndim != 3:
                raise ValueError(f"expected [B, C, L], got shape {tuple(x.shape)}")
            B, C, L = x.shape
            if L > self.max_len:
                raise ValueError(f"sequence length {L} exceeds positional buffer {self.max_len}")

            h = self.conv_stem(x)                       # [B, L, d_model]
            h = h + self.pos_emb[:L].unsqueeze(0)
            h = self.encoder(h)                          # [B, L, d_model]

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

    return TransformerConvStem5Head()


def main() -> None:
    """Smoke test: forward pass + parameter count for a few configs."""
    torch = __import__("torch")
    n_bins = 681
    x = torch.zeros(4, 6, n_bins)

    configs = [
        ("v1 k7 L1 conv (baseline)", 192, 8, 4, 768, 7, 1),
        ("k7 L2 conv", 192, 8, 4, 768, 7, 2),
        ("k7 L3 conv", 192, 8, 4, 768, 7, 3),
        ("k5 L1 conv", 192, 8, 4, 768, 5, 1),
    ]
    for name, d, h, L, ff, k, nl in configs:
        model = _build_transformer_conv_stem_5head(
            in_channels=6, d_model=d, nhead=h, num_layers=L, dim_ff=ff,
            conv_kernel=k, num_conv_layers=nl,
        )
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] params={n_params:,}")
        for kk, v in out.items():
            print(f"    {kk}: {tuple(v.shape)}  range=[{v.min():.3f}, {v.max():.3f}]")


if __name__ == "__main__":
    main()
