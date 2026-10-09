"""Minimal per-pixel Transformer encoder with the DLA five-head output contract.

Design goal (v0 - simplest):
  - no conv stem, no patching: 1 wavelength pixel == 1 token
  - per-token linear projection of the input channels
  - learned positional embedding
  - standard pre-norm TransformerEncoder (nn.TransformerEncoderLayer,
    norm_first=True, gelu)
  - the same five heads as the dilated CNN baseline so the existing
    train/decode/score pipeline works unchanged

Output contract (matches hybrid_ensemble/model.py:HybridDlaNet):
  center_logits : [B, L]   raw logits -> sigmoid = heatmap
  region_logits : [B, L]   raw logits -> sigmoid = region
  lognhi_raw    : [B, L]   logNHI = 20.3 + lognhi_raw (pipeline convention)
  offset_raw    : [B, L]   tanh-bounded sub-pixel offset
  count_logits  : [B, 3]   DLA count 0/1/2 logits
"""
from __future__ import annotations

from typing import Any


# Pipeline convention: predicted logNHI = PIPELINE_BIAS + lognhi_raw.
# Keep in sync with hybrid_ensemble/model.py and train_hybrid.py.
PIPELINE_BIAS = 20.3


def _build_transformer_5head(
    in_channels: int = 6,
    d_model: int = 192,
    nhead: int = 8,
    num_layers: int = 4,
    dim_ff: int = 768,
    dropout: float = 0.1,
    n_count_classes: int = 3,
    use_offset: bool = True,
    max_len: int = 1024,
) -> Any:
    """Construct the minimal Transformer five-head model.

    Parameters
    ----------
    in_channels:
        Number of input feature channels per pixel. For ``input_mode="flux"``
        this is 6 (flux, smooth, z_qso, snr, wave_norm, blue_mask) and is
        directly comparable to the dilated_flux CNN baseline.
    d_model / nhead / num_layers / dim_ff / dropout:
        Standard Transformer encoder hyper-parameters. With d_model=192,
        num_layers=4 the network has ~2.0M parameters, matching the CNN
        baseline (hidden=96, num_blocks=4).
    max_len:
        Upper bound on sequence length for the learned positional embedding.
        The 681-pixel grid fits comfortably.
    """
    import torch
    nn = torch.nn

    class Transformer5Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = in_channels
            self.d_model = d_model
            self.use_offset = bool(use_offset)
            self.max_len = max_len

            # Per-pixel linear embedding: [B, C, L] -> [B, L, d_model]
            self.proj = nn.Linear(in_channels, d_model)
            # Learned absolute positional embedding, sliced to the actual length.
            self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm: more stable without warmup
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                norm=nn.LayerNorm(d_model),
            )

            # Dense per-position heads keep the decode pipeline (NMS / offset
            # / count decoding) identical to the CNN baseline.
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
            # x: [B, C, L] -> [B, L, C]
            if x.ndim != 3:
                raise ValueError(f"expected [B, C, L], got shape {tuple(x.shape)}")
            x = x.transpose(1, 2)
            B, L, C = x.shape
            if L > self.max_len:
                raise ValueError(f"sequence length {L} exceeds positional buffer {self.max_len}")

            h = self.proj(x) + self.pos_emb[:L].unsqueeze(0)
            h = self.encoder(h)  # [B, L, d_model]

            center_logits = self.center_head(h).squeeze(-1)
            region_logits = self.region_head(h).squeeze(-1)
            lognhi_raw = self.lognhi_head(h).squeeze(-1)
            count_logits = self.count_head(h.mean(dim=1))  # global mean pool -> [B, 3]

            out = {
                "center_logits": center_logits,
                "region_logits": region_logits,
                "lognhi_raw": lognhi_raw,
                "count_logits": count_logits,
            }
            if self.use_offset:
                out["offset_raw"] = torch.tanh(self.offset_head(h).squeeze(-1))
            return out

    return Transformer5Head()


def main() -> None:
    """Smoke test: forward pass + parameter count for a few configs."""
    torch = __import__("torch")
    n_bins = 681
    x = torch.zeros(4, 6, n_bins)

    configs = [
        ("d192 L4 h8 (baseline)", 192, 8, 4, 768),
        ("d128 L4 h4", 128, 4, 4, 512),
        ("d256 L6 h8", 256, 8, 6, 1024),
    ]
    for name, d, h, L, ff in configs:
        model = _build_transformer_5head(
            in_channels=6, d_model=d, nhead=h, num_layers=L, dim_ff=ff,
        )
        with torch.no_grad():
            out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] params={n_params:,}")
        for k, v in out.items():
            print(f"    {k}: {tuple(v.shape)}  range=[{v.min():.3f}, {v.max():.3f}]")


if __name__ == "__main__":
    main()
