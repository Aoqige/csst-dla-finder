"""Trainable feature-level fusion of a CNN (dilated) tower and a Transformer tower.

Motivation
----------
``DualTowerFusionNet`` joins two fully-convolutional towers (dilated + wzx).
This module replaces one tower with a conv-stem Transformer encoder so the
fusion combines a *local* (CNN) view and a *global attention* (Transformer)
view of the same spectrum.

The two towers are deliberately fed different views:

  * CNN tower      -> the dataset channels (``flux`` or ``flux_aug``, i.e. 6 or
                      8 channels).  With ``flux_aug`` it also sees ``resid`` and
                      ``grad``, the high-frequency views the failure-sample
                      analysis showed the models currently miss.
  * Transformer tower -> the first ``in_channels`` channels only (the original
                      six ``flux`` channels), so it keeps the exact input
                      distribution its checkpoint was trained on.

``flux_aug`` is laid out as ``[flux, smooth, zq, snr, wave_norm, blue_mask,
resid, grad]`` so ``hybrid[:, :6]`` reproduces ``input_mode="flux"`` exactly.

Merge modes
-----------
``residual_transformer`` (default) starts from the transformer tower's native
heads and learns a correction informed by the CNN tower -- epoch 0 therefore
equals the transformer baseline exactly (a free control).  ``residual_cnn``
mirrors the existing SOTA recipe (CNN heads + correction).  ``plain`` learns
all heads from the fused features.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class FusionResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class CTFusionNet(nn.Module):
    """Fuse a dilated-CNN tower with a conv-stem Transformer tower."""

    def __init__(
        self,
        cnn_backbone: nn.Module,
        transformer_backbone: nn.Module,
        merge_mode: Literal["plain", "residual_transformer", "residual_cnn"] = "residual_transformer",
        width: int = 128,
        depth: int = 3,
        freeze_backbones: bool = True,
        freeze_cnn: bool | None = None,
        freeze_transformer: bool | None = None,
    ):
        super().__init__()
        if merge_mode not in {"plain", "residual_transformer", "residual_cnn"}:
            raise ValueError(f"unknown merge_mode={merge_mode!r}")
        self.cnn_backbone = cnn_backbone
        self.transformer_backbone = transformer_backbone
        self.merge_mode = merge_mode
        self.freeze_backbones = bool(freeze_backbones)
        # Per-tower freeze control: default follows freeze_backbones, but each
        # tower can be overridden (used by the end-to-end heterogeneous run,
        # which keeps the pretrained CNN frozen while training a fresh
        # attention tower that has no pretrained weights).
        self.freeze_cnn = self.freeze_backbones if freeze_cnn is None else bool(freeze_cnn)
        self.freeze_transformer = (
            self.freeze_backbones if freeze_transformer is None else bool(freeze_transformer)
        )

        cnn_channels = int(self.cnn_backbone.refine[0].out_channels)
        trans_channels = int(self.transformer_backbone.d_model)
        self.transformer_in_channels = int(getattr(self.transformer_backbone, "in_channels", 6))

        self.fuse = nn.Sequential(
            nn.Conv1d(cnn_channels + trans_channels, width, kernel_size=5, padding=2),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            *[FusionResidualBlock(width) for _ in range(depth)],
        )
        self.center_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.region_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.lognhi_delta = nn.Conv1d(width, 1, kernel_size=1)
        self.offset_delta = nn.Conv1d(width, 1, kernel_size=3, padding=1)
        self.count_delta = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(),
            nn.Linear(width, 3),
        )
        for parameter in self.cnn_backbone.parameters():
            parameter.requires_grad_(not self.freeze_cnn)
        for parameter in self.transformer_backbone.parameters():
            parameter.requires_grad_(not self.freeze_transformer)
        if merge_mode != "plain":
            for head in (self.center_delta, self.region_delta, self.lognhi_delta, self.offset_delta):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            nn.init.zeros_(self.count_delta[-1].weight)
            nn.init.zeros_(self.count_delta[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_cnn:
            self.cnn_backbone.eval()
        if self.freeze_transformer:
            self.transformer_backbone.eval()
        return self

    def _cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.cnn_backbone
        original_length = x.shape[-1]
        h = backbone.stem(x)
        skips = []
        for idx, stage in enumerate(backbone.stages):
            h = stage(h)
            if backbone.use_skip and idx < len(backbone.stages) - 1:
                skips.append(backbone.skip_projections[idx](h))
        h = backbone.upsample1(h)
        if backbone.use_skip and skips:
            h = torch.cat([h] + skips, dim=1)
        if h.shape[-1] != original_length:
            h = F.interpolate(h, size=original_length, mode="linear", align_corners=False)
        return backbone.refine(h)

    def _transformer_sequence(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.transformer_backbone
        if hasattr(backbone, "conv_stem"):
            h = backbone.conv_stem(x)
        else:
            # Pure-attention tower (v0 style): per-pixel linear projection, no
            # conv stem -- this is what makes the tower genuinely heterogeneous
            # w.r.t. the CNN tower (no built-in local receptive field).
            proj = getattr(backbone, "proj", None)
            if proj is None:
                raise ValueError("transformer backbone exposes neither conv_stem nor proj")
            h = proj(x.transpose(1, 2))
        length = h.shape[1]
        if hasattr(backbone, "pos_emb"):
            h = h + backbone.pos_emb[:length].unsqueeze(0)
        return backbone.encoder(h)

    def _cnn_base(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = self.cnn_backbone
        return {
            "center_logits": backbone.heat_head(features).squeeze(1),
            "region_logits": backbone.region_head(features).squeeze(1),
            "lognhi_raw": 0.2 + 1.5 * backbone.lognhi_head(features).squeeze(1),
            "offset_raw": torch.tanh(backbone.offset_head(features).squeeze(1)),
            "count_logits": backbone.count_head(features.mean(dim=-1)),
        }

    def _transformer_base(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone = self.transformer_backbone
        return {
            "center_logits": backbone.center_head(sequence).squeeze(-1),
            "region_logits": backbone.region_head(sequence).squeeze(-1),
            "lognhi_raw": backbone.lognhi_head(sequence).squeeze(-1),
            "offset_raw": torch.tanh(backbone.offset_head(sequence).squeeze(-1)),
            "count_logits": backbone.count_head(sequence.mean(dim=1)),
        }

    def forward(self, hybrid: torch.Tensor) -> dict[str, torch.Tensor]:
        transformer_input = hybrid[:, : self.transformer_in_channels]
        if self.freeze_cnn:
            with torch.no_grad():
                cnn_features = self._cnn_features(hybrid)
        else:
            cnn_features = self._cnn_features(hybrid)
        if self.freeze_transformer:
            with torch.no_grad():
                transformer_sequence = self._transformer_sequence(transformer_input)
        else:
            transformer_sequence = self._transformer_sequence(transformer_input)

        transformer_features = transformer_sequence.transpose(1, 2)
        shared = self.fuse(torch.cat([cnn_features, transformer_features], dim=1))
        center_delta = self.center_delta(shared).squeeze(1)
        region_delta = self.region_delta(shared).squeeze(1)
        lognhi_delta = self.lognhi_delta(shared).squeeze(1)
        offset_delta = self.offset_delta(shared).squeeze(1)
        pooled = torch.cat(
            [
                F.adaptive_avg_pool1d(shared, 1).flatten(1),
                F.adaptive_max_pool1d(shared, 1).flatten(1),
            ],
            dim=1,
        )
        count_delta = self.count_delta(pooled)

        if self.merge_mode == "plain":
            return {
                "center_logits": center_delta,
                "region_logits": region_delta,
                "lognhi_raw": lognhi_delta,
                "offset_raw": torch.tanh(offset_delta),
                "count_logits": count_delta,
            }
        if self.merge_mode == "residual_transformer":
            base = self._transformer_base(transformer_sequence)
        else:
            base = self._cnn_base(cnn_features)
        return {
            "center_logits": base["center_logits"] + center_delta,
            "region_logits": base["region_logits"] + region_delta,
            "lognhi_raw": base["lognhi_raw"] + lognhi_delta,
            "offset_raw": torch.clamp(base["offset_raw"] + 0.5 * torch.tanh(offset_delta), -1.0, 1.0),
            "count_logits": base["count_logits"] + count_delta,
        }
