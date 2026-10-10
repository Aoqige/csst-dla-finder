from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DilatedResidualBlock1D(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        padding = dilation * 3
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=5, padding=dilation * 2, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ConvHead(nn.Module):
    def __init__(self, channels: int, hidden: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DLAMultiHeadNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        base_channels: int = 96,
        num_blocks: int = 8,
        dropout: float = 0.1,
        lognhi_min: float = 19.0,
        lognhi_max: float = 23.0,
        use_context_channels: bool = True,
    ):
        super().__init__()
        self.lognhi_min = float(lognhi_min)
        self.lognhi_max = float(lognhi_max)
        self.use_context_channels = bool(use_context_channels)
        stem_input_channels = input_channels + 2 if self.use_context_channels else input_channels

        self.stem = nn.Sequential(
            nn.Conv1d(stem_input_channels, base_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.GELU(),
        )
        dilations = [1, 2, 4, 8]
        blocks = []
        for block_idx in range(num_blocks):
            blocks.append(
                DilatedResidualBlock1D(
                    base_channels,
                    dilation=dilations[block_idx % len(dilations)],
                    dropout=dropout * 0.35,
                )
            )
        self.backbone = nn.Sequential(*blocks)

        self.heatmap_head = ConvHead(base_channels, base_channels // 2, 1, dropout=dropout * 0.25)
        self.region_head = ConvHead(base_channels, base_channels // 2, 1, dropout=dropout * 0.25)
        self.offset_head = ConvHead(base_channels, base_channels // 2, 1, dropout=dropout * 0.25)
        self.lognhi_head = ConvHead(base_channels, base_channels // 2, 1, dropout=dropout * 0.25)

        self.count_head = nn.Sequential(
            nn.Linear(base_channels * 2 + 1, base_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, 3),
        )

    def _append_context_channels(self, spectrum: torch.Tensor, z_qso: torch.Tensor) -> torch.Tensor:
        if not self.use_context_channels:
            return spectrum
        batch_size, _channels, length = spectrum.shape
        position = torch.linspace(
            -1.0,
            1.0,
            length,
            device=spectrum.device,
            dtype=spectrum.dtype,
        ).view(1, 1, length).expand(batch_size, 1, length)
        if z_qso.ndim == 1:
            z_context = z_qso.view(batch_size, 1, 1)
        else:
            z_context = z_qso.view(batch_size, -1)[:, :1].view(batch_size, 1, 1)
        z_context = ((z_context.to(dtype=spectrum.dtype) - 1.7) / 0.7).expand(batch_size, 1, length)
        return torch.cat([spectrum, position, z_context], dim=1)

    def forward(self, spectrum: torch.Tensor, z_qso: torch.Tensor) -> dict[str, torch.Tensor]:
        model_input = self._append_context_channels(spectrum, z_qso)
        features = self.backbone(self.stem(model_input))
        avg_features = F.adaptive_avg_pool1d(features, 1).flatten(1)
        max_features = F.adaptive_max_pool1d(features, 1).flatten(1)
        if z_qso.ndim == 1:
            z_qso = z_qso.unsqueeze(1)
        count_logits = self.count_head(torch.cat([avg_features, max_features, z_qso], dim=1))

        heatmap_logits = self.heatmap_head(features).squeeze(1)
        region_logits = self.region_head(features).squeeze(1)
        offset = 0.5 * torch.tanh(self.offset_head(features).squeeze(1))
        lognhi_raw = self.lognhi_head(features).squeeze(1)
        lognhi = self.lognhi_min + (self.lognhi_max - self.lognhi_min) * torch.sigmoid(lognhi_raw)
        return {
            "count_logits": count_logits,
            "heatmap_logits": heatmap_logits,
            "region_logits": region_logits,
            "offset": offset,
            "lognhi": lognhi,
        }


def create_model(
    input_channels: int,
    base_channels: int = 96,
    num_blocks: int = 8,
    dropout: float = 0.1,
    lognhi_min: float = 19.0,
    lognhi_max: float = 23.0,
    use_context_channels: bool = True,
) -> DLAMultiHeadNet:
    return DLAMultiHeadNet(
        input_channels=input_channels,
        base_channels=base_channels,
        num_blocks=num_blocks,
        dropout=dropout,
        lognhi_min=lognhi_min,
        lognhi_max=lognhi_max,
        use_context_channels=use_context_channels,
    )
