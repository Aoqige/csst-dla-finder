from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class LossWeights:
    count: float = 1.0
    heatmap: float = 1.0
    region: float = 0.25
    lognhi: float = 0.25
    offset: float = 0.2
    heatmap_positive_weight: float = 10.0
    region_positive_weight: float = 2.0


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, min_weight: float) -> torch.Tensor:
    mask = weight > min_weight
    if not mask.any():
        return pred.new_tensor(0.0)
    raw = F.smooth_l1_loss(pred[mask], target[mask], reduction="none")
    weights = weight[mask]
    return (raw * weights).sum() / weights.sum().clamp_min(1e-6)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    numerator = 2.0 * (pred * target).sum(dim=1)
    denominator = pred.sum(dim=1) + target.sum(dim=1) + eps
    return (1.0 - numerator / denominator).mean()


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    count_criterion: nn.Module,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    count_loss = count_criterion(outputs["count_logits"], batch["n_dla"])

    heat_weight = 1.0 + weights.heatmap_positive_weight * batch["heatmap"]
    heat_raw = F.binary_cross_entropy_with_logits(outputs["heatmap_logits"], batch["heatmap"], reduction="none")
    heatmap_loss = (heat_raw * heat_weight).mean()

    region_weight = 1.0 + weights.region_positive_weight * batch["region"]
    region_raw = F.binary_cross_entropy_with_logits(outputs["region_logits"], batch["region"], reduction="none")
    region_loss = (region_raw * region_weight).mean() + soft_dice_loss(outputs["region_logits"], batch["region"])

    lognhi_loss = weighted_smooth_l1(
        outputs["lognhi"],
        batch["lognhi_map"],
        batch["lognhi_weight"],
        min_weight=0.05,
    )
    offset_loss = weighted_smooth_l1(
        outputs["offset"],
        batch["offset"],
        batch["offset_weight"],
        min_weight=0.5,
    )

    total = (
        weights.count * count_loss
        + weights.heatmap * heatmap_loss
        + weights.region * region_loss
        + weights.lognhi * lognhi_loss
        + weights.offset * offset_loss
    )
    parts = {
        "count_loss": float(count_loss.detach().cpu()),
        "heatmap_loss": float(heatmap_loss.detach().cpu()),
        "region_loss": float(region_loss.detach().cpu()),
        "lognhi_loss": float(lognhi_loss.detach().cpu()),
        "offset_loss": float(offset_loss.detach().cpu()),
        "loss": float(total.detach().cpu()),
    }
    return total, parts
