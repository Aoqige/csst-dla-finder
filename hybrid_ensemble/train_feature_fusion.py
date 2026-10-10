#!/usr/bin/env python3
"""Train one feature-level, dual-backbone DLA fusion model."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))
sys.path.insert(0, str(ROOT / "vendor"))

from csst_dla.scoring import labels_to_truth, score_catalog
from decode import decode_validation_catalog
from evaluate_hybrid import load_checkpoint, resolve_device
from feature_fusion import DualFusionTrainDataset, DualTowerFusionNet
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint


def focal_bce_with_logits(logits, target, alpha=0.85, gamma=2.0, weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def focal_cross_entropy_with_logits(logits, target, gamma=2.0, weight=None):
    """Multi-class focal cross entropy (Lin et al. 2017).

    Rationale for this project: plain weighted CE is bounded below by the label
    entropy H(p), so the count head can never release the largest share of the
    loss budget it takes at initialisation; it is locked in by an
    information-theoretic floor, not by a hyper-parameter choice.  The focusing
    factor is the only thing that removes that floor.

    ``gamma=0`` reproduces ``nn.functional.cross_entropy`` EXACTLY, including the
    ``weight`` normalisation (PyTorch divides by the summed weight of the observed
    targets, not by the batch size) -- that identity is what
    ``staging/headprobe/test_cnn_loss_fix.py`` asserts.
    """
    log_prob = nn.functional.log_softmax(logits, dim=-1)
    log_pt = log_prob.gather(1, target.view(-1, 1)).squeeze(1)
    pt = log_pt.exp().clamp(0.0, 1.0)
    loss = -(1.0 - pt).pow(gamma) * log_pt
    if weight is not None:
        w = weight.gather(0, target.view(-1))
        return (loss * w).sum() / w.sum().clamp_min(1e-12)
    return loss.mean()


HEAD_KEYS = ("center", "region", "lognhi", "offset", "count")


def head_budget(head_losses, args):
    """Weighted contribution and normalised share of every head's loss.

    ``head_losses`` holds the UNWEIGHTED per-head means returned by the training
    loop, so multiplying by the loss weights here is what makes the reported
    share comparable with the scalar that is actually back-propagated.
    """
    weights = {
        "center": 1.0,
        "region": args.region_loss_weight,
        "lognhi": args.lognhi_loss_weight,
        "offset": args.offset_loss_weight,
        "count": args.count_loss_weight,
    }
    weighted = {k: weights[k] * head_losses[k] for k in HEAD_KEYS}
    total = sum(weighted.values())
    share = {k: (weighted[k] / total if total > 0.0 else 0.0) for k in HEAD_KEYS}
    return {"raw": dict(head_losses), "weight": weights,
            "weighted": weighted, "share": share, "total": total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--train-fits", required=True)
    parser.add_argument("--dilated-checkpoint", required=True)
    parser.add_argument("--wzx-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--merge-mode", choices=["plain", "residual_dilated", "residual_wzx"], required=True)
    parser.add_argument("--fusion-width", type=int, default=128)
    parser.add_argument("--fusion-depth", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-z-dla", type=float, default=1.10)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--lognhi-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-loss-weight", type=float, default=0.1)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--count-loss-type", choices=["ce", "focal"], default="ce",
                        help="ce=plain cross entropy (original, reproduces the "
                             "feature_no_clean_flux_context_v1 recipe); "
                             "focal=multi-class focal CE (gamma=0 reproduces ce).")
    parser.add_argument("--count-focal-gamma", type=float, default=2.0,
                        help="Focusing parameter for --count-loss-type=focal.")
    parser.add_argument("--high-lognhi-threshold", type=float, default=22.0)
    parser.add_argument("--high-lognhi-center-weight", type=float, default=4.0)
    parser.add_argument("--high-lognhi-log-weight", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--train-backbones", action="store_true", help="Unfreeze both pretrained towers; off by default.")
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Separate AdamW learning rate for the two pretrained towers when "
                             "--train-backbones is set.  Default None keeps the legacy single-group "
                             "behaviour (one LR for everything), which applies --lr to the towers too.")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="With --train-backbones: train the towers' WEIGHTS but keep their "
                             "BatchNorm running statistics and Dropout frozen in eval mode.  "
                             "Removes two of the three simultaneous changes the raw flag makes.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def loss_for_batch(model, batch, device, args):
    hybrid, wzx, zq, center, region, lognhi, mask, offset, offset_weight, count, _ = batch
    hybrid = hybrid.to(device, non_blocking=True)
    wzx = wzx.to(device, non_blocking=True)
    zq = zq.to(device, non_blocking=True)
    center = center.to(device, non_blocking=True)
    region = region.to(device, non_blocking=True)
    lognhi = lognhi.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    offset = offset.to(device, non_blocking=True)
    offset_weight = offset_weight.to(device, non_blocking=True)
    count = count.to(device, non_blocking=True)
    output = model(hybrid, wzx, zq)
    high_mask = ((lognhi >= args.high_lognhi_threshold) & (mask > 0)).float()
    center_weight = 1.0 + (args.high_lognhi_center_weight - 1.0) * high_mask
    center_loss = focal_bce_with_logits(output["center_logits"], center, weight=center_weight)
    region_loss = focal_bce_with_logits(output["region_logits"], region, alpha=0.75)
    if mask.sum() > 0:
        log_weight = mask * (1.0 + (args.high_lognhi_log_weight - 1.0) * high_mask)
        log_loss = (((20.3 + output["lognhi_raw"] - lognhi) ** 2) * log_weight).sum() / log_weight.sum().clamp_min(1.0)
    else:
        log_loss = torch.zeros((), device=device)
    if offset_weight.sum() > 0:
        offset_loss = (((output["offset_raw"] - offset) ** 2) * offset_weight).sum() / offset_weight.sum().clamp_min(1.0)
    else:
        offset_loss = torch.zeros((), device=device)
    if getattr(args, "count_loss_type", "ce") == "focal":
        count_loss = focal_cross_entropy_with_logits(
            output["count_logits"], count, gamma=args.count_focal_gamma
        )
    else:
        count_loss = nn.functional.cross_entropy(output["count_logits"], count)
    loss = (
        center_loss
        + args.region_loss_weight * region_loss
        + args.lognhi_loss_weight * log_loss
        + args.offset_loss_weight * offset_loss
        + args.count_loss_weight * count_loss
    )
    # Pure diagnostic: the per-head values are detached scalars, so recording
    # them cannot touch the autograd graph or change the optimised objective.
    head_losses = {}
    with torch.no_grad():
        for key, value in zip(
            HEAD_KEYS, (center_loss, region_loss, log_loss, offset_loss, count_loss)
        ):
            head_losses[key] = float(value.detach().cpu())
    return loss, output, head_losses


@torch.inference_mode()
def predict_validation(model, loader, device):
    model.eval()
    heat, lognhi, offset, count_logits, rows = [], [], [], [], []
    for batch in loader:
        hybrid, wzx, zq = batch[:3]
        output = model(hybrid.to(device), wzx.to(device), zq.to(device))
        heat.append(torch.sigmoid(output["center_logits"]).cpu().numpy())
        lognhi.append((20.3 + output["lognhi_raw"]).cpu().numpy())
        offset.append(output["offset_raw"].cpu().numpy())
        count_logits.append(output["count_logits"].cpu().numpy())
        rows.append(np.asarray(batch[-1], dtype=np.int64))
    return {
        "heatmap": np.concatenate(heat),
        "lognhi": np.concatenate(lognhi),
        "offset": np.concatenate(offset),
        "count_logits": np.concatenate(count_logits),
        "rows": np.concatenate(rows),
    }


def save_checkpoint(path: Path, model, args, epoch_row: dict, dilated_config: dict, wzx_config: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "merge_mode": args.merge_mode,
                "fusion_width": args.fusion_width,
                "fusion_depth": args.fusion_depth,
                "freeze_backbones": not args.train_backbones,
                "dilated_checkpoint": args.dilated_checkpoint,
                "wzx_checkpoint": args.wzx_checkpoint,
                "dilated_config": dilated_config,
                "wzx_config": wzx_config,
                "dilated_input_mode": args.dilated_input_mode,
                "wzx_feature_mode": args.wzx_feature_mode,
            },
            "threshold": args.threshold,
            "score": epoch_row,
            "training_config": vars(args),
        },
        path,
    )


def score_validation(model, val_loader, val_ds, device, args, epoch: int, loss: float | None = None) -> dict:
    prediction = predict_validation(model, val_loader, device)
    catalog = decode_validation_catalog(
        prediction,
        val_ds.indices,
        val_ds.labels,
        val_ds.wavelength,
        threshold=args.threshold,
        min_distance=10,
        min_z_dla=args.min_z_dla,
        lognhi_min=20.3,
        lognhi_max=22.5,
    )
    truth = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=20.3)
    score = score_catalog(truth, catalog)
    return {
        "epoch": epoch,
        "loss": loss,
        "final_score": score.final_score,
        "detection_score": score.detection_score,
        "parameter_score": score.parameter_score,
        "completeness": score.completeness,
        "purity": score.purity,
        "n_pred": score.n_pred,
        "n_match": score.n_match,
        "std_dv": score.std_dv,
        "std_dlognhi": score.std_dlognhi,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dilated_wrapper, dilated_config = load_checkpoint(args.dilated_checkpoint, device)
    wzx_model, wzx_config = load_model_from_checkpoint(args.wzx_checkpoint, device)
    args.dilated_input_mode = str(dilated_config.get("input_mode", "all"))
    args.wzx_feature_mode = str(wzx_config.get("feature_mode", "all"))
    print(
        json.dumps(
            {
                "stage": "loading_datasets",
                "merge_mode": args.merge_mode,
                "dilated_input_mode": args.dilated_input_mode,
                "wzx_feature_mode": args.wzx_feature_mode,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_ds = DualFusionTrainDataset(
        args.targets,
        args.train_fits,
        "train",
        args.max_train_samples,
        dilated_input_mode=args.dilated_input_mode,
        wzx_feature_mode=args.wzx_feature_mode,
    )
    val_ds = DualFusionTrainDataset(
        args.targets,
        args.train_fits,
        "val",
        args.max_val_samples,
        dilated_input_mode=args.dilated_input_mode,
        wzx_feature_mode=args.wzx_feature_mode,
    )
    print(json.dumps({"stage": "datasets_ready", "train": len(train_ds), "val": len(val_ds)}, ensure_ascii=False), flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    model = DualTowerFusionNet(
        dilated_wrapper.model,
        wzx_model,
        merge_mode=args.merge_mode,
        width=args.fusion_width,
        depth=args.fusion_depth,
        freeze_backbones=not args.train_backbones,
        freeze_backbone_norm=bool(args.freeze_bn),
    ).to(device)
    if args.train_backbones and args.backbone_lr is not None:
        # Discriminative LRs: a pretrained tower normally needs a rate one to two
        # decades below the freshly initialised head, so a single global LR would
        # make "unfreeze hurts" indistinguishable from "unfreeze LR was too high".
        tower_parameters = list(model.dilated_backbone.parameters()) + list(model.wzx_backbone.parameters())
        tower_ids = {id(p) for p in tower_parameters}
        head_parameters = [
            p for p in model.parameters() if p.requires_grad and id(p) not in tower_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": tower_parameters, "lr": float(args.backbone_lr)},
                {"params": head_parameters, "lr": float(args.lr)},
            ],
            lr=float(args.lr),
            weight_decay=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4)
    print(
        json.dumps(
            {
                "stage": "unfreeze_protocol",
                "train_backbones": bool(args.train_backbones),
                "freeze_bn": bool(args.freeze_bn),
                "backbone_lr": args.backbone_lr,
                "lr": args.lr,
                "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "n_total": int(sum(p.numel() for p in model.parameters())),
                "optimizer_groups": [
                    {"lr": g["lr"], "n_params": int(sum(p.numel() for p in g["params"]))}
                    for g in optimizer.param_groups
                ],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    history = []
    best_score = -float("inf")
    # Residual modes are intentionally initialized as a source model plus a
    # zero correction.  Record this epoch-0 control so training cannot hide a
    # regression by overwriting the usable base checkpoint.
    if args.merge_mode != "plain":
        initial = score_validation(model, val_loader, val_ds, device, args, epoch=0)
        history.append(initial)
        best_score = initial["final_score"]
        save_checkpoint(out_dir / "initial_model.pt", model, args, initial, dilated_config, wzx_config)
        print(json.dumps(initial, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        # Per-head supervision accounting.  ``total_loss`` keeps its original
        # formula (weighted batch loss x batch size, divided by the full split
        # size), so the "loss" logged by the archived runs stays reproducible;
        # the per-head sums below are a pure diagnostic and never touch the
        # optimised objective.
        head_sum = {k: 0.0 for k in HEAD_KEYS}
        progress_every = max(1, len(train_loader) // 5)
        for batch_index, batch in enumerate(train_loader, start=1):
            loss, _, head_losses = loss_for_batch(model, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch[0])
            for key in HEAD_KEYS:
                head_sum[key] += head_losses[key] * len(batch[0])
            if batch_index == 1 or batch_index % progress_every == 0:
                print(json.dumps({"progress": "train", "epoch": epoch, "batch": batch_index, "total_batches": len(train_loader)}, ensure_ascii=False), flush=True)
        row = score_validation(model, val_loader, val_ds, device, args, epoch=epoch, loss=total_loss / len(train_ds))
        epoch_head_losses = {k: v / len(train_ds) for k, v in head_sum.items()}
        row["head_losses"] = epoch_head_losses
        row["head_budget"] = head_budget(epoch_head_losses, args)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if row["final_score"] > best_score:
            best_score = row["final_score"]
            save_checkpoint(out_dir / "best_model.pt", model, args, row, dilated_config, wzx_config)
    save_checkpoint(out_dir / "model.pt", model, args, history[-1], dilated_config, wzx_config)
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
