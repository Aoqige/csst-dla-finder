#!/usr/bin/env python3
"""Train one minimal Transformer five-head DLA model.

This is a sibling of ``train_hybrid.py``: it reuses the same data pipeline,
targets, losses, validation decode, and scoring, and only swaps the backbone
for the per-pixel Transformer in ``models/transformer_5head.py``.

Default input_mode="flux" (6 channels, no FLUX_CLEAN) so results are directly
comparable to the dilated_flux CNN baseline (Final 0.3625 on test).
"""
from __future__ import annotations

import math
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from csst_dla.scoring import labels_to_truth, score_catalog
from data import HybridTrainDataset
from decode import average_predictions, decode_validation_catalog, predict_member
from models.transformer_5head import _build_transformer_5head
from models.transformer_conv_stem_5head import _build_transformer_conv_stem_5head
from models.transformer_conv_stem_rope_5head import _build_transformer_conv_stem_rope_5head
from models.transformer_conv_stem_alibi_5head import _build_transformer_conv_stem_alibi_5head


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def focal_bce_with_logits(logits, target, alpha=0.85, gamma=2.0, weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * bce
    if weight is not None:
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def balanced_class_weights(counts: np.ndarray, power: float) -> list[float]:
    class_counts = np.bincount(counts, minlength=3).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("count head requires all three classes in the training split")
    weights = (len(counts) / (3.0 * class_counts)) ** power
    return (weights / weights.mean()).astype(np.float32).tolist()


def focal_cross_entropy_with_logits(logits, target, gamma=2.0, weight=None):
    """Multi-class focal cross entropy (Lin et al. 2017).

    Rationale for this project: plain weighted CE is bounded below by the label
    entropy H(p), so the count head can never release the ~73% of the loss
    budget it takes at initialisation; it is locked in by an information-theoretic
    floor, not by an hyper-parameter choice.  The focusing factor is the only
    thing that removes that floor.

    ``gamma=0`` reproduces ``nn.functional.cross_entropy`` EXACTLY, including the
    ``weight`` normalisation (PyTorch divides by the summed weight of the observed
    targets, not by the batch size) -- that identity is what
    ``staging/headprobe/test_count_loss_fix.py`` asserts.
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

    ``head_losses`` holds the UNWEIGHTED per-head means returned by
    ``train_one_epoch``, so multiplying by the loss weights here is what makes the
    reported share comparable with the scalar that is actually back-propagated.
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


def input_channels(input_mode: str) -> int:
    table = {
        "raw": 1,
        "flux": 6,
        "flux_sig": 7,
        "flux_aug": 8,
        "flux_feat": 9,
        "residual": 7,
        "all": 8,
        "all_wzx": 11,
    }
    if input_mode not in table:
        raise ValueError(f"unknown input_mode: {input_mode}")
    return table[input_mode]


def train_one_epoch(model, loader, opt, device, args, count_class_weights=None, epoch=0, scheduler=None):
    model.train()
    total = 0.0
    # Per-head supervision accounting.  ``total`` keeps its original formula
    # (weighted batch loss x batch size, divided by the full split size), so the
    # "loss" logged by every archived run stays reproducible; the per-head sums
    # below are a pure diagnostic and never touch the autograd graph.
    head_sum = {k: 0.0 for k in HEAD_KEYS}
    progress_every = max(1, len(loader) // 5)
    for batch_idx, (x, center, region, lognhi, mask, offset, offset_weight, count, _) in enumerate(loader):
        x = x.to(device)
        center = center.to(device)
        region = region.to(device)
        lognhi = lognhi.to(device)
        mask = mask.to(device)
        offset = offset.to(device)
        offset_weight = offset_weight.to(device)
        count = count.to(device)

        out = model(x)
        high_mask = ((lognhi >= args.high_lognhi_threshold) & (mask > 0)).float()
        center_weight = 1.0 + (args.high_lognhi_center_weight - 1.0) * high_mask
        center_loss = focal_bce_with_logits(out["center_logits"], center, weight=center_weight)
        region_loss = focal_bce_with_logits(out["region_logits"], region, alpha=0.75)
        if mask.sum() > 0:
            log_weight = mask * (1.0 + (args.high_lognhi_log_weight - 1.0) * high_mask)
            log_loss = (((20.3 + out["lognhi_raw"] - lognhi) ** 2) * log_weight).sum()
            log_loss = log_loss / log_weight.sum().clamp_min(1.0)
        else:
            log_loss = torch.tensor(0.0, device=device)
        if "offset_raw" in out and offset_weight.sum() > 0:
            offset_loss = (((out["offset_raw"] - offset) ** 2) * offset_weight).sum()
            offset_loss = offset_loss / offset_weight.sum().clamp_min(1.0)
        else:
            offset_loss = torch.tensor(0.0, device=device)
        if getattr(args, "count_loss_type", "ce") == "focal":
            count_loss = focal_cross_entropy_with_logits(
                out["count_logits"], count,
                gamma=args.count_focal_gamma, weight=count_class_weights,
            )
        else:
            count_loss = nn.functional.cross_entropy(
                out["count_logits"], count, weight=count_class_weights
            )
        loss = (
            center_loss
            + args.region_loss_weight * region_loss
            + args.lognhi_loss_weight * log_loss
            + args.offset_loss_weight * offset_loss
            + args.count_loss_weight * count_loss
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.detach().cpu()) * len(x)
        with torch.no_grad():
            for key, value in zip(
                HEAD_KEYS, (center_loss, region_loss, log_loss, offset_loss, count_loss)
            ):
                head_sum[key] += float(value.detach().cpu()) * len(x)
        if batch_idx == 0 or (batch_idx + 1) % progress_every == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(
                json.dumps(
                    {
                        "progress": "train",
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "total_batches": len(loader),
                        "lr": cur_lr,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    n = len(loader.dataset)
    return total / n, {k: v / n for k, v in head_sum.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Transformer five-head DLA model.")
    parser.add_argument("--targets", required=True)
    parser.add_argument("--train-fits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--input-mode",
        choices=["raw", "flux", "flux_sig", "flux_aug", "flux_feat", "residual", "all", "all_wzx"],
        default="flux",
        help="flux=6 channels (no FLUX_CLEAN), directly comparable to dilated_flux.",
    )
    parser.add_argument(
        "--arch",
        choices=["transformer", "transformer_conv_stem", "transformer_conv_stem_rope", "transformer_conv_stem_alibi"],
        default="transformer",
        help="transformer=v0; transformer_conv_stem=v1+; transformer_conv_stem_rope=v5 RoPE relative-position.",
    )
    # Transformer hyper-parameters
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-ff", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1024)
    # Conv stem (only used when arch=transformer_conv_stem)
    parser.add_argument("--conv-kernel", type=int, default=7,
                        help="Conv1d stem kernel size (must be odd). 7 -> +/-3 px context.")
    parser.add_argument("--num-conv-layers", type=int, default=1,
                        help="Number of conv layers in the stem. 1 = minimal local bias.")
    # Optimisation
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # LR scheduler (v3b experiments)
    parser.add_argument("--lr-schedule", choices=["none", "cosine"], default="none",
                        help="none=constant lr; cosine=warmup then cosine decay to 0.")
    parser.add_argument("--lr-warmup-epochs", type=float, default=1.0,
                        help="Linear warmup length (in epochs). Only used when --lr-schedule=cosine.")
    # Loss / decode
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--min-z-dla", type=float, default=1.1)
    parser.add_argument("--region-loss-weight", type=float, default=0.2)
    parser.add_argument("--lognhi-loss-weight", type=float, default=0.05)
    parser.add_argument("--offset-loss-weight", type=float, default=0.1)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--count-class-weight-power", type=float, default=0.0)
    # Count-head loss shape.  Plain weighted CE is floored by the label entropy
    # H(p)=0.55646, and that floor is what locks ~73% (at init) -> 91-96%
    # (converged) of the whole supervision budget into this single head; focal
    # CE is the only change that removes the floor.  Default "ce" keeps every
    # archived run reproducible.
    parser.add_argument("--count-loss-type", choices=["ce", "focal"], default="ce",
                        help="ce=weighted cross entropy (original, reproduces v3c); "
                             "focal=multi-class focal CE (gamma=0 reproduces ce).")
    parser.add_argument("--count-focal-gamma", type=float, default=2.0,
                        help="Focusing parameter for --count-loss-type=focal.")
    parser.add_argument("--truth-min-lognhi", type=float, default=20.3)
    parser.add_argument("--high-lognhi-threshold", type=float, default=22.0)
    parser.add_argument("--high-lognhi-center-weight", type=float, default=4.0)
    parser.add_argument("--high-lognhi-log-weight", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--allow-legacy-input-modes",
        action="store_true",
        help="Permit flux_aug / residual / all / all_wzx. Only for reproducing archived runs.",
    )
    args = parser.parse_args()

    if args.input_mode == "flux_aug" and not args.allow_legacy_input_modes:
        raise SystemExit(
            "--input-mode flux_aug is disabled for transformer towers.\n"
            "  resid = (I - B15) flux and grad = D flux are EXACT LINEAR functions of the flux\n"
            "  channel, so the first convolution reproduces them at zero cost; their marginal\n"
            "  information is I(Y; resid, grad | flux) = 0.  Measured: +4.7 pp on the weak CNN,\n"
            "  but -1.5 pp (v3c) and -3.5 pp (SOTA) on transformers.\n"
            "  Use --input-mode flux (6ch), flux_sig (7ch) or flux_feat (9ch) instead.\n"
            "  Pass --allow-legacy-input-modes only to reproduce an archived run."
        )
    if args.input_mode in {"residual", "all", "all_wzx"} and not args.allow_legacy_input_modes:
        raise SystemExit(
            f"--input-mode {args.input_mode} consumes FLUX_CLEAN, which the challenge bans:\n"
            "  its 0.8333 clean reference is an upper bound, not a legal score.\n"
            "  Pass --allow-legacy-input-modes if you only want that reference."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_ch = input_channels(args.input_mode)
    train_ds = HybridTrainDataset(
        args.targets, args.train_fits, "train", args.input_mode,
        max_samples=args.max_train_samples, cache_channels=True,
    )
    val_ds = HybridTrainDataset(
        args.targets, args.train_fits, "val", args.input_mode,
        max_samples=args.max_val_samples, cache_channels=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
    )
    device = resolve_device(args.device)

    train_counts = np.asarray(train_ds.count, dtype=np.int64)
    class_counts = np.bincount(train_counts, minlength=3).tolist()
    if args.count_class_weight_power > 0.0:
        class_weights = balanced_class_weights(train_counts, args.count_class_weight_power)
    else:
        class_weights = [1.0, 1.0, 1.0]
    count_class_weights = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if args.count_class_weight_power > 0.0
        else None
    )
    print(json.dumps({
        "stage": "datasets_ready",
        "train": len(train_ds), "val": len(val_ds),
        "train_count_distribution": dict(enumerate(class_counts)),
        "count_class_weights": class_weights,
        "in_channels": in_ch,
        "truth_min_lognhi": args.truth_min_lognhi,
        "target_min_lognhi": val_ds.target_min_lognhi,
    }, ensure_ascii=False), flush=True)

    if args.arch == "transformer_conv_stem":
        model = _build_transformer_conv_stem_5head(
            in_channels=in_ch,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            conv_kernel=args.conv_kernel,
            num_conv_layers=args.num_conv_layers,
            use_offset=True,
            max_len=args.max_len,
        ).to(device)
    elif args.arch == "transformer_conv_stem_rope":
        model = _build_transformer_conv_stem_rope_5head(
            in_channels=in_ch,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            conv_kernel=args.conv_kernel,
            num_conv_layers=args.num_conv_layers,
            use_offset=True,
            max_len=args.max_len,
        ).to(device)
    elif args.arch == "transformer_conv_stem_alibi":
        model = _build_transformer_conv_stem_alibi_5head(
            in_channels=in_ch,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            conv_kernel=args.conv_kernel,
            num_conv_layers=args.num_conv_layers,
            use_offset=True,
            max_len=args.max_len,
        ).to(device)
    else:
        model = _build_transformer_5head(
            in_channels=in_ch,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_ff=args.dim_ff,
            dropout=args.dropout,
            use_offset=True,
            max_len=args.max_len,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"stage": "model_built", "arch": args.arch, "params": n_params,
                        "d_model": args.d_model, "nhead": args.nhead,
                        "num_layers": args.num_layers, "dim_ff": args.dim_ff,
                        "dropout": args.dropout,
                        "conv_kernel": getattr(args, "conv_kernel", None),
                        "num_conv_layers": getattr(args, "num_conv_layers", None)}), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_schedule == "cosine":
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        warmup_steps = int(args.lr_warmup_epochs * steps_per_epoch)
        warmup_steps = min(warmup_steps, max(1, total_steps - 1))
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        print(json.dumps({"stage": "scheduler_built", "schedule": args.lr_schedule,
                            "total_steps": total_steps, "warmup_steps": warmup_steps,
                            "base_lr": args.lr}), flush=True)

    model_config = {
        "arch": args.arch,
        "in_channels": in_ch,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
        "n_count_classes": 3,
        "use_offset": True,
        "max_len": args.max_len,
        "input_mode": args.input_mode,
        "conv_kernel": getattr(args, "conv_kernel", None),
        "num_conv_layers": getattr(args, "num_conv_layers", None),
        "lr_schedule": args.lr_schedule,
        "lr_warmup_epochs": args.lr_warmup_epochs,
    }

    history = []
    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        loss, head_losses = train_one_epoch(model, train_loader, opt, device, args,
                                            count_class_weights, epoch=epoch, scheduler=scheduler)
        pred = predict_member(model, val_loader, device)
        avg = average_predictions([pred])
        catalog = decode_validation_catalog(
            avg, val_ds.indices, val_ds.labels, val_ds.wavelength,
            threshold=args.threshold, min_z_dla=args.min_z_dla,
            lognhi_min=args.truth_min_lognhi,
        )
        truth = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=args.truth_min_lognhi)
        score = score_catalog(truth, catalog)
        row = {
            "epoch": epoch,
            "loss": loss,
            "head_losses": head_losses,
            "head_budget": head_budget(head_losses, args),
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
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if score.final_score > best_score:
            best_score = score.final_score
            torch.save({
                "model_state": model.state_dict(),
                "config": model_config,
                "threshold": args.threshold,
                "score": row,
                "training_config": vars(args),
            }, out_dir / "best_model.pt")

    torch.save({
        "model_state": model.state_dict(),
        "config": model_config,
        "threshold": args.threshold,
        "score": history[-1],
        "training_config": vars(args),
    }, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "training_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'best_model.pt'}  best_val_final={best_score:.4f}")


if __name__ == "__main__":
    main()
