from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DLATrainDataset, TargetConfig, feature_channels, feature_requires_flux_clean
from .fits_io import load_train_fits
from .losses import LossWeights, compute_loss
from .model import create_model
from .utils import RunningAverage, ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multi-head DLA detector.")
    parser.add_argument("--train_fits", default="/Users/cham/project/Astro/train.fits")
    parser.add_argument("--output_dir", default="outputs/exp_multitask")
    parser.add_argument("--feature_mode", choices=["flux", "flux_clean", "residual", "flux_residual", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_size", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_channels", type=int, default=96)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no_context_channels",
        action="store_true",
        help="Disable the optimized wavelength-position and Z_QSO context channels.",
    )
    parser.add_argument("--sigma_bins", type=float, default=2.0)
    parser.add_argument("--region_half_width_bins", type=int, default=8)
    parser.add_argument("--region_lognhi_scale", type=float, default=2.0)
    parser.add_argument("--lambda_count", type=float, default=1.0)
    parser.add_argument("--lambda_heatmap", type=float, default=1.0)
    parser.add_argument("--lambda_region", type=float, default=0.25)
    parser.add_argument("--lambda_lognhi", type=float, default=0.25)
    parser.add_argument("--lambda_offset", type=float, default=0.2)
    parser.add_argument("--heatmap_positive_weight", type=float, default=10.0)
    parser.add_argument("--region_positive_weight", type=float, default=2.0)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    return parser.parse_args()


def make_split(labels_n_dla: np.ndarray, val_size: float, split_seed: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels_n_dla), dtype=np.int64)
    working_indices = all_indices
    if max_samples is not None and max_samples < len(all_indices):
        working_indices, _ = train_test_split(
            all_indices,
            train_size=max_samples,
            random_state=split_seed,
            stratify=labels_n_dla,
        )
        working_indices = np.asarray(sorted(working_indices), dtype=np.int64)

    if val_size <= 0:
        return np.asarray(working_indices, dtype=np.int64), np.asarray([], dtype=np.int64)
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be 0 or between 0 and 1, got {val_size}")

    train_idx, val_idx = train_test_split(
        working_indices,
        test_size=val_size,
        random_state=split_seed,
        stratify=labels_n_dla[working_indices],
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def format_n_dla_counts(n_dla: np.ndarray, indices: np.ndarray) -> str:
    counts = np.bincount(n_dla[indices], minlength=3)
    return ", ".join(f"N_DLA={label}: {int(counts[label])}" for label in range(3))


def make_loaders(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_clean: np.ndarray | None,
    labels,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader | None]:
    target_config = TargetConfig(
        sigma_bins=args.sigma_bins,
        region_half_width_bins=args.region_half_width_bins,
        region_lognhi_scale=args.region_lognhi_scale,
    )
    train_dataset = DLATrainDataset(
        wavelength,
        flux,
        flux_clean,
        labels,
        feature_mode=args.feature_mode,
        target_config=target_config,
        indices=train_idx,
    )
    device = get_device()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if len(val_idx):
        val_dataset = DLATrainDataset(
            wavelength,
            flux,
            flux_clean,
            labels,
            feature_mode=args.feature_mode,
            target_config=target_config,
            indices=val_idx,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
    return train_loader, val_loader


def class_weight_tensor(n_dla: np.ndarray, train_idx: np.ndarray, device: torch.device, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = np.bincount(n_dla[train_idx], minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    loss_weights: LossWeights,
    optimizer: torch.optim.Optimizer | None,
    show_progress: bool,
    epoch: int,
    epochs: int,
    phase: str,
    learning_rate: float | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_avg = RunningAverage()
    parts_sum: dict[str, float] = {}
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    count_correct = 0
    count_total = 0

    iterator = tqdm(
        loader,
        desc=f"epoch {epoch:03d}/{epochs} {phase}",
        leave=True,
        disable=not show_progress,
        dynamic_ncols=True,
    )
    for batch in iterator:
        batch = {key: value.to(device) for key, value in batch.items()}
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["spectrum"], batch["z_qso"])
        loss, parts = compute_loss(outputs, batch, criterion, loss_weights)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(batch["spectrum"].shape[0])
        loss_avg.update(float(loss.detach().cpu()), batch_size)
        for key, value in parts.items():
            parts_sum[key] = parts_sum.get(key, 0.0) + value * batch_size
        true_batch = batch["n_dla"].detach().cpu().numpy()
        pred_batch = torch.argmax(outputs["count_logits"], dim=1).detach().cpu().numpy()
        y_true.append(true_batch)
        y_pred.append(pred_batch)
        count_correct += int((true_batch == pred_batch).sum())
        count_total += batch_size

        postfix = {
            "loss": f"{loss_avg.value:.4f}",
            "count_acc": f"{count_correct / max(count_total, 1):.3f}",
        }
        if is_train and learning_rate is not None:
            postfix["lr"] = f"{learning_rate:.2e}"
        for source_key, display_key in (
            ("count_loss", "count"),
            ("heatmap_loss", "heat"),
            ("region_loss", "region"),
            ("lognhi_loss", "lognhi"),
            ("offset_loss", "offset"),
        ):
            if source_key in parts_sum:
                postfix[display_key] = f"{parts_sum[source_key] / max(loss_avg.count, 1):.4f}"
        iterator.set_postfix(postfix)

    y_true_arr = np.concatenate(y_true)
    y_pred_arr = np.concatenate(y_pred)
    metrics = {
        "loss": loss_avg.value,
        "count_accuracy": float((y_true_arr == y_pred_arr).mean()),
        "count_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
    }
    for key, value in parts_sum.items():
        metrics[key] = value / max(loss_avg.count, 1)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    print(f"Loading train FITS: {args.train_fits}")
    wavelength, flux, flux_clean, labels = load_train_fits(
        args.train_fits,
        load_flux_clean=feature_requires_flux_clean(args.feature_mode),
    )
    n_dla = np.clip(labels["N_DLA"].to_numpy(dtype=np.int64), 0, 2)
    train_idx, val_idx = make_split(n_dla, args.val_size, args.split_seed, args.max_samples)
    np.savez(output_dir / "splits.npz", train_idx=train_idx, val_idx=val_idx)

    train_loader, val_loader = make_loaders(wavelength, flux, flux_clean, labels, train_idx, val_idx, args)
    used_count = int(len(train_idx) + len(val_idx))
    sample_note = f"max_samples={args.max_samples}" if args.max_samples is not None else "all samples"
    print("Dataset split:")
    print(f"  FITS spectra: {len(labels)}")
    print(f"  Used spectra: {used_count} ({sample_note})")
    print(f"  Train spectra: {len(train_idx)} [{format_n_dla_counts(n_dla, train_idx)}]")
    if len(val_idx):
        print(f"  Validation spectra: {len(val_idx)} [{format_n_dla_counts(n_dla, val_idx)}]")
    else:
        print("  Validation spectra: 0 (train.fits is used for training only)")
    print(f"  val_size={args.val_size}, split_seed={args.split_seed}")
    val_batches = len(val_loader) if val_loader is not None else 0
    print(f"  Batches per epoch: train={len(train_loader)}, val={val_batches}, batch_size={args.batch_size}")

    input_channels = feature_channels(args.feature_mode)
    use_context_channels = not args.no_context_channels
    model = create_model(
        input_channels=input_channels,
        base_channels=args.base_channels,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        use_context_channels=use_context_channels,
    ).to(device)
    class_weights = class_weight_tensor(n_dla, train_idx, device, enabled=not args.no_class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss_weights = LossWeights(
        count=args.lambda_count,
        heatmap=args.lambda_heatmap,
        region=args.lambda_region,
        lognhi=args.lambda_lognhi,
        offset=args.lambda_offset,
        heatmap_positive_weight=args.heatmap_positive_weight,
        region_positive_weight=args.region_positive_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "input_channels": input_channels,
            "n_total": int(len(labels)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "train_n_dla_counts": format_n_dla_counts(n_dla, train_idx),
            "val_n_dla_counts": format_n_dla_counts(n_dla, val_idx),
            "use_context_channels": use_context_channels,
            "spectrum_length": int(len(wavelength)),
            "wavelength_min": float(np.nanmin(wavelength)),
            "wavelength_max": float(np.nanmax(wavelength)),
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        }
    )
    save_json(config, output_dir / "config.json")

    best_loss = float("inf")
    best_f1 = -float("inf")
    best_f1_available = False
    history = []
    print(f"Training on device: {device}")
    print(f"Context channels: {'enabled' if use_context_channels else 'disabled'}")
    print(f"Progress bars: {'enabled' if not args.disable_tqdm else 'disabled'}")
    for epoch in range(1, args.epochs + 1):
        current_lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            criterion,
            loss_weights,
            optimizer=optimizer,
            show_progress=not args.disable_tqdm,
            epoch=epoch,
            epochs=args.epochs,
            phase="train",
            learning_rate=current_lr,
        )
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    criterion,
                    loss_weights,
                    optimizer=None,
                    show_progress=not args.disable_tqdm,
                    epoch=epoch,
                    epochs=args.epochs,
                    phase="val",
                )
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            **{f"train_{k}": v for k, v in train_metrics.items()},
        }
        if val_metrics is not None:
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)
        if val_metrics is not None:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_count_acc={val_metrics['count_accuracy']:.4f} "
                f"val_count_f1={val_metrics['count_macro_f1']:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.5f} "
                f"train_count_acc={train_metrics['count_accuracy']:.4f} "
                f"train_count_f1={train_metrics['count_macro_f1']:.4f}"
            )

        monitor_metrics = val_metrics if val_metrics is not None else train_metrics
        monitor_name = "val_loss" if val_metrics is not None else "train_loss"
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "monitor_name": monitor_name,
        }
        torch.save({**checkpoint, "best_metric": monitor_metrics["loss"]}, output_dir / "last_model.pt")
        if monitor_metrics["loss"] < best_loss:
            best_loss = monitor_metrics["loss"]
            torch.save({**checkpoint, "best_metric": best_loss}, output_dir / "best_loss_model.pt")
            torch.save({**checkpoint, "best_metric": best_loss}, output_dir / "best_model.pt")
        if val_metrics is not None and val_metrics["count_macro_f1"] > best_f1:
            best_f1 = val_metrics["count_macro_f1"]
            best_f1_available = True
            torch.save({**checkpoint, "best_metric": best_f1}, output_dir / "best_f1_model.pt")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Best {monitor_name}: {best_loss:.5f}")
    if best_f1_available:
        print(f"Best val count macro-F1: {best_f1:.5f}")
    else:
        print("Best val count macro-F1: n/a (no train.fits validation split)")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
