#!/usr/bin/env python3
"""Generate a submission from a trained dual-tower feature fusion checkpoint."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "hybrid_ensemble"))
sys.path.insert(0, str(ROOT / "vendor"))

from evaluate_hybrid import load_checkpoint, resolve_device
from feature_fusion import DualFusionTestDataset, DualTowerFusionNet
from predict_hybrid import write_submission
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-fits", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--min-distance", type=int, default=10)
    parser.add_argument("--min-z-dla", type=float, default=1.10)
    parser.add_argument(
        "--dilated-checkpoint", default=None,
        help="Override config['dilated_checkpoint']. Not needed for checkpoints that "
             "embed dilated_config; use it only for legacy checkpoints.")
    parser.add_argument(
        "--wzx-checkpoint", default=None,
        help="Override config['wzx_checkpoint']. Not needed for checkpoints that "
             "embed wzx_config; use it only for legacy checkpoints.")
    return parser.parse_args()


def _build_dilated(config: dict, device: torch.device, override: str | None = None):
    """Rebuild the dilated tower.

    The fused checkpoint embeds ``dilated_config``, so the original tower
    checkpoint is not needed.  ``override`` (``--dilated-checkpoint``) forces the
    legacy path, for checkpoints that carry only the training-time location.
    """
    from model import HybridDlaNet, input_channels

    if override is None and "dilated_config" in config:
        dc = config["dilated_config"]
        print("[fusion] building the dilated tower from the embedded dilated_config", flush=True)
        # Build through HybridDlaNet, exactly as the training code does.  The
        # dilation schedule (stages=_stages_for(num_blocks)) and n_bins are NOT
        # part of the state dict, so building the raw module with defaults gives
        # identical weights but a different network.
        return HybridDlaNet(
            in_channels=input_channels(dc.get("input_mode", "flux")),
            hidden=int(dc.get("hidden", 96)),
            num_blocks=int(dc.get("num_blocks", 4)),
            with_offset=bool(dc.get("with_offset", True)),
            norm_type=str(dc.get("norm_type", "layer")),
            head_layers=int(dc.get("head_layers", 1)),
        ).model.to(device)
    path = override or config["dilated_checkpoint"]
    print(f"[fusion] loading the dilated tower from {path}", flush=True)
    dilated, _ = load_checkpoint(path, device)
    return dilated.model


def _build_wzx(config: dict, device: torch.device, override: str | None = None):
    """Rebuild the WZX tower, same policy as :func:`_build_dilated`."""
    from csst_dla_wzx_pkg.data import feature_channels
    from csst_dla_wzx_pkg.model import create_model

    if override is None and "wzx_config" in config:
        wc = config["wzx_config"]
        print("[fusion] building the WZX tower from the embedded wzx_config", flush=True)
        return create_model(
            input_channels=int(
                wc.get("input_channels", feature_channels(wc.get("feature_mode", "all")))
            ),
            base_channels=int(wc.get("base_channels", 96)),
            num_blocks=int(wc.get("num_blocks", 8)),
            dropout=float(wc.get("dropout", 0.1)),
            use_context_channels=bool(wc.get("use_context_channels", True)),
        ).to(device)
    path = override or config["wzx_checkpoint"]
    print(f"[fusion] loading the WZX tower from {path}", flush=True)
    wzx, _ = load_model_from_checkpoint(path, device)
    return wzx


def load_fusion(path: str | Path, device: torch.device,
                dilated_ckpt: str | None = None, wzx_ckpt: str | None = None):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = DualTowerFusionNet(
        _build_dilated(config, device, dilated_ckpt),
        _build_wzx(config, device, wzx_ckpt),
        merge_mode=config["merge_mode"],
        width=int(config["fusion_width"]),
        depth=int(config["fusion_depth"]),
        freeze_backbones=bool(config["freeze_backbones"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, dataset, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    heat, lognhi, offset, count_logits, rows = [], [], [], [], []
    for hybrid, wzx, zq, row in loader:
        output = model(hybrid.to(device), wzx.to(device), zq.to(device))
        heat.append(torch.sigmoid(output["center_logits"]).cpu().numpy())
        lognhi.append((20.3 + output["lognhi_raw"]).cpu().numpy())
        offset.append(output["offset_raw"].cpu().numpy())
        count_logits.append(output["count_logits"].cpu().numpy())
        rows.append(np.asarray(row, dtype=np.int64))
    return {
        "heatmap": np.concatenate(heat),
        "lognhi": np.concatenate(lognhi),
        "offset": np.concatenate(offset),
        "count_logits": np.concatenate(count_logits),
        "rows": np.concatenate(rows),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_fusion(args.checkpoint, device,
                                  args.dilated_checkpoint, args.wzx_checkpoint)
    fusion_config = checkpoint["config"]
    dilated_input_mode = str(
        fusion_config.get("dilated_input_mode", fusion_config.get("dilated_config", {}).get("input_mode", "all"))
    )
    wzx_feature_mode = str(
        fusion_config.get("wzx_feature_mode", fusion_config.get("wzx_config", {}).get("feature_mode", "all"))
    )
    dataset = DualFusionTestDataset(
        args.test_fits,
        dilated_input_mode=dilated_input_mode,
        wzx_feature_mode=wzx_feature_mode,
    )
    prediction = predict(model, dataset, device, args.batch_size)
    config = {
        "threshold": float(args.threshold if args.threshold is not None else checkpoint.get("threshold", 0.45)),
        "min_distance": args.min_distance,
        "min_z_dla": args.min_z_dla,
        "use_offset": True,
        "lognhi_clip_min": 20.3,
        "lognhi_clip_max": 22.5,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(output, dataset, prediction, config)
    output.with_suffix(".config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
