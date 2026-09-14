#!/usr/bin/env python3
"""Forward once, decode+score across multiple thresholds."""
from __future__ import annotations
import sys, json, subprocess, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import HybridTestDataset
from decode import average_predictions, predict_member
from evaluate_hybrid import load_checkpoint, resolve_device
from predict_hybrid import write_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble-config", required=True)
    p.add_argument("--test-fits", default="/data/aoqige/test.fits")
    p.add_argument("--truth", default="/data/aoqige/test_truth.fits")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-lognhi", type=float, default=20.3)
    p.add_argument("--snr-field", default="SNR_GU")
    args = p.parse_args()

    config = json.loads(Path(args.ensemble_config).read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    preds, weights, ref = [], [], None
    for m in config["models"]:
        model, cfg = load_checkpoint(m["path"], device)
        ds = HybridTestDataset(args.test_fits, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        preds.append(predict_member(model, loader, device))
        weights.append(float(m.get("weight", 1.0)))
        ref = ds
    avg = average_predictions(preds, weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ths = [float(x) for x in args.thresholds.split(",") if x.strip()]
    here = Path(__file__).resolve().parent
    print("th     Final  Det    Param  compl  purity std_dv std_logn n_pred n_match")
    for th in ths:
        config["threshold"] = th
        csv_p = out_dir / f"pred_th{th:.2f}.csv"
        write_submission(csv_p, ref, avg, config)
        sj = out_dir / f"score_th{th:.2f}.json"
        subprocess.run(
            [sys.executable, str(here / "score_test.py"),
             "--predictions", str(csv_p), "--truth", args.truth,
             "--out", str(sj), "--min-lognhi", str(args.min_lognhi),
             "--snr-field", args.snr_field],
            check=True,
        )
        s = json.loads(sj.read_text(encoding="utf-8"))
        print("{:.2f}  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:5.0f} {:.4f} {:5d} {:5d}".format(
            th, s["final_score"], s["detection_score"], s["parameter_score"],
            s["completeness"], s["purity"], s["std_dv"], s["std_dlognhi"],
            s["n_pred"], s["n_match"]))


if __name__ == "__main__":
    main()
