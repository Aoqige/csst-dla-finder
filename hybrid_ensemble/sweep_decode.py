"""Sweep the detection threshold of the three transformer members of the
4-way union, offline.  Inference runs ONCE per member; each threshold only
re-runs the cheap CSV decode + merge + score.

Why: the bin breakdown shows the low-SNR mid-logNHI bins are
completeness-limited (compl 0.15-0.31) while purity is still 0.57-0.67, i.e.
the threshold is too conservative there; and the weak-DLA bins
(TRUE logNHI in [20.3,20.5)) get almost no predictions at all.

Run on dataCop-119:  CUDA_VISIBLE_DEVICES=1 python sweep_decode.py
"""
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import HybridTestDataset                    # noqa: E402
from decode import predict_member                     # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
from predict_hybrid import write_submission           # noqa: E402
import score_test                                     # noqa: E402
from csst_dla.scoring import score_catalog            # noqa: E402

PY = "/home/dingjch/anaconda3/envs/ML_env/bin/python"
TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
MERGE = os.path.join(HB, "merge_predictions.py")
SOTA_CSV = os.path.expanduser(
    "~/csst_dla_runs/20260904/feature_no_clean_flux_context_v1/predictions.csv")

MEMBERS = [
    ("v3c", os.path.expanduser("~/csst_dla_runs/20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5", os.path.expanduser("~/csst_dla_runs/20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6", os.path.expanduser("~/csst_dla_runs/20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]

THRESHOLDS = [0.45, 0.35, 0.28, 0.20, 0.12]
DV_MERGE = 1500.0


def merge(a, b, out):
    subprocess.run([PY, MERGE, a, b, out, "union", str(DV_MERGE)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    device = resolve_device("cuda")
    cache = {}
    for name, ckpt in MEMBERS:
        model, cfg = load_checkpoint(ckpt, device)
        ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pred = predict_member(model, loader, device)
        cache[name] = (pred, ds, cfg)
        print("inference done: %s  heatmap%s" % (name, pred["heatmap"].shape), flush=True)
        del model
        torch.cuda.empty_cache()

    truth = score_test.load_truth(TRUTH)
    workdir = "/tmp/sweep_decode"
    os.makedirs(workdir, exist_ok=True)

    results = []
    for t in THRESHOLDS:
        csvs = {}
        for name, _ in MEMBERS:
            pred, ds, cfg = cache[name]
            c = dict(cfg)
            c["threshold"] = t
            c["min_distance"] = 10
            c["min_z_dla"] = 1.10
            c["use_offset"] = True
            c["lognhi_clip_min"] = 19.5
            c["lognhi_clip_max"] = 22.5
            p = os.path.join(workdir, "%s_t%.2f.csv" % (name, t))
            write_submission(p, ds, pred, c)
            csvs[name] = p
        chain = os.path.join(workdir, "u_%s_1.csv" % t)
        merge(csvs["v3c"], csvs["v5"], chain)
        ch2 = os.path.join(workdir, "u_%s_2.csv" % t)
        merge(chain, csvs["v6"], ch2)
        ch3 = os.path.join(workdir, "u_%s_3.csv" % t)
        merge(ch2, SOTA_CSV, ch3)

        # score with the official scorer (mirrors score_test.main)
        import csv as _csv

        snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
        catalog = np.genfromtxt(ch3, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if catalog.ndim == 0:
            catalog = np.asarray([catalog])
        pred = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "SNR": []}
        for row in catalog:
            tid = int(row["id"])
            for slot in (1, 2):
                z = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
                if z <= 0 or ln < 20.3:
                    continue
                if not (2550.0 <= 1215.67 * (1 + z) < 4200.0):
                    continue
                pred["TARGETID"].append(tid); pred["Z_DLA"].append(z)
                pred["LOG_NHI"].append(ln); pred["SNR"].append(snr_by_target.get(tid, np.nan))
        for k in ("TARGETID",):
            pred[k] = np.asarray(pred[k], np.int64)
        for k in ("Z_DLA", "LOG_NHI", "SNR"):
            pred[k] = np.asarray(pred[k], np.float32)
        r = score_catalog(truth, pred)
        results.append((t, r))
        print("thr=%.2f  Final=%.4f  Det=%.4f  Param=%.4f  compl=%.4f  pur=%.4f  "
              "n_pred=%4d  n_match=%4d  std_dv=%.1f"
              % (t, r.final_score, r.detection_score, r.parameter_score,
                 r.completeness, r.purity, r.n_pred, r.n_match, r.std_dv), flush=True)

    print()
    best = max(results, key=lambda x: x[1].final_score)
    print("BEST threshold = %.2f  Final = %.4f" % (best[0], best[1].final_score))
    # weak-bin check at best threshold
    bd = [b for b in best[1].bin_details if b["n_truth"] > 0
          and b["lognhi_bin"] == "[20.3,20.5)"]
    print("weak-DLA bins at best thr:")
    for b in bd:
        print("   SNR%-10s n_truth=%4d n_pred=%3d compl=%.3f pur=%.3f f1=%.3f"
              % (b["snr_bin"], b["n_truth"], b["n_pred"], b["completeness"],
                 b["purity"], b["f1"]))


if __name__ == "__main__":
    main()
