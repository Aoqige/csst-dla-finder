"""Is the *count head* the recall gate?

Threshold sweep from 0.45 -> 0.12 added only 16 candidates (963 -> 979), which
means the heatmap is saturated/over-confident: lowering the peak threshold
cannot surface new candidates.  The remaining gate is
`n_pick = argmax(count_prob)` in write_submission, i.e. a 3-class global count
head decides *whether* any peak is emitted at all.

This script
  1. reports the empirical distribution of argmax(count_prob) and of the peak
     heatmap value,
  2. sweeps a bias added to the count logits (pushing the model to emit more
     peaks) and re-scores the 4-way union.

Run on dataCop-119:  CUDA_VISIBLE_DEVICES=1 python sweep_count.py
"""
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import HybridTestDataset                          # noqa: E402
from decode import predict_member, softmax                  # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device # noqa: E402
from predict_hybrid import write_submission                 # noqa: E402
import score_test                                           # noqa: E402
from csst_dla.scoring import score_catalog                  # noqa: E402

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

# (label, count_bias, threshold)
VARIANTS = [
    ("baseline            bias[0,0,0] thr.45", [0.0, 0.0, 0.0], 0.45),
    ("bias[0,1,1]         thr.45", [0.0, 1.0, 1.0], 0.45),
    ("bias[0,2,2]         thr.45", [0.0, 2.0, 2.0], 0.45),
    ("bias[0,3,3]         thr.45", [0.0, 3.0, 3.0], 0.45),
    ("bias[0,6,6]         thr.45", [0.0, 6.0, 6.0], 0.45),
    ("bias[0,2,2]         thr.30", [0.0, 2.0, 2.0], 0.30),
    ("bias[0,3,3]         thr.30", [0.0, 3.0, 3.0], 0.30),
]
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
        # --- gate diagnostics ---
        cp = softmax(pred["count_logits"])
        pick = np.argmax(cp, axis=1)
        hm = pred["heatmap"]
        mx = hm.max(axis=1)
        print("--- %s gate diagnostics ---" % name)
        print("   argmax(count) distribution: n0=%d (%.1f%%)  n1=%d (%.1f%%)  n2=%d (%.1f%%)"
              % (int((pick == 0).sum()), 100 * (pick == 0).mean(),
                 int((pick == 1).sum()), 100 * (pick == 1).mean(),
                 int((pick == 2).sum()), 100 * (pick == 2).mean()))
        for lo, hi in [(0, 0.12), (0.12, 0.2), (0.2, 0.45), (0.45, 0.7), (0.7, 1.01)]:
            m = (mx >= lo) & (mx < hi)
            print("   max-heatmap in [%.2f,%.2f): %5d (%.1f%%)   of which argmax(count)=0: %d"
                  % (lo, hi, int(m.sum()), 100 * m.mean(), int((m & (pick == 0)).sum())))
        del model
        torch.cuda.empty_cache()

    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    workdir = "/tmp/sweep_count"
    os.makedirs(workdir, exist_ok=True)

    results = []
    for label, bias, thr in VARIANTS:
        csvs = {}
        for name, _ in MEMBERS:
            pred, ds, cfg = cache[name]
            c = dict(cfg)
            c["threshold"] = thr
            c["min_distance"] = 10
            c["min_z_dla"] = 1.10
            c["use_offset"] = True
            c["lognhi_clip_min"] = 19.5
            c["lognhi_clip_max"] = 22.5
            c["count_bias"] = bias
            p = os.path.join(workdir, "%s_%s.csv" % (name, label.split()[0] + str(bias[1]) + str(thr)))
            write_submission(p, ds, pred, c)
            csvs[name] = p
        chain = os.path.join(workdir, "u1_%s.csv" % (label.split()[0] + str(bias[1]) + str(thr)))
        merge(csvs["v3c"], csvs["v5"], chain)
        ch2 = os.path.join(workdir, "u2_%s.csv" % (label.split()[0] + str(bias[1]) + str(thr)))
        merge(chain, csvs["v6"], ch2)
        ch3 = os.path.join(workdir, "u3_%s.csv" % (label.split()[0] + str(bias[1]) + str(thr)))
        merge(ch2, SOTA_CSV, ch3)

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
        pred["TARGETID"] = np.asarray(pred["TARGETID"], np.int64)
        for k in ("Z_DLA", "LOG_NHI", "SNR"):
            pred[k] = np.asarray(pred[k], np.float32)
        r = score_catalog(truth, pred)
        results.append((label, r))
        weak = int(sum(b["n_pred"] for b in r.bin_details
                       if b["lognhi_bin"] == "[20.3,20.5)" and b["n_truth"] > 0))
        print("%-34s Final=%.4f Det=%.4f Param=%.4f compl=%.4f pur=%.4f n_pred=%5d "
              "n_match=%4d weak_pred=%3d"
              % (label, r.final_score, r.detection_score, r.parameter_score,
                 r.completeness, r.purity, r.n_pred, r.n_match, weak), flush=True)

    print()
    best = max(results, key=lambda x: x[1].final_score)
    print("BEST: %s -> Final %.4f (baseline 0.5188)" % (best[0], best[1].final_score))


if __name__ == "__main__":
    main()
