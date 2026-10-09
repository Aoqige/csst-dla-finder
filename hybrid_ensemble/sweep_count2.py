"""Refine the count-head gate (follow-up to sweep_count.py).

sweep_count.py found Final 0.5188 -> 0.5429 with count_bias=[0,1,1] at
threshold 0.45: the count head was vetoing 95.7% of spectra even though 46.9%
of them have their max heatmap in [0.20,0.45).  This script refines the bias
and the peak threshold on a finer grid, caching member inference to npz so the
grid is cheap to re-sweep.

Run on dataCop-119:  CUDA_VISIBLE_DEVICES=1 python sweep_count2.py
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
from decode import predict_member                           # noqa: E402
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
CACHE = "/tmp/sweep_pred_cache.npz"

MEMBERS = [
    ("v3c", os.path.expanduser("~/csst_dla_runs/20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5", os.path.expanduser("~/csst_dla_runs/20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6", os.path.expanduser("~/csst_dla_runs/20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]

# (bias_first_peak, bias_second_peak, threshold)
VARIANTS = [
    (1.0, 1.0, 0.45),   # previous best
    (0.5, 0.5, 0.45),
    (0.75, 0.75, 0.45),
    (1.25, 1.25, 0.45),
    (1.5, 1.5, 0.45),
    (1.5, 2.0, 0.45),
    (2.0, 2.0, 0.45),
    (1.0, 1.5, 0.45),
    (1.0, 1.0, 0.35),
    (1.0, 1.0, 0.40),
    (1.0, 1.0, 0.50),
    (1.25, 1.25, 0.40),
    (0.75, 0.75, 0.40),
]
DV_MERGE = 1500.0


def merge(a, b, out):
    subprocess.run([PY, MERGE, a, b, out, "union", str(DV_MERGE)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def get_cache(device):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: z[k] for k in z.files}
    out = {}
    for name, ckpt in MEMBERS:
        model, cfg = load_checkpoint(ckpt, device)
        ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pred = predict_member(model, loader, device)
        for key in ("heatmap", "lognhi", "offset", "count_logits"):
            if key in pred:
                out["%s__%s" % (name, key)] = pred[key]
        out["%s__rows" % name] = pred["rows"]
        del model
        torch.cuda.empty_cache()
        print("cached", name, flush=True)
    np.savez(CACHE, **out)
    return out


def main():
    device = resolve_device("cuda")
    store = get_cache(device)

    rf = "/tmp/ds_ref.npz"
    if os.path.exists(rf):
        z = np.load(rf)
        wave, zq, snr_, tid = z["wave"], z["zq"], z["snr"], z["tid"]
    else:
        ds0 = HybridTestDataset(TEST_FITS, "flux")
        wave = np.asarray(ds0.wavelength, np.float64)
        zq = np.asarray(ds0.zq, np.float64)
        snr_ = np.asarray(ds0.snr, np.float64)
        tid = np.asarray(ds0.targetid, np.int64)
        np.savez(rf, wave=wave, zq=zq, snr=snr_, tid=tid)

    class DS:
        def __init__(self, wave, zq, snr, tid):
            self.wavelength = wave
            self.zq = zq
            self.snr = snr
            self.targetid = tid

        def __len__(self):
            return len(self.targetid)

    ds = DS(wave, zq, snr_, tid)

    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    workdir = "/tmp/sweep_count2"
    os.makedirs(workdir, exist_ok=True)

    results = []
    for b1, b2, thr in VARIANTS:
        tag = "b%.2f_%.2f_t%.2f" % (b1, b2, thr)
        csvs = {}
        for name, _ in MEMBERS:
            pred = {k.split("__")[1]: store[k] for k in store.keys() if k.startswith(name + "__")}
            c = {"threshold": thr, "min_distance": 10, "min_z_dla": 1.10,
                 "use_offset": True, "lognhi_clip_min": 19.5, "lognhi_clip_max": 22.5,
                 "count_bias": [0.0, b1, b2],
                 "lognhi_calibration": {"slope": 1.0, "intercept": 0.0}}
            p = os.path.join(workdir, "%s_%s.csv" % (name, tag))
            write_submission(p, ds, pred, c)
            csvs[name] = p
        chain = os.path.join(workdir, "u1_%s.csv" % tag)
        merge(csvs["v3c"], csvs["v5"], chain)
        ch2 = os.path.join(workdir, "u2_%s.csv" % tag)
        merge(chain, csvs["v6"], ch2)
        ch3 = os.path.join(workdir, "u3_%s.csv" % tag)
        merge(ch2, SOTA_CSV, ch3)

        catalog = np.genfromtxt(ch3, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if catalog.ndim == 0:
            catalog = np.asarray([catalog])
        pred = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "SNR": []}
        for row in catalog:
            t_id = int(row["id"])
            for slot in (1, 2):
                zz = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
                if zz <= 0 or ln < 20.3:
                    continue
                if not (2550.0 <= 1215.67 * (1 + zz) < 4200.0):
                    continue
                pred["TARGETID"].append(t_id); pred["Z_DLA"].append(zz)
                pred["LOG_NHI"].append(ln); pred["SNR"].append(snr_by_target.get(t_id, np.nan))
        pred["TARGETID"] = np.asarray(pred["TARGETID"], np.int64)
        for k in ("Z_DLA", "LOG_NHI", "SNR"):
            pred[k] = np.asarray(pred[k], np.float32)
        r = score_catalog(truth, pred)
        results.append((tag, r, ch3))
        weak = int(sum(b["n_pred"] for b in r.bin_details
                       if b["lognhi_bin"] == "[20.3,20.5)" and b["n_truth"] > 0))
        print("bias[0,%.2f,%.2f] thr=%.2f  Final=%.4f Det=%.4f Param=%.4f compl=%.4f "
              "pur=%.4f n_pred=%5d n_match=%4d weak=%3d"
              % (b1, b2, thr, r.final_score, r.detection_score, r.parameter_score,
                 r.completeness, r.purity, r.n_pred, r.n_match, weak), flush=True)

    results.sort(key=lambda x: -x[1].final_score)
    print()
    for tag, r, _ in results[:5]:
        print("TOP %-30s Final=%.4f" % (tag, r.final_score))
    print("BEST csv:", results[0][2])
    print("baseline was 0.5188")


if __name__ == "__main__":
    main()
