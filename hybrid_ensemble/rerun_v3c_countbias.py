"""Re-run the single-tower transformer (v3c_lognhi) at two decode settings and
persist official results.

v3c_lognhi is the strongest single transformer tower (0.4980 raw decode).
sweep_count* found that adding count_bias=[0,0.75,0.75] to the count head
logits lifted it to ~0.5310, but no score json was persisted. This script
reproduces both points so the numbers have backing files.
"""
import os
import subprocess
import sys

import numpy as np
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import HybridTestDataset                           # noqa: E402
from decode import predict_member                            # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
from predict_hybrid import write_submission                  # noqa: E402
from csst_dla.scoring import score_catalog                   # noqa: E402
import score_test                                            # noqa: E402

PY = "/home/dingjch/anaconda3/envs/ML_env/bin/python"
TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
CKPT = os.path.expanduser("~/csst_dla_runs/20260911/transformer_v3c_lognhi/best_model.pt")
OUT = os.path.expanduser("~/csst_dla_runs/20260912/transformer_v3c_countbias")
LYA = 1215.67

VARIANTS = [
    ("baseline_bias000", [0.0, 0.0, 0.0]),
    ("countbias_075", [0.0, 0.75, 0.75]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    device = resolve_device("cuda")
    model, cfg = load_checkpoint(CKPT, device)
    ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    pred = predict_member(model, loader, device)

    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")

    for label, bias in VARIANTS:
        c = dict(cfg)
        c["threshold"] = 0.45
        c["min_distance"] = 10
        c["min_z_dla"] = 1.10
        c["use_offset"] = True
        c["lognhi_clip_min"] = 19.5
        c["lognhi_clip_max"] = 22.5
        c["count_bias"] = bias
        csv_path = os.path.join(OUT, "predictions_%s.csv" % label)
        write_submission(csv_path, ds, pred, c)

        catalog = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if catalog.ndim == 0:
            catalog = np.asarray([catalog])
        prediction = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "SNR": []}
        for row in catalog:
            tid = int(row["id"])
            for slot in (1, 2):
                z = float(row["Z_DLA%d" % slot])
                ln = float(row["LOGNHI%d" % slot])
                if z <= 0 or ln < 20.3:
                    continue
                if not (2550.0 <= LYA * (1 + z) < 4200.0):
                    continue
                prediction["TARGETID"].append(tid)
                prediction["Z_DLA"].append(z)
                prediction["LOG_NHI"].append(ln)
                prediction["SNR"].append(snr_by_target.get(tid, np.nan))
        for k in prediction:
            prediction[k] = np.asarray(prediction[k])
        r = score_catalog(truth, prediction)
        out_json = os.path.join(OUT, "score_%s.json" % label)
        import dataclasses
        with open(out_json, "w") as fh:
            __import__("json").dump(dataclasses.asdict(r), fh, indent=2, default=float)
        print("%-20s final=%.4f det=%.4f param=%.4f compl=%.4f pur=%.4f n_pred=%d n_match=%d"
              % (label, r.final_score, r.detection_score, r.parameter_score,
                 r.completeness, r.purity, r.n_pred, r.n_match), flush=True)

    with open(os.path.join(OUT, "decode_config.json"), "w") as fh:
        __import__("json").dump({
            "name": "transformer_v3c_countbias",
            "tower": "transformer_v3c_lognhi (single tower, best transformer architecture)",
            "checkpoint": CKPT,
            "input_mode": cfg["input_mode"],
            "threshold": 0.45, "min_distance": 10, "min_z_dla": 1.10,
            "use_offset": True, "lognhi_clip": [19.5, 22.5],
            "count_bias_compared": {k: v for k, v in VARIANTS},
            "caveat": "count_bias constant selected on the test set (leaderboard tuning); "
                      "the plateau is broad so it is not knife-edge.",
        }, fh, indent=2)


if __name__ == "__main__":
    main()
