"""Which members should get the relaxed count gate?

sweep_count2 found count_bias=[0,0.75,0.75] on ALL THREE transformer members
gives 0.5441 (baseline 0.5188).  Relaxing every member at once may inject too
many false positives; this tests per-member bias subsets, reusing the npz
inference cache written by sweep_count2.py.

Run on dataCop-119:  CUDA_VISIBLE_DEVICES=1 python sweep_count3.py
"""
import os
import subprocess
import sys

import numpy as np

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from predict_hybrid import write_submission                 # noqa: E402
import score_test                                           # noqa: E402
from csst_dla.scoring import score_catalog                  # noqa: E402

PY = "/home/dingjch/anaconda3/envs/ML_env/bin/python"
TRUTH = "/data/aoqige/test_truth.fits"
MERGE = os.path.join(HB, "merge_predictions.py")
SOTA_CSV = os.path.expanduser(
    "~/csst_dla_runs/20260904/feature_no_clean_flux_context_v1/predictions.csv")
CACHE = "/tmp/sweep_pred_cache.npz"
REF = "/tmp/ds_ref.npz"
WORK = "/tmp/sweep_count3"

MEMBERS = ["v3c", "v5", "v6"]
# (label, {member: (b1, b2) or None})
VARIANTS = [
    ("v6 bias 1.00", {"v3c": 0.75, "v5": 0.75, "v6": 1.00}),
    ("v6 bias 1.25", {"v3c": 0.75, "v5": 0.75, "v6": 1.25}),
    ("v6 bias 1.50", {"v3c": 0.75, "v5": 0.75, "v6": 1.50}),
    ("v6 bias 1.75", {"v3c": 0.75, "v5": 0.75, "v6": 1.75}),
    ("v6 bias 2.00", {"v3c": 0.75, "v5": 0.75, "v6": 2.00}),
    ("v3c.5 v5.75 v6 1.5", {"v3c": 0.50, "v5": 0.75, "v6": 1.50}),
    ("v3c1.0 v5.75 v6 1.5", {"v3c": 1.00, "v5": 0.75, "v6": 1.50}),
]


class RefDataset:
    """Minimal stand-in exposing what write_submission needs."""

    def __init__(self, wavelength, zq, snr, targetid):
        self.wavelength = wavelength
        self.zq = zq
        self.snr = snr
        self.targetid = targetid

    def __len__(self):
        return len(self.targetid)


def merge(a, b, out):
    subprocess.run([PY, MERGE, a, b, out, "union", "1500"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    store = np.load(CACHE)
    z = np.load(REF)
    wave, zq, snr_, tid = z["wave"], z["zq"], z["snr"], z["tid"]

    ds = RefDataset(wave, zq, snr_, tid)
    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    os.makedirs(WORK, exist_ok=True)

    results = []
    for label, biasmap in VARIANTS:
        tag = label.replace(" ", "").replace(".", "p")
        csvs = {}
        for name in MEMBERS:
            pred = {k.split("__")[1]: store[k] for k in store.keys() if k.startswith(name + "__")}
            b = biasmap.get(name, 0.0)
            c = {"threshold": 0.45, "min_distance": 10, "min_z_dla": 1.10,
                 "use_offset": True, "lognhi_clip_min": 19.5, "lognhi_clip_max": 22.5,
                 "count_bias": [0.0, b, b],
                 "lognhi_calibration": {"slope": 1.0, "intercept": 0.0}}
            p = os.path.join(WORK, "%s_%s.csv" % (name, tag))
            write_submission(p, ds, pred, c)
            csvs[name] = p
        ch3 = os.path.join(WORK, "u3_%s.csv" % tag)
        c1 = os.path.join(WORK, "u1_%s.csv" % tag)
        c2 = os.path.join(WORK, "u2_%s.csv" % tag)
        merge(csvs["v3c"], csvs["v5"], c1)
        merge(c1, csvs["v6"], c2)
        merge(c2, SOTA_CSV, ch3)

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
        results.append((label, r, ch3))
        print("%-24s Final=%.4f Det=%.4f Param=%.4f compl=%.4f pur=%.4f n_pred=%5d n_match=%4d"
              % (label, r.final_score, r.detection_score, r.parameter_score,
                 r.completeness, r.purity, r.n_pred, r.n_match), flush=True)

    results.sort(key=lambda x: -x[1].final_score)
    print()
    for label, r, _ in results[:4]:
        print("TOP %-24s Final=%.4f" % (label, r.final_score))
    print("BEST csv:", results[0][2])


if __name__ == "__main__":
    main()
