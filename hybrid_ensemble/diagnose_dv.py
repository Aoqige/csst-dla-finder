"""Is the 233 km/s z-scatter noise-limited, or a fixable floor?

All five members (3 transformers + CNN fusion + the union) land at std_dv = 230-235 km/s,
which means the scatter is NOT independent per-model noise (otherwise 4-way averaging
would have cut it) -- it is a shared property of the z-decode.

This script asks where that scatter lives:
  * vs SNR            -> noise-limited (irreducible without more SNR)?
  * vs predicted-pixel phase -> integer-grid / offset-head artifact (fixable)?
  * vs logNHI         -> weak-line dominated?
  * heavy tail        -> how much of the loss is a handful of grossly-off pairs?

Run:  /home/dingjch/anaconda3/envs/ML_env/bin/python diagnose_dv.py <csv>
"""
import os
import sys

import numpy as np

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

import score_test                                          # noqa: E402
from csst_dla.scoring import score_catalog, greedy_match, C_KMS  # noqa: E402

TRUTH = "/data/aoqige/test_truth.fits"
WAVE_START, WAVE_STEP = 2554.0, 8.0
LYA = 1215.67


def build_pred(csv_path, snr_by_target):
    catalog = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if catalog.ndim == 0:
        catalog = np.asarray([catalog])
    out = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "SNR": []}
    for row in catalog:
        tid = int(row["id"])
        for slot in (1, 2):
            z = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
            if z <= 0 or ln < 20.3:
                continue
            if not (2550.0 <= LYA * (1 + z) < 4200.0):
                continue
            out["TARGETID"].append(tid); out["Z_DLA"].append(z)
            out["LOG_NHI"].append(ln); out["SNR"].append(snr_by_target.get(tid, np.nan))
    out["TARGETID"] = np.asarray(out["TARGETID"], np.int64)
    for k in ("Z_DLA", "LOG_NHI", "SNR"):
        out[k] = np.asarray(out[k], np.float32)
    return out


def table(name, key, dv, dlog, snr):
    print("\n  --- std_dv / T_dv by %s ---" % name)
    print("  %-16s %5s %8s %8s %8s %8s" % (name, "n", "mean_dv", "std_dv", "T_dv_if_only", "T_dl"))
    keys = sorted(set(key))
    for k in keys:
        m = key == k
        if m.sum() < 5:
            continue
        s = dv[m].std()
        t = np.exp(-s / 300.0) * np.exp(-abs(dv[m].mean()) / 150.0)
        td = np.exp(-dlog[m].std() / 0.25) * np.exp(-abs(dlog[m].mean()) / 0.1)
        print("  %-16s %5d %8.1f %8.1f %8.4f %8.4f"
              % (str(k), int(m.sum()), dv[m].mean(), s, t, td))


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
        "~/csst_dla_runs/20260912/ensemble_4way_v3/predictions.csv")
    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    pred = build_pred(csv_path, snr_by_target)
    r = score_catalog(truth, pred)
    m = greedy_match(truth, pred)
    ti = np.asarray([x[0] for x in m], int)
    pi = np.asarray([x[1] for x in m], int)
    dv = C_KMS * (pred["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
    dl = pred["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
    snr = truth["SNR"][ti]
    lam = LYA * (1.0 + pred["Z_DLA"][pi])
    pix = (lam - WAVE_START) / WAVE_STEP
    frac = pix - np.floor(pix)
    zt = truth["Z_DLA"][ti]

    print("### %s   Final=%.4f Param=%.4f  n_match=%d" % (csv_path, r.final_score,
                                                          r.parameter_score, len(m)))
    print("  overall std_dv=%.1f mean_dv=%+.1f | std_dlog=%.4f" % (dv.std(), dv.mean(), dl.std()))

    bins = np.array([0, 1, 2, 3, 4, 6, 8, 12, 99.0])
    idx = np.digitize(snr, bins) - 1
    keys = np.array(["SNR[%.0f,%.0f)" % (bins[i], bins[i + 1]) for i in range(len(bins) - 1)])
    table("SNR_GU", keys[idx], dv, dl, snr)

    zb = np.array([1.1, 1.4, 1.7, 2.0, 2.3, 2.5])
    idx = np.clip(np.digitize(zt, zb) - 1, 0, len(zb) - 2)
    keys = np.array(["z[%.1f,%.1f)" % (zb[i], zb[i + 1]) for i in range(len(zb) - 1)])
    table("z_DLA truth", keys[idx], dv, dl, snr)

    lb = np.array([20.3, 20.5, 20.8, 21.1, 21.5, 22.0, 30.0])
    idx = np.clip(np.digitize(truth["LOG_NHI"][ti], lb) - 1, 0, len(lb) - 2)
    keys = np.array(["NHI[%.1f,%.1f)" % (lb[i], lb[i + 1]) for i in range(len(lb) - 1)])
    table("logNHI truth", keys[idx], dv, dl, snr)

    pb = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0])
    idx = np.clip(np.digitize(frac, pb) - 1, 0, len(pb) - 2)
    keys = np.array(["phase[%.2f,%.2f)" % (pb[i], pb[i + 1]) for i in range(len(pb) - 1)])
    table("pixel phase", keys[idx], dv, dl, snr)

    print("\n  --- heavy tail ---")
    for thr in (150, 300, 450, 600):
        m_ = np.abs(dv) > thr
        print("   |dv| > %4d : %4d pairs (%.1f%%)  -> they carry %.1f%% of the dv variance"
              % (thr, m_.sum(), 100 * m_.mean(), 100 * dv[m_].var() * m_.sum() / (len(dv) * dv.var())))
    core = np.abs(dv) < 150
    print("   std_dv overall         = %.1f" % dv.std())
    print("   std_dv over |dv|<150   = %.1f  (%.1f%% of pairs)" % (dv[core].std(), 100 * core.mean()))


if __name__ == "__main__":
    main()
