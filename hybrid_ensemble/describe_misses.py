"""Characterise the 303 truths that NO member sees (the model-failure floor) and the 86 that
are visible but z-shifted, using the cached pool.

Run:  /home/dingjch/anaconda3/envs/ML_env/bin/python describe_misses.py
"""
import os
import sys

import numpy as np

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

import score_test                                           # noqa: E402
from csst_dla.scoring import C_KMS                           # noqa: E402

TRUTH = "/data/aoqige/test_truth.fits"
CACHE = "/tmp/pool_votes.npz"


def main():
    d = np.load(CACHE)
    tid, z = d["tid"], d["z"]
    truth = score_test.load_truth(TRUTH)
    tt, tz, tn, ts = truth["TARGETID"], truth["Z_DLA"], truth["LOG_NHI"], truth["SNR"]

    order = np.argsort(tt, kind="stable")
    tts = tt[order]
    nearest = np.full(len(tt), np.inf)
    for i in range(len(tt)):
        lo = np.searchsorted(tts, tt[i], "left")
        hi = np.searchsorted(tts, tt[i], "right")
        if lo == hi:
            continue
        idx = order[lo:hi]
        # candidates in this spectrum
        cm = tid == tt[i]
        if not cm.any():
            continue
        zc = z[cm]
        nearest[i] = (C_KMS * np.abs(zc - tz[i]) / (1.0 + tz[i])).min()

    grp = np.where(nearest < 600, "A matched (pool)",
          np.where(nearest < 2000, "B z-off ~1px", "C model-blind"))
    print("=== truth-side grouping (permissive pool, 4 members) ===")
    for g in ("A matched (pool)", "B z-off ~1px", "C model-blind"):
        m = grp == g
        print("  %-16s n=%4d (%.1f%%)  median SNR=%5.2f  median logNHI=%.2f"
              % (g, m.sum(), 100 * m.mean(), np.median(ts[m]), np.median(tn[m])))

    print("\n=== the %d model-blind truths, by (SNR x logNHI) ===" % (grp == "C model-blind").sum())
    sb = np.array([0, 1, 2, 3, 5, 99.0])
    lb = np.array([20.3, 20.5, 21.0, 21.5, 99.0])
    tot = (grp == "C model-blind").sum()
    print("  %-12s" % "SNR\\NHI" + "".join("%12s" % ("%.1f-%.1f" % (lb[i], lb[i + 1]))
                                          for i in range(len(lb) - 1)))
    for i in range(len(sb) - 1):
        line = "  %-12s" % ("%.0f-%.0f" % (sb[i], sb[i + 1]))
        for j in range(len(lb) - 1):
            m = (grp == "C model-blind") & (ts >= sb[i]) & (ts < sb[i + 1]) & \
                (tn >= lb[j]) & (tn < lb[j + 1])
            line += "%12s" % ("%d(%.1f%%)" % (m.sum(), 100 * m.sum() / max(tot, 1)))
        print(line)
    print("\n  sanity: total truth by SNR bin (all 1644):")
    for i in range(len(sb) - 1):
        m = (ts >= sb[i]) & (ts < sb[i + 1])
        print("    SNR %.0f-%.0f: n=%4d  of which blind=%3d (%.1f%%)"
              % (sb[i], sb[i + 1], m.sum(), (m & (grp == "C model-blind")).sum(),
                 100 * (m & (grp == "C model-blind")).mean()))
    print("  sanity: total truth by logNHI bin:")
    for j in range(len(lb) - 1):
        m = (tn >= lb[j]) & (tn < lb[j + 1])
        print("    NHI %.1f-%.1f: n=%4d  of which blind=%3d (%.1f%%)"
              % (lb[j], lb[j + 1], m.sum(), (m & (grp == "C model-blind")).sum(),
                 100 * (m & (grp == "C model-blind")).mean()))


if __name__ == "__main__":
    main()
