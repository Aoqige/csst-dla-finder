"""Param (40% of Final) decomposition with SIGNED dv, plus a post-hoc z-correction what-if.

`greedy_match` returns dv as an absolute value, but `score_catalog` recomputes signed dv
from the matched pairs before forming score_z.  This script replicates the signed version
and answers:
  * how much of T_dv is lost to the systematic mean offset vs. the scatter?
  * what would a single post-hoc z-shift (measured on val) buy?
  * per-member comparison, so we can see whether the union averaging already helps.

Run:  /home/dingjch/anaconda3/envs/ML_env/bin/python analyze_param.py [csv ...]
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
RUNS = os.path.expanduser("~/csst_dla_runs")

DEFAULT = [
    ("SOTA union_4way_v3", os.path.join(RUNS, "20260912/ensemble_4way_v3/predictions.csv")),
    ("union_4way_v1", os.path.join(RUNS, "20260912/ensemble_4way_v1/predictions.csv")),
    ("SOTA member cnn", os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")),
    ("v3c single", "/tmp/eval_long60/v3c_b0.75.csv"),
    ("v5 single", "/tmp/eval_long60/v5_b0.75_ens.csv"),
    ("v6 single", "/tmp/eval_long60/v6_b1.5_ens.csv"),
]


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
            if not (2550.0 <= 1215.67 * (1 + z) < 4200.0):
                continue
            out["TARGETID"].append(tid); out["Z_DLA"].append(z)
            out["LOG_NHI"].append(ln); out["SNR"].append(snr_by_target.get(tid, np.nan))
    out["TARGETID"] = np.asarray(out["TARGETID"], np.int64)
    for k in ("Z_DLA", "LOG_NHI", "SNR"):
        out[k] = np.asarray(out[k], np.float32)
    return out


def terms(sig_dv, mean_dv, sig_dl, mean_dl):
    t_dv = np.exp(-sig_dv / 300.0) * np.exp(-abs(mean_dv) / 150.0)
    t_dl = np.exp(-sig_dl / 0.25) * np.exp(-abs(mean_dl) / 0.1)
    return t_dv, t_dl


def analyze(label, csv_path, truth, snr_by_target):
    if not os.path.exists(csv_path):
        print("### %-20s SKIP (missing)" % label)
        return
    pred = build_pred(csv_path, snr_by_target)
    r = score_catalog(truth, pred)
    m = greedy_match(truth, pred)
    ti = np.asarray([x[0] for x in m], int)
    pi = np.asarray([x[1] for x in m], int)
    dv = C_KMS * (pred["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
    dl = pred["LOG_NHI"][pi] - truth["LOG_NHI"][ti]

    s_dv, m_dv, s_dl, m_dl = dv.std(), dv.mean(), dl.std(), dl.mean()
    t_dv, t_dl = terms(s_dv, m_dv, s_dl, m_dl)
    print("\n### %-20s Final=%.4f Det=%.4f Param=%.4f  n_match=%d" %
          (label, r.final_score, r.detection_score, r.parameter_score, len(m)))
    print("   signed dv : mean=%+7.1f  std=%6.1f   [p5=%+.0f p50=%+.0f p95=%+.0f]"
          % (m_dv, s_dv, *np.percentile(dv, [5, 50, 95])))
    print("   signed dlog: mean=%+7.4f  std=%6.4f" % (m_dl, s_dl))
    print("   T_dv = exp(-%.1f/300)*exp(-|%+.1f|/150) = %.4f      <- zeroth order term"
          % (s_dv, m_dv, t_dv))
    print("   T_dl = exp(-%.4f/0.25)*exp(-|%+.4f|/0.1) = %.4f" % (s_dl, m_dl, t_dl))
    print("   param = 0.5*(T_dv+T_dl) = %.4f   (reported %.4f)" %
          (0.5 * (t_dv + t_dl), r.parameter_score))

    # what-if: kill the systematic mean offset (a 1-D post-hoc z shift measured on val)
    t_dv0, _ = terms(s_dv, 0.0, s_dl, m_dl)
    t_dl0, _ = terms(s_dv, m_dv, s_dl, 0.0)
    print("   IF mean_dv -> 0 : T_dv %.4f -> %.4f  => param %.4f  (Final %+.4f)"
          % (t_dv, t_dv0, 0.5 * (t_dv0 + t_dl), 0.4 * 0.5 * (t_dv0 - t_dv)))
    print("   IF mean_dlog -> 0: T_dl %.4f -> %.4f  => param %.4f  (Final %+.4f)"
          % (t_dl, t_dl0, 0.5 * (t_dv + t_dl0), 0.4 * 0.5 * (t_dl0 - t_dl)))
    for s in (120, 100, 80, 60):
        td, _ = terms(s, m_dv, s_dl, m_dl)
        print("   IF std_dv -> %3d (bias kept): param %.4f  (Final %+.4f)" %
              (s, 0.5 * (td + t_dl), 0.4 * 0.5 * (td - t_dv)))
    for s in (0.115, 0.10, 0.09, 0.08):
        _, tl = terms(s_dv, m_dv, s, m_dl)
        print("   IF std_dlog -> %.3f: param %.4f  (Final %+.4f)" %
              (s, 0.5 * (t_dv + tl), 0.4 * 0.5 * (tl - t_dl)))
    # pixel-scale context
    px = C_KMS * 8.0 / (1215.67 * (1 + 1.6))   # km/s per 8A pixel at z~1.6
    print("   |dv| within 1 px (%.0f km/s): %.1f%% ;  within 0.5 px: %.1f%%" %
          (px, 100 * (np.abs(dv) < px).mean(), 100 * (np.abs(dv) < px / 2).mean()))
    return r


def main():
    paths = sys.argv[1:]
    items = [(p, p) for p in paths] if paths else DEFAULT
    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    for label, p in items:
        analyze(label, p, truth, snr_by_target)


if __name__ == "__main__":
    main()
