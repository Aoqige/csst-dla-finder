"""Where are the remaining points?  Bin-loss table + Param decomposition + headroom math.

Reads one or more predictions.csv, re-scores them with the official scorer, and prints
  1. the (SNR x logNHI) bin-loss table sorted by weight  n_truth*(1-f1),
  2. aggregates by SNR and by logNHI,
  3. the two Param sub-terms and how sensitive Final is to std_dv / std_dlognhi,
  4. the matched-pair dv distribution (core vs long tail).

Run:  /home/dingjch/anaconda3/envs/ML_env/bin/python analyze_headroom.py  [csv ...]
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
    ("SOTA ensemble_4way_v3", os.path.join(RUNS, "20260912/ensemble_4way_v3/predictions.csv")),
    ("ensemble_4way_v1", os.path.join(RUNS, "20260912/ensemble_4way_v1/predictions.csv")),
    ("SOTA member (cnn fusion)", os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")),
    ("v3c single bias.75", "/tmp/eval_long60/v3c_b0.75.csv"),
    ("v3c single bias0", "/tmp/eval_long60/v3c_b0.0.csv"),
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


def analyze(label, csv_path, truth, snr_by_target):
    if not os.path.exists(csv_path):
        print("\n### %s -- SKIP (missing %s)" % (label, csv_path))
        return
    pred = build_pred(csv_path, snr_by_target)
    r = score_catalog(truth, pred)
    print("\n" + "=" * 96)
    print("### %s" % label)
    print("  Final=%.4f  Det=%.4f  Param=%.4f  compl=%.4f  pur=%.4f  n_pred=%d n_match=%d n_truth=%d"
          % (r.final_score, r.detection_score, r.parameter_score, r.completeness,
             r.purity, r.n_pred, r.n_match, r.n_truth))
    lost_final = 0.6 * (1 - r.detection_score) + 0.4 * (1 - r.parameter_score)
    print("  headroom: Det lost 0.6*%.4f=%.4f  |  Param lost 0.4*%.4f=%.4f  |  total %.4f"
          % (1 - r.detection_score, 0.6 * (1 - r.detection_score),
             1 - r.parameter_score, 0.4 * (1 - r.parameter_score), lost_final))

    # ---- 1. bin loss table ----
    W = r.n_truth
    rows = []
    for b in r.bin_details:
        rows.append((b["n_truth"] * (1 - b["f1"]), b))
    rows.sort(key=lambda x: -x[0])
    print("\n  --- bin loss table (top 15 by n_truth*(1-f1)) ---")
    print("  %-14s %-12s %6s %6s %6s %6s %7s %7s" %
          ("SNR", "logNHI", "nTruth", "nPred", "compl", "pur", "f1", "loss"))
    tot = sum(x[0] for x in rows)
    for loss, b in rows[:15]:
        print("  %-14s %-12s %6d %6d %6.3f %6.3f %7.4f %7.1f  (%.1f%% of loss)"
              % (b["snr_bin"], b["lognhi_bin"], b["n_truth"], b["n_pred"],
                 b["completeness"], b["purity"], b["f1"], loss, 100 * loss / W))
    print("  total loss weight = %.1f (= n_truth)  -> det = %.4f" % (tot, r.detection_score))

    # ---- 2. aggregates ----
    for axis in ("snr_bin", "lognhi_bin"):
        agg = {}
        for b in r.bin_details:
            k = b[axis]
            a = agg.setdefault(k, [0, 0, 0.0])
            a[0] += b["n_truth"]; a[1] += b["n_pred"]; a[2] += b["n_truth"] * b["f1"]
        print("\n  --- by %s ---" % axis)
        for k in sorted(agg, key=lambda z: (len(z), z)):
            nt, npd, wf = agg[k]
            if nt == 0:
                continue
            print("   %-14s n_truth=%4d n_pred=%4d  bin-weight=%5.1f%%  f1_avg=%.3f  loss_share=%5.1f%%"
                  % (k, nt, npd, 100 * nt / W, wf / nt, 100 * nt * (1 - wf / nt) / W))

    # ---- 3. param decomposition + sensitivity ----
    matches = greedy_match(truth, pred)
    dv = np.array([m[2] for m in matches])
    dlog = np.array([pred["LOG_NHI"][m[1]] - truth["LOG_NHI"][m[0]] for m in matches])
    T_dv = np.exp(-dv.std() / 300.0) * np.exp(-abs(dv.mean()) / 150.0)
    T_ln = np.exp(-dlog.std() / 0.25) * np.exp(-abs(dlog.mean()) / 0.1)
    print("\n  --- Param decomposition (param = 0.5*(T_dv + T_log)) ---")
    print("   std_dv=%.1f mean_dv=%.1f  -> T_dv = exp(-s/300)*exp(-|m|/150) = %.4f  (gap %.4f)" %
          (dv.std(), dv.mean(), T_dv, 1 - T_dv))
    print("   std_dlog=%.4f mean_dlog=%+.4f -> T_log = exp(-s/0.25)*exp(-|m|/0.1) = %.4f  (gap %.4f)" %
          (dlog.std(), dlog.mean(), T_ln, 1 - T_ln))
    print("   dv percentiles: p10=%.0f p25=%.0f p50=%.0f p75=%.0f p90=%.0f p95=%.0f max=%.0f" %
          tuple(np.percentile(dv, [10, 25, 50, 75, 90, 95, 100])))
    for lo, hi in ((0, 150), (150, 300), (300, 450), (450, 600)):
        m = (dv >= lo) & (dv < hi)
        print("   dv in [%3d,%3d): %4d matches (%.1f%%)" % (lo, hi, m.sum(), 100 * m.mean()))
    print("   !! if the dv>450 tail were snapped to <150: std_dv would be %.1f" %
          dv[dv < 450].std())
    print("   sensitivity: Final gain if")
    for tgt_s, tgt_l in ((200, 0.11), (167, 0.10), (150, 0.087), (100, 0.07), (50, 0.05)):
        f_dv = np.exp(-tgt_s / 300.0) * np.exp(-abs(dv.mean()) / 150.0)
        f_ln = np.exp(-tgt_l / 0.25) * np.exp(-abs(dlog.mean()) / 0.1)
        new_param = 0.5 * (f_dv + f_ln)
        print("      std_dv->%5.0f & std_dlog->%.3f : Param %.4f -> %.4f   Final +%.4f"
              % (tgt_s, tgt_l, r.parameter_score, new_param,
                 0.4 * (new_param - r.parameter_score)))
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
