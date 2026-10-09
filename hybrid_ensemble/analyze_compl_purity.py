"""Focus on completeness & purity (not the composite score).

Builds a PERMISSIVE candidate pool (all members, low threshold, count gate wide open),
then answers three structural questions:

  Q1 (compl)  the misses: was a candidate in the pool near the truth (= a *selection*
              failure, fixable by a better ranker) or was the pool empty there
              (= a *model* failure)?
  Q2 (purity) the false positives: are they near a truth (600<dv<2000) i.e. the right
              DLA with a wrong z -- which costs BOTH a miss and an FP -- or pure noise?
  Q3 (ranker) is there any per-candidate signal that separates TP from FP?
              AUC per feature + the compl/purity frontier obtained by ranking.

Run:  CUDA_VISIBLE_DEVICES=0 /home/dingjch/anaconda3/envs/ML_env/bin/python analyze_compl_purity.py
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import HybridTestDataset                          # noqa: E402
from decode import predict_member, softmax, pick_peaks, pixel_to_wavelength  # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
import score_test                                           # noqa: E402
from csst_dla.scoring import C_KMS                          # noqa: E402

TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
RUNS = os.path.expanduser("~/csst_dla_runs")
LYA = 1215.67

TRANSFORMERS = [
    ("v3c", os.path.join(RUNS, "20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5",  os.path.join(RUNS, "20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6",  os.path.join(RUNS, "20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]
SOTA_CSV = os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")

POOL = dict(threshold=0.20, min_distance=10, min_z_dla=1.10, count_bias=[0.0, 3.0, 3.0],
            use_offset=True, lognhi_clip_min=19.5, lognhi_clip_max=22.5)
MATCH_DV = 600.0
NEAR_DV = 2000.0
DEDUP_DV = 1500.0


def build_pool(device):
    """Return dict of parallel arrays over pool candidates."""
    cols = {k: [] for k in ("tid", "z", "lognhi", "hm", "cprob", "snr", "offpix", "member")}
    ds = None
    for name, ckpt in TRANSFORMERS:
        model, cfg = load_checkpoint(ckpt, device)
        ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pred = predict_member(model, loader, device)
        cp = softmax(pred["count_logits"] + np.asarray(POOL["count_bias"], np.float32)[None, :])
        n_pick = np.argmax(cp, axis=1)
        rows = pred["rows"]
        for r_local, row in enumerate(rows):
            n = int(n_pick[r_local])
            hm = pred["heatmap"][r_local]
            peaks = pick_peaks(hm, ds.wavelength, float(ds.zq[row]), n_pick=n,
                               threshold=POOL["threshold"], min_distance=POOL["min_distance"],
                               min_z_dla=POOL["min_z_dla"])
            for pix in peaks:
                lam = pixel_to_wavelength(pix + float(pred["offset"][r_local, pix]), ds.wavelength)
                cols["tid"].append(int(ds.targetid[row]))
                cols["z"].append(float(lam / LYA - 1.0))
                cols["lognhi"].append(float(np.clip(pred["lognhi"][r_local, pix],
                                                    POOL["lognhi_clip_min"], POOL["lognhi_clip_max"])))
                cols["hm"].append(float(hm[pix]))
                cols["cprob"].append(float(cp[r_local, n]))
                cols["snr"].append(float(ds.snr[row]))
                cols["offpix"].append(abs(float(pred["offset"][r_local, pix])))
                cols["member"].append(name)
        print("  pool[%s] members done" % name, flush=True)
        del model
        torch.cuda.empty_cache()
    # SOTA member comes from a finished CSV (already thresholded at 0.45, <=2 rows/spectrum)
    tid2row = {int(t): i for i, t in enumerate(ds.targetid)}
    cat = np.genfromtxt(SOTA_CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if cat.ndim == 0:
        cat = np.asarray([cat])
    for r in cat:
        tid = int(r["id"])
        for slot in (1, 2):
            z = float(r["Z_DLA%d" % slot]); ln = float(r["LOGNHI%d" % slot])
            if z <= 0:
                continue
            row = tid2row.get(tid)
            cols["tid"].append(tid); cols["z"].append(z); cols["lognhi"].append(ln)
            cols["hm"].append(np.nan); cols["cprob"].append(np.nan)
            cols["snr"].append(float(ds.snr[row]) if row is not None else np.nan)
            cols["offpix"].append(np.nan); cols["member"].append("cnn")
    print("  pool[cnn] merged from csv", flush=True)
    return {k: np.asarray(v) for k, v in cols.items()}, ds


def classify(cand, truth):
    """For every candidate: dv to nearest truth in the same spectrum (inf if none)."""
    truth_tid = truth["TARGETID"]
    tz = truth["Z_DLA"]
    tn = truth["LOG_NHI"]
    order = np.argsort(truth_tid, kind="stable")
    tid_sorted = truth_tid[order]
    dv = np.full(len(cand["tid"]), np.inf)
    dlog = np.full(len(cand["tid"]), np.inf)
    for i, (tid, z) in enumerate(zip(cand["tid"], cand["z"])):
        lo = np.searchsorted(tid_sorted, tid, "left")
        hi = np.searchsorted(tid_sorted, tid, "right")
        if lo == hi:
            continue
        idx = order[lo:hi]
        d = C_KMS * np.abs(z - tz[idx]) / (1.0 + tz[idx])
        j = int(np.argmin(d))
        dv[i] = d[j]
        dlog[i] = cand["lognhi"][i] - tn[idx[j]]
    return dv, dlog


def main():
    device = resolve_device("cuda")
    print("building permissive candidate pool ...", flush=True)
    cand, ds = build_pool(device)
    truth = score_test.load_truth(TRUTH)
    n_truth = len(truth["TARGETID"])
    print("pool size = %d candidates over %d spectra | n_truth = %d"
          % (len(cand["tid"]), len(ds), n_truth))

    dv, dlog = classify(cand, truth)
    tp = dv < MATCH_DV                      # greedy match would pair these (upper bound)
    near = (dv >= MATCH_DV) & (dv < NEAR_DV)
    far = dv >= NEAR_DV
    print("\n=== Q2  the candidate pool, by distance to nearest truth ===")
    print("   TP      (|dv|<%4.0f) : %5d (%.1f%%)" % (MATCH_DV, tp.sum(), 100 * tp.mean()))
    print("   near-FP (%4.0f<=|dv|<%4.0f): %5d (%.1f%%)  <- right DLA, wrong z"
          % (MATCH_DV, NEAR_DV, near.sum(), 100 * near.mean()))
    print("   far-FP  (|dv|>=%4.0f)   : %5d (%.1f%%)  <- pure noise / no truth in line"
          % (NEAR_DV, far.sum(), 100 * far.mean()))
    print("   pool compl = %.4f   pool purity = %.4f  (if ALL pool candidates were emitted)"
          % (tp.sum() / n_truth, tp.sum() / len(dv)))

    # ---- Q1: the misses ----
    matched_truth = np.zeros(n_truth, bool)
    for i in np.flatnonzero(tp):
        # mark the truth this candidate matches (nearest, same tid)
        tid = cand["tid"][i]; z = cand["z"][i]
        idx = np.flatnonzero(truth["TARGETID"] == tid)
        if idx.size:
            d = C_KMS * np.abs(z - truth["Z_DLA"][idx]) / (1.0 + truth["Z_DLA"][idx])
            matched_truth[idx[int(np.argmin(d))]] = True
    # truth-side nearest candidate
    tdv = np.full(n_truth, np.inf)
    for i in range(n_truth):
        tid = truth["TARGETID"][i]; z = truth["Z_DLA"][i]
        idx = np.flatnonzero(cand["tid"] == tid)
        if idx.size == 0:
            continue
        tdv[i] = np.min(C_KMS * np.abs(cand["z"][idx] - z) / (1.0 + z))
    miss = ~matched_truth
    print("\n=== Q1  the %d misses (truth not matched by the pool) ===" % miss.sum())
    print("   pool HAD a candidate within  %4.0f km/s : %4d (%.1f%%)  <- SELECTION failure"
          % (MATCH_DV, (tdv[miss] < MATCH_DV).sum(), 100 * (tdv[miss] < MATCH_DV).mean()))
    print("   pool had one within %4.0f-%.0f  km/s : %4d (%.1f%%)  <- z off by ~1 px"
          % (MATCH_DV, NEAR_DV, ((tdv[miss] >= MATCH_DV) & (tdv[miss] < NEAR_DV)).sum(),
             100 * ((tdv[miss] >= MATCH_DV) & (tdv[miss] < NEAR_DV)).mean()))
    print("   no candidate within %4.0f km/s      : %4d (%.1f%%)  <- MODEL failure"
          % (NEAR_DV, (tdv[miss] >= NEAR_DV).sum(), 100 * (tdv[miss] >= NEAR_DV).mean()))

    # ---- Q3: is any per-candidate feature rankable? ----
    print("\n=== Q3  AUC of per-candidate features for TP vs FP (0.5 = useless) ===")
    def auc(score, label):
        ok = np.isfinite(score)
        s, y = score[ok], label[ok].astype(np.int64)
        if y.sum() == 0 or y.sum() == len(y):
            return np.nan
        order = np.argsort(s)
        ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        n1, n0 = y.sum(), len(y) - y.sum()
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    for feat in ("hm", "cprob", "lognhi", "snr", "offpix"):
        a = auc(cand[feat], tp)
        print("   %-8s AUC = %.4f" % (feat, a))
    comb = cand["hm"] * cand["cprob"]
    print("   hm*cprob AUC = %.4f" % auc(comb, tp))

    # ---- frontier: rank by a score, greedy top-2 per spectrum, dedup 1500 ----
    def frontier(score, name):
        order = np.argsort(-np.nan_to_num(score, nan=-1.0))
        taken = {}
        keep = []
        for i in order:
            tid = int(cand["tid"][i])
            lst = taken.setdefault(tid, [])
            if len(lst) >= 2:
                continue
            z = cand["z"][i]
            if any(C_KMS * abs(z - zz) / (1 + zz) < DEDUP_DV for zz in lst):
                continue
            lst.append(z); keep.append(i)
        keep = np.asarray(keep)
        mt = int(tp[keep].sum())
        return len(keep), mt / n_truth, mt / len(keep)

    print("\n=== compl / purity frontier (greedy top-2 per line, dedup 1500 km/s) ===")
    print("   %-14s %8s %8s %8s" % ("ranker", "n_pred", "compl", "purity"))
    for score, name in ((cand["hm"], "heatmap"), (cand["cprob"], "count_prob"),
                        (comb, "hm*count"), (-dv, "ORACLE(dv)")):
        n, c, p = frontier(score, name)
        print("   %-14s %8d %8.4f %8.4f" % (name, n, c, p))


if __name__ == "__main__":
    main()
