"""compl / purity frontier: how much of the pool ceiling can a ranker capture?

The permissive pool (4 members, thr 0.20, count gate wide open) contains 1255 of the 1644
truths, but the shipped union only recovers 955.  That gap is a *selection* gap.  This
script walks candidates in rank order (per-line cap 2, dedup 1500 km/s -- the same
structure as the shipped pipeline) and reports compl / purity at matched n_pred.

Rankers compared:
   lognhi   <- what merge_predictions.py actually does (top-2 by predicted logNHI)
   hm       <- peak heatmap value
   hm*cprob <- heatmap x count-head probability
   oracle   <- |dv| to truth (upper bound, diagnostic only, NOT usable for submission)

Run:  CUDA_VISIBLE_DEVICES=0 /home/dingjch/anaconda3/envs/ML_env/bin/python frontier_compl_purity.py
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
from csst_dla.scoring import greedy_match, C_KMS            # noqa: E402

TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
RUNS = os.path.expanduser("~/csst_dla_runs")
LYA = 1215.67
TARGETS = [800, 1000, 1200, 1470, 1800, 2200, 2800, 3600, 5000, 7000]

TRANSFORMERS = [
    ("v3c", os.path.join(RUNS, "20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5",  os.path.join(RUNS, "20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6",  os.path.join(RUNS, "20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]
SOTA_CSV = os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")

POOL = dict(threshold=0.20, min_distance=10, min_z_dla=1.10, count_bias=[0.0, 3.0, 3.0],
            use_offset=True, lognhi_clip_min=19.5, lognhi_clip_max=22.5)
DEDUP_DV = 1500.0


def build_pool(device):
    cols = {k: [] for k in ("tid", "z", "lognhi", "hm", "cprob")}
    ds = None
    for name, ckpt in TRANSFORMERS:
        model, cfg = load_checkpoint(ckpt, device)
        ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pred = predict_member(model, loader, device)
        cp = softmax(pred["count_logits"] + np.asarray(POOL["count_bias"], np.float32)[None, :])
        n_pick = np.argmax(cp, axis=1)
        for r, row in enumerate(pred["rows"]):
            hm = pred["heatmap"][r]
            peaks = pick_peaks(hm, ds.wavelength, float(ds.zq[row]), n_pick=int(n_pick[r]),
                               threshold=POOL["threshold"], min_distance=POOL["min_distance"],
                               min_z_dla=POOL["min_z_dla"])
            for pix in peaks:
                lam = pixel_to_wavelength(pix + float(pred["offset"][r, pix]), ds.wavelength)
                cols["tid"].append(int(ds.targetid[row]))
                cols["z"].append(float(lam / LYA - 1.0))
                cols["lognhi"].append(float(np.clip(pred["lognhi"][r, pix],
                                                    POOL["lognhi_clip_min"], POOL["lognhi_clip_max"])))
                cols["hm"].append(float(hm[pix]))
                cols["cprob"].append(float(cp[r, int(n_pick[r])]))
        print("  pool[%s]: %d candidates so far" % (name, len(cols["tid"])), flush=True)
        del model
        torch.cuda.empty_cache()
    tid2row = {int(t): i for i, t in enumerate(ds.targetid)}
    cat = np.genfromtxt(SOTA_CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if cat.ndim == 0:
        cat = np.asarray([cat])
    for row in cat:
        tid = int(row["id"])
        for slot in (1, 2):
            z = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
            if z <= 0:
                continue
            cols["tid"].append(tid); cols["z"].append(z); cols["lognhi"].append(ln)
            cols["hm"].append(0.45); cols["cprob"].append(1.0)
    return {k: np.asarray(v) for k, v in cols.items()}, ds


def pick_candidates(cand, score):
    """Walk in rank order, per-line cap 2, dedup 1500 km/s. Return accepted index list."""
    order = np.argsort(-np.nan_to_num(score, nan=-1e9), kind="stable")
    taken: dict[int, list[float]] = {}
    keep = []
    for i in order:
        tid = int(cand["tid"][i])
        lst = taken.setdefault(tid, [])
        if len(lst) >= 2:
            continue
        z = cand["z"][i]
        if any(C_KMS * abs(z - zz) / (1.0 + zz) < DEDUP_DV for zz in lst):
            continue
        lst.append(z)
        keep.append(i)
    return np.asarray(keep)


def evaluate(cand, keep, truth, snr_by_target):
    pred = {"TARGETID": cand["tid"][keep].astype(np.int64),
            "Z_DLA": cand["z"][keep].astype(np.float32),
            "LOG_NHI": cand["lognhi"][keep].astype(np.float32)}
    pred["SNR"] = np.asarray([snr_by_target.get(int(t), np.nan) for t in pred["TARGETID"]],
                             np.float32)
    m = greedy_match(truth, pred)
    n_pred = len(pred["TARGETID"])
    return n_pred, len(m) / len(truth["TARGETID"]), (len(m) / n_pred if n_pred else 0.0), len(m)


def main():
    device = resolve_device("cuda")
    print("building permissive candidate pool ...", flush=True)
    cand, ds = build_pool(device)
    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    print("pool = %d candidates | n_truth = %d" % (len(cand["tid"]), len(truth["TARGETID"])))

    # dv to nearest truth, per candidate (oracle ranker only)
    tz, tn = truth["Z_DLA"], truth["TARGETID"]
    dv = np.full(len(cand["tid"]), 1e9)
    for i in range(len(cand["tid"])):
        idx = np.flatnonzero(tn == cand["tid"][i])
        if idx.size:
            d = C_KMS * np.abs(cand["z"][i] - tz[idx]) / (1.0 + tz[idx])
            dv[i] = d.min()

    rankers = [
        ("lognhi (shipped)", cand["lognhi"]),
        ("heatmap", cand["hm"]),
        ("hm*count", cand["hm"] * cand["cprob"]),
        ("ORACLE |dv|", -dv),
    ]
    curves = {}
    for name, sc in rankers:
        keep = pick_candidates(cand, sc)
        pts = []
        for t in TARGETS:
            if t > len(keep):
                break
            n, c, p, mt = evaluate(cand, keep[:t], truth, snr_by_target)
            pts.append((n, c, p, mt))
        curves[name] = pts
        print("  %-18s pool captured: %d candidates" % (name, len(keep)), flush=True)

    print("\n=== compl / purity at matched n_pred ===")
    hdr = "  %-12s" % "n_pred"
    for name, _ in rankers:
        hdr += "| %-22s" % name
    print(hdr)
    print("  %-12s" % "" + "".join("| %9s %11s " % ("compl", "purity") for _ in rankers))
    for j, t in enumerate(TARGETS):
        line = "  %-12d" % t
        any_ = False
        for name, _ in rankers:
            pts = curves[name]
            if j < len(pts):
                _, c, p, _ = pts[j]
                line += "| %9.4f %11.4f " % (c, p)
                any_ = True
            else:
                line += "| %9s %11s " % ("-", "-")
        if any_:
            print(line)

    print("\n=== the shipped union for reference ===")
    union = os.path.join(RUNS, "20260912/ensemble_4way_v3/predictions.csv")
    cat = np.genfromtxt(union, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if cat.ndim == 0:
        cat = np.asarray([cat])
    c2 = {k: [] for k in ("tid", "z", "lognhi", "hm", "cprob")}
    for row in cat:
        tid = int(row["id"])
        for slot in (1, 2):
            z = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
            if z <= 0:
                continue
            c2["tid"].append(tid); c2["z"].append(z); c2["lognhi"].append(ln)
            c2["hm"].append(0.0); c2["cprob"].append(0.0)
    c2 = {k: np.asarray(v) for k, v in c2.items()}
    n, c, p, mt = evaluate(c2, np.arange(len(c2["tid"])), truth, snr_by_target)
    print("   shipped union: n_pred=%d compl=%.4f purity=%.4f n_match=%d" % (n, c, p, mt))
    print("   pool ceiling (ORACLE row at n_pred~%d): compl=%.4f purity=%.4f"
          % (curves["ORACLE |dv|"][-1][0], curves["ORACLE |dv|"][-1][1],
             curves["ORACLE |dv|"][-1][2]))
    print("   of the pool's achievable %d truths, shipped recovers %d" %
          (int(curves["ORACLE |dv|"][-1][1] * len(truth["TARGETID"])), mt))


if __name__ == "__main__":
    main()
