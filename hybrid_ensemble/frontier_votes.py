"""Is member AGREEMENT the missing selection signal?

The frontier experiment shows the pooled candidate set already contains 1255 of 1644 truths,
but the shipped pipeline only extracts 955 -- and ranking by the heatmap alone recovers just
~950 at the same budget.  So the heatmap carries no more usable ordering power.

merge_predictions.py collapses members pairwise and throws away the information "how many
members independently flagged this peak".  Here we measure whether that vote count is
predictive, and how much of the gap it closes.

Pool is cached to /tmp/pool_votes.npz after the first run.

Run:  CUDA_VISIBLE_DEVICES=0 /home/dingjch/anaconda3/envs/ML_env/bin/python frontier_votes.py
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
CACHE = "/tmp/pool_votes.npz"
TARGETS = [800, 1000, 1470, 2000, 3000, 4500, 6100]

TRANSFORMERS = [
    ("v3c", os.path.join(RUNS, "20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5",  os.path.join(RUNS, "20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6",  os.path.join(RUNS, "20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]
SOTA_CSV = os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")
MEMBERS = ["v3c", "v5", "v6", "cnn"]

POOL = dict(threshold=0.20, min_distance=10, min_z_dla=1.10, count_bias=[0.0, 3.0, 3.0],
            use_offset=True, lognhi_clip_min=19.5, lognhi_clip_max=22.5)
DEDUP_DV = 1500.0


def build_pool(device):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        print("pool loaded from cache")
        return {k: d[k] for k in d.files}
    cols = {k: [] for k in ("tid", "z", "lognhi", "hm", "cprob", "mem")}
    ds = None
    for mi, (name, ckpt) in enumerate(TRANSFORMERS):
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
                cols["mem"].append(mi)
        print("  pool[%s] done" % name, flush=True)
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
            # CNN member has no heatmap; give it a value typical of its own operating point
            cols["hm"].append(0.30); cols["cprob"].append(1.0); cols["mem"].append(3)
    out = {k: np.asarray(v) for k, v in cols.items()}
    np.savez(CACHE, **out)
    return out


def votes(cand):
    """Number of distinct members flagging this candidate (within DEDUP_DV in the same line)."""
    v = np.zeros(len(cand["tid"]), np.int32)
    order = np.lexsort((cand["z"], cand["tid"]))
    by_tid: dict[int, list[int]] = {}
    tid = cand["tid"]; z = cand["z"]; mem = cand["mem"]
    for i in order:
        by_tid.setdefault(int(tid[i]), []).append(i)
    for t, idxs in by_tid.items():
        for i in idxs:
            zs = {int(mem[j]) for j in idxs
                  if abs(C_KMS * (z[j] - z[i]) / (1 + z[i])) < DEDUP_DV}
            v[i] = len(zs)
    return v


def pick_candidates(cand, score):
    order = np.argsort(-np.nan_to_num(score, nan=-1e9), kind="stable")
    taken: dict[int, list[float]] = {}
    keep = []
    for i in order:
        t = int(cand["tid"][i])
        lst = taken.setdefault(t, [])
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
    n = len(pred["TARGETID"])
    return n, len(m) / len(truth["TARGETID"]), (len(m) / n if n else 0.0)


def main():
    device = resolve_device("cuda")
    cand = build_pool(device)
    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")
    v = votes(cand)
    print("pool = %d candidates | vote distribution: %s" %
          (len(v), {int(k): int((v == k).sum()) for k in np.unique(v)}))

    tz, ttid = truth["Z_DLA"], truth["TARGETID"]
    dvt = np.full(len(cand["tid"]), 1e9)
    for i in range(len(cand["tid"])):
        idx = np.flatnonzero(ttid == cand["tid"][i])
        if idx.size:
            dvt[i] = (C_KMS * np.abs(cand["z"][i] - tz[idx]) / (1.0 + tz[idx])).min()
    is_tp = dvt < 600.0

    def auc(s, y):
        ok = np.isfinite(s)
        s, y = s[ok], y[ok].astype(np.int64)
        o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
        n1, n0 = y.sum(), len(y) - y.sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan

    print("\n=== AUC for TP vs FP over the whole pool ===")
    for nm, s in (("votes", v.astype(float)), ("heatmap", cand["hm"]),
                  ("hm*(1+votes)", cand["hm"] * (1 + v)),
                  ("hm*cprob*(1+votes)", cand["hm"] * cand["cprob"] * (1 + v))):
        print("   %-20s AUC = %.4f" % (nm, auc(s, is_tp)))

    rankers = [("heatmap", cand["hm"]),
               ("votes", v.astype(float)),
               ("hm*(1+votes)", cand["hm"] * (1 + v)),
               ("hm*count*(1+votes)", cand["hm"] * cand["cprob"] * (1 + v)),
               ("ORACLE", -dvt)]
    curves = {}
    for nm, s in rankers:
        keep = pick_candidates(cand, s)
        pts = []
        for t in TARGETS:
            if t > len(keep):
                break
            pts.append(evaluate(cand, keep[:t], truth, snr_by_target))
        curves[nm] = pts
        print("  ranked[%s] -> %d candidates" % (nm, len(keep)), flush=True)

    print("\n=== compl / purity at matched n_pred ===")
    print("  %-8s" % "n_pred" + "".join("| %-21s" % nm for nm, _ in rankers))
    for j, t in enumerate(TARGETS):
        line = "  %-8d" % t
        for nm, _ in rankers:
            pts = curves[nm]
            line += "| %8.4f %11.4f " % pts[j][1:] if j < len(pts) else "| %8s %11s " % ("-", "-")
        print(line)
    print("\n=== shipped union reference: n_pred=1470 compl=0.5809 purity=0.6497 ===")


if __name__ == "__main__":
    main()
