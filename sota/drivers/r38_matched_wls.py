#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r38_matched_wls.py -- R38 decoder-aligned matched-readout deterministic WLS audit.

One pre-registered policy per seed: fit the 129-parameter logNHI readout
(lognhi_delta.weight/bias) by ordinary least squares on the TRAIN-split
predictions that the frozen E decoder actually selected AND that the official
matching judged correctly matched.  Equal weight per matched object; no
mask/high-logNHI/SNR/bin weighting; no ridge; no hyper-parameter sweep; no TEST.

Declared sanity tolerance (fixed BEFORE validation, same as R37): 1e-4 absolute
on logNHI.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.linalg  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

ROOT = Path("/home/heruihua/csst-dla-finder")
for _p in (str(ROOT / "src"), str(ROOT / "hybrid_ensemble"), "/home/heruihua"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from csst_dla.scoring import labels_to_truth, score_catalog, greedy_match, C_KMS  # noqa: E402
from csst_dla.targets import LYA  # noqa: E402
from decode import (decode_validation_catalog, softmax, pick_peaks,  # noqa: E402
                    pixel_to_wavelength)
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
from feature_fusion import DualFusionTrainDataset, DualTowerFusionNet  # noqa: E402
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint  # noqa: E402

SANITY_TOL = 1e-4
CAT_FIELDS = ("TARGETID", "Z_QSO", "Z_DLA", "CONFIDENCE", "SNR")
GEO_KEYS = ("center_logits", "region_logits", "offset_raw", "count_logits")
HEAD_KEYS = ("lognhi_delta.weight", "lognhi_delta.bias")
THR = 0.45
BATCH = 512
DEC_ARGS = dict(threshold=0.45, min_distance=10, min_z_dla=1.10,
                count_bias=[0.0, 0.0, 0.0], count_min_prob=0.0,
                lognhi_min=20.3, lognhi_max=22.5)


def md5_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_bytes(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


class RowView(torch.utils.data.Dataset):
    def __init__(self, ds, rows):
        self.ds = ds
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        return tuple(self.ds[r]) + (int(r),)


def set_head(net, w, b):
    net.lognhi_delta.weight.data.copy_(
        torch.as_tensor(np.asarray(w, dtype=np.float32).reshape(1, -1, 1)))
    net.lognhi_delta.bias.data.copy_(
        torch.as_tensor(np.asarray(b, dtype=np.float32).reshape(-1)))


def get_head(net):
    return (net.lognhi_delta.weight.detach().cpu().numpy().reshape(-1).copy(),
            net.lognhi_delta.bias.detach().cpu().numpy().reshape(-1).copy())


def head_from_pt(path: str):
    """Read a 129-param logNHI head from either a bare head .pt or a full model ckpt."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "lognhi_delta_weight" in ck:
        w, b = ck["lognhi_delta_weight"], ck["lognhi_delta_bias"]
    else:
        sd = ck["model_state"]
        w, b = sd["lognhi_delta.weight"], sd["lognhi_delta.bias"]
    return (w.detach().cpu().numpy().reshape(-1).astype(np.float64),
            b.detach().cpu().numpy().reshape(-1).astype(np.float64))


@torch.no_grad()
def predict_split(net, ds, device, limit=0):
    """Full frozen forward over a split -> prediction dict for the official decoder."""
    net.eval()
    heat, logn, offs, cnt, rows = [], [], [], [], []
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    sub = RowView(ds, list(range(n)))
    n_batches = 0
    for b in DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False):
        o = net(b[0].to(device), b[1].to(device), b[2].to(device))
        n_batches += 1
        heat.append(torch.sigmoid(o["center_logits"]).float().cpu().numpy())
        logn.append((20.3 + o["lognhi_raw"]).float().cpu().numpy())
        offs.append(o["offset_raw"].float().cpu().numpy())
        cnt.append(o["count_logits"].float().cpu().numpy())
        rows.append(np.asarray(b[-1], dtype=np.int64))
    return ({"heatmap": np.concatenate(heat), "lognhi": np.concatenate(logn),
             "offset": np.concatenate(offs), "count_logits": np.concatenate(cnt),
             "rows": np.concatenate(rows)}, n_batches)


def decode_with_pixels(pred, indices, labels, wavelength, **kw):
    """Verbatim replica of decode_validation_catalog that also records (row, pixel).

    Fidelity is asserted against the official function in main().
    """
    rows_meta = []
    bias = np.asarray(kw.get("count_bias", [0.0, 0.0, 0.0]), dtype=np.float32)
    count_prob = softmax(pred["count_logits"] + bias[None, :])
    threshold = kw.get("threshold", 0.45)
    min_distance = kw.get("min_distance", 10)
    min_z_dla = kw.get("min_z_dla", 1.10)
    count_min_prob = kw.get("count_min_prob", 0.0)
    lognhi_min = kw.get("lognhi_min", 20.3)
    lognhi_max = kw.get("lognhi_max", 22.5)
    for row, idx in enumerate(indices.astype(int)):
        n_pick = int(np.argmax(count_prob[row]))
        if n_pick > 0 and float(count_prob[row, n_pick]) < count_min_prob:
            n_pick = 0
        peaks = pick_peaks(pred["heatmap"][row], wavelength, float(labels["Z_QSO"][idx]),
                           n_pick=n_pick, threshold=threshold, min_distance=min_distance,
                           min_z_dla=min_z_dla)
        for pix in peaks:
            raw_log = float(pred["lognhi"][row, pix])
            lambda_dla = pixel_to_wavelength(pix + float(pred["offset"][row, pix]), wavelength)
            z_dla = float(lambda_dla / LYA - 1.0)
            lognhi_out = float(np.clip(raw_log, lognhi_min, lognhi_max))
            confidence = float(pred["heatmap"][row, pix] * count_prob[row, n_pick])
            snr = float(labels["SNR_GU"][idx])
            rows_meta.append({"TARGETID": int(idx), "Z_QSO": float(labels["Z_QSO"][idx]),
                              "Z_DLA": z_dla, "LOG_NHI": lognhi_out,
                              "CONFIDENCE": confidence, "SNR": snr,
                              "row": row, "pix": int(pix)})
    cat = {}
    for k in ("TARGETID", "Z_QSO", "Z_DLA", "LOG_NHI", "CONFIDENCE", "SNR"):
        cat[k] = np.asarray([r[k] for r in rows_meta],
                            dtype=np.int64 if k == "TARGETID" else np.float32)
    meta = [(r["row"], r["pix"]) for r in rows_meta]
    return cat, meta


@torch.no_grad()
def features_at_rows(net, ds, rows_list, device):
    """dict row -> (F (L,128) float32, base (L,) float32) via the model's own fuse/head."""
    net.eval()
    cap = {}
    h1 = net.fuse.register_forward_hook(lambda m, i, o: cap.__setitem__("shared", o.detach()))
    h2 = net.dilated_backbone.lognhi_head.register_forward_hook(
        lambda m, i, o: cap.__setitem__("lh", o.detach()))
    out, n_batches = {}, 0
    sub = RowView(ds, rows_list)
    for b in DataLoader(sub, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False):
        net(b[0].to(device), b[1].to(device), b[2].to(device))
        n_batches += 1
        sh = cap["shared"].permute(0, 2, 1).float().cpu().numpy()          # (B,L,128)
        base = (0.2 + 1.5 * cap["lh"].squeeze(1)).float().cpu().numpy()    # (B,L)
        idx = np.asarray(b[-1], dtype=np.int64)
        for j, r in enumerate(idx):
            out[int(r)] = (sh[j], base[j])
    h1.remove()
    h2.remove()
    return out, n_batches


@torch.no_grad()
def extract_allmask(net, ds, device):
    """R37-style all-mask feature/target extraction (for the distribution comparison)."""
    net.eval()
    mk_all = np.asarray(ds.mask, dtype=np.float64)
    lg_all = np.asarray(ds.lognhi, dtype=np.float64)
    rows_nz = np.nonzero((mk_all > 0).any(axis=1))[0]
    cap = {}
    h1 = net.fuse.register_forward_hook(lambda m, i, o: cap.__setitem__("shared", o.detach()))
    h2 = net.dilated_backbone.lognhi_head.register_forward_hook(
        lambda m, i, o: cap.__setitem__("lh", o.detach()))
    F, B_, Y = [], [], []
    for b in DataLoader(RowView(ds, rows_nz.tolist()), batch_size=BATCH, shuffle=False,
                        num_workers=0, pin_memory=False):
        net(b[0].to(device), b[1].to(device), b[2].to(device))
        sh = cap["shared"].permute(0, 2, 1).float().cpu().numpy()
        base = (0.2 + 1.5 * cap["lh"].squeeze(1)).float().cpu().numpy()
        idx = np.asarray(b[-1], dtype=np.int64)
        for j, r in enumerate(idx):
            cols = np.nonzero(mk_all[r] > 0)[0]
            if cols.size == 0:
                continue
            F.append(sh[j, cols, :])
            B_.append(base[j, cols])
            Y.append(lg_all[r, cols])
    h1.remove()
    h2.remove()
    return (np.concatenate(F, axis=0), np.concatenate(B_), np.concatenate(Y))


@torch.no_grad()
def eval_pass(net, loader, device):
    net.eval()
    acc = {k: [] for k in ("heatmap", "lognhi", "offset", "count_logits",
                           "center_logits", "region_logits", "offset_raw", "rows")}
    for b in loader:
        o = net(b[0].to(device), b[1].to(device), b[2].to(device))
        acc["heatmap"].append(torch.sigmoid(o["center_logits"]).float().cpu().numpy())
        acc["lognhi"].append((20.3 + o["lognhi_raw"]).float().cpu().numpy())
        acc["offset"].append(o["offset_raw"].float().cpu().numpy())
        acc["count_logits"].append(o["count_logits"].float().cpu().numpy())
        for k in ("center_logits", "region_logits", "offset_raw"):
            acc[k].append(o[k].float().cpu().numpy())
        acc["rows"].append(np.asarray(b[-1], dtype=np.int64))
    return {k: np.concatenate(v) for k, v in acc.items()}


def decode_and_score(pred, ds, wavelength, truth, thr=THR):
    cat = decode_validation_catalog(pred, ds.indices, ds.labels, wavelength, **DEC_ARGS)
    s = score_catalog(truth, cat)
    m = greedy_match(truth, cat)
    ti = np.asarray([x[0] for x in m], dtype=int)
    pi = np.asarray([x[1] for x in m], dtype=int)
    dv = C_KMS * (cat["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
    dl = cat["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
    zb, zs = float(dv.mean()), float(dv.std())
    nb, ns = float(dl.mean()), float(dl.std())
    return cat, {
        "final": float(s.final_score), "detection": float(s.detection_score),
        "parameter": float(s.parameter_score), "precision": float(s.purity),
        "recall": float(s.completeness), "n_pred": int(s.n_pred), "n_match": int(s.n_match),
        "score_z": float(np.exp(-zs / 300.0) * np.exp(-abs(zb) / 150.0)),
        "score_nhi": float(np.exp(-ns / 0.25) * np.exp(-abs(nb) / 0.1)),
        "z_signed_bias": zb, "z_std": zs,
        "lognhi_signed_bias": nb, "lognhi_abs_bias": abs(nb), "lognhi_std": ns,
        "lognhi_mae": float(np.abs(dl).mean()), "lognhi_rmse": float(np.sqrt((dl ** 2).mean())),
        "_pi": pi.tolist(), "_ti": ti.tolist(), "_bin_details": s.bin_details,
        "_fixed": {"signed_bias": nb, "abs_bias": abs(nb), "std": ns,
                   "mae": float(np.abs(dl).mean()), "rmse": float(np.sqrt((dl ** 2).mean())),
                   "score_nhi": float(np.exp(-ns / 0.25) * np.exp(-abs(nb) / 0.1)),
                   "n_matched": int(len(ti))},
    }


def resid_stats(pred, y):
    e = pred - y
    return {"mse": float((e ** 2).mean()), "signed_bias": float(e.mean()),
            "abs_bias": float(abs(e.mean())), "std": float(e.std(ddof=0)),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--e-ckpt", required=True)
    ap.add_argument("--refit-head", default=None)
    ap.add_argument("--r37-head", default=None)
    ap.add_argument("--l-ckpt", default=None)
    ap.add_argument("--r33-json", default=None)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--train-fits", required=True)
    ap.add_argument("--dilated-ckpt", required=True)
    ap.add_argument("--wzx-ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-limit", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    t_start = time.time()

    device = resolve_device(args.device)
    g, gcfg = load_checkpoint(args.dilated_ckpt, device)
    f, fcfg = load_model_from_checkpoint(args.wzx_ckpt, device)
    d_in = str(gcfg.get("input_mode", "flux"))
    w_in = str(fcfg.get("feature_mode", "flux"))
    train_ds = DualFusionTrainDataset(args.targets, args.train_fits, "train", None,
                                      dilated_input_mode=d_in, wzx_feature_mode=w_in)
    val_ds = DualFusionTrainDataset(args.targets, args.train_fits, "val", None,
                                    dilated_input_mode=d_in, wzx_feature_mode=w_in)
    wavelength = np.asarray(val_ds.wavelength, dtype=np.float64)
    truth_val = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=20.3)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0,
                            pin_memory=False)

    net = DualTowerFusionNet(g.model, f, merge_mode="residual_dilated", width=128,
                             depth=3).to(device)
    ckE = torch.load(args.e_ckpt, map_location="cpu", weights_only=False)
    miss, unexp = net.load_state_dict(ckE["model_state"], strict=True)
    assert not miss and not unexp, (miss, unexp)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    tr = [n for n, p in net.named_parameters() if n in HEAD_KEYS]
    n_tr_t = len(tr)
    n_tr_p = int(sum(p.numel() for n, p in net.named_parameters() if n in HEAD_KEYS))
    assert sorted(tr) == sorted(HEAD_KEYS) and n_tr_t == 2 and n_tr_p == 129
    n_frozen_p = int(sum(p.numel() for n, p in net.named_parameters() if n not in HEAD_KEYS))

    sc = ckE.get("score") or {}
    res = {
        "round": "R38", "seed": seed,
        "e_ckpt": args.e_ckpt, "e_ckpt_md5": md5_file(Path(args.e_ckpt)),
        "e_ckpt_epoch": sc.get("epoch"), "e_ckpt_weight_kind": sc.get("weight_kind"),
        "e_ckpt_threshold": ckE.get("threshold"),
        "dilated_ckpt_md5": md5_file(Path(args.dilated_ckpt)),
        "wzx_ckpt_md5": md5_file(Path(args.wzx_ckpt)),
        "targets_npz_md5": md5_file(Path(args.targets)),
        "decoder_config": DEC_ARGS,
        "head_only_trainable": {"names": sorted(tr), "n_tensors": n_tr_t, "n_params": n_tr_p,
                                "n_frozen_params": n_frozen_p, "pass": True},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda,
                     "threads": {k: os.environ.get(k) for k in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
    }
    with np.load(args.targets) as z:
        res["split"] = {"train_idx_md5": md5_bytes(z["train_idx"]),
                        "val_idx_md5": md5_bytes(z["val_idx"]),
                        "n_train_idx": int(z["train_idx"].size),
                        "n_val_idx": int(z["val_idx"].size),
                        "intersection": int(np.intersect1d(z["train_idx"], z["val_idx"]).size)}

    # ---------- 1) full E forward over the train split ----------
    t0 = time.time()
    pred_tr, n_batch_tr = predict_split(net, train_ds, device, args.train_limit)
    res["train_forward"] = {"batches": n_batch_tr, "seconds": time.time() - t0,
                            "train_limit": args.train_limit}
    n_idx_used = pred_tr["heatmap"].shape[0]
    idx_used = train_ds.indices[:n_idx_used]

    # ---------- 2) official decoder (E decides count gate / peaks / offset / z) ----------
    t0 = time.time()
    cat_tr, pix_meta = decode_with_pixels(pred_tr, idx_used, train_ds.labels, wavelength,
                                          **DEC_ARGS)
    cat_ref = decode_validation_catalog(pred_tr, idx_used, train_ds.labels, wavelength,
                                        **DEC_ARGS)
    dec_equal = {k: bool(np.array_equal(np.asarray(cat_tr[k]), np.asarray(cat_ref[k])))
                 for k in CAT_FIELDS + ("LOG_NHI",)}
    res["decoder_replica_check"] = {"fields_equal_to_official": dec_equal,
                                    "all_equal": bool(all(dec_equal.values())),
                                    "seconds": time.time() - t0}
    assert all(dec_equal.values()), dec_equal

    # ---------- 3) train truth + official matching ----------
    truth_tr = labels_to_truth(train_ds.labels, idx_used, min_lognhi=20.3)
    matches = greedy_match(truth_tr, cat_tr)          # (ti, pi, dv)
    ti = np.asarray([m[0] for m in matches], dtype=int)
    pi = np.asarray([m[1] for m in matches], dtype=int)
    n_truth_tr = int(len(truth_tr["TARGETID"]))
    n_pred_tr = int(len(cat_tr["TARGETID"]))
    n_match_tr = int(len(matches))
    res["train_catalog"] = {
        "train_truth_dla": n_truth_tr, "train_n_pred": n_pred_tr, "train_n_match": n_match_tr,
        "wls_sample_count": n_match_tr,
        "truth_over_pred": (n_truth_tr / n_pred_tr) if n_pred_tr else None,
        "pred_over_truth": (n_pred_tr / n_truth_tr) if n_truth_tr else None,
    }
    if n_match_tr < 130:
        res["fatal"] = "matched sample count too small for a 129-parameter fit"
        Path(out_dir / f"r38_matched_wls_seed{seed}.json").write_text(
            json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"seed": seed, "fatal": res["fatal"], "n_match": n_match_tr}), flush=True)
        return 1

    # ---------- 4) features at the matched selected pixels ----------
    rows_needed = sorted({pix_meta[p][0] for p in pi})
    t0 = time.time()
    feats, n_batch_feat = features_at_rows(net, train_ds, rows_needed, device)
    res["matched_feature_extraction"] = {"unique_spectra": len(rows_needed),
                                         "batches": n_batch_feat,
                                         "seconds": time.time() - t0}

    Fm = np.empty((n_match_tr, 128), dtype=np.float32)
    bm = np.empty(n_match_tr, dtype=np.float32)
    ym = np.empty(n_match_tr, dtype=np.float64)
    pm = np.empty(n_match_tr, dtype=np.float64)   # model logNHI at the selected pixel
    for k, p in enumerate(pi):
        row, pix = pix_meta[p]
        Fj, bj = feats[row]
        Fm[k] = Fj[pix]
        bm[k] = bj[pix]
        ym[k] = float(truth_tr["LOG_NHI"][ti[k]])
        pm[k] = float(pred_tr["lognhi"][row, pix])

    x64 = np.concatenate([Fm.astype(np.float64), np.ones((n_match_tr, 1))], axis=1)  # (n,129)
    r64 = ym - 20.3 - bm.astype(np.float64)
    wE32, bE32 = get_head(net)
    theta0 = np.concatenate([wE32.astype(np.float64), bE32.astype(np.float64)])
    res["design_matrix"] = {"shape": list(x64.shape),
                            "rows_equal_train_n_match": bool(x64.shape[0] == n_match_tr),
                            "cols": int(x64.shape[1]),
                            "x_dtype": str(x64.dtype), "finite": bool(np.isfinite(x64).all()),
                            "finite_targets": bool(np.isfinite(ym).all()),
                            "finite_base": bool(np.isfinite(bm).all())}

    # ---------- 5) hand-readout sanity ----------
    pred_E = 20.3 + bm.astype(np.float64) + x64 @ theta0
    d_np64 = float(np.abs(pred_E - pm).max())
    x32 = np.concatenate([Fm, np.ones((n_match_tr, 1), dtype=np.float32)], axis=1)
    pred_E32 = (np.float32(20.3) + bm + (Fm @ wE32 + np.float32(bE32[0]))).astype(np.float64)
    d_torch = float(np.abs(pred_E32 - pm).max())
    res["sanity_readout"] = {
        "declared_tolerance": SANITY_TOL,
        "formula": "20.3 + base_j + x_j @ theta_E at the E-selected pixel",
        "max_abs_diff_numpy64_vs_official": d_np64,
        "max_abs_diff_torch_f32_vs_official": d_torch,
        "within_declared_tol": bool(max(d_np64, d_torch) <= SANITY_TOL),
    }

    # ---------- 6) the single pre-registered solve ----------
    A = x64
    t_vec = r64
    d_vec = t_vec - A @ theta0
    cond = float(max(A.shape) * np.finfo(np.float64).eps)
    t0 = time.time()
    delta, residues, rank, sv = scipy.linalg.lstsq(A, d_vec, cond=cond,
                                                   lapack_driver="gelsd", check_finite=True)
    fit_seconds = time.time() - t0
    theta_star = theta0 + delta
    delta2, _, rank2, _ = scipy.linalg.lstsq(A, d_vec, cond=cond, lapack_driver="gelsd",
                                             check_finite=True)

    def J_match(theta, A_=A, t_=t_vec):
        return float(((A_ @ np.asarray(theta, dtype=np.float64) - t_) ** 2).mean())

    J_E = J_match(theta0)
    J_R38 = J_match(theta_star)
    resid_vec = A @ delta - d_vec
    gvec = A.T @ resid_vec
    s_sorted = np.sort(np.asarray(sv, dtype=np.float64))[::-1]
    smax, smin = float(s_sorted[0]), float(s_sorted[-1])
    res["solver"] = {
        "call": "scipy.linalg.lstsq(A, d, cond=max(A.shape)*eps64, lapack_driver='gelsd')",
        "cond": cond, "dtype": "float64", "cpu_fit_seconds": fit_seconds,
        "rank": int(rank), "n_singular_values": int(s_sorted.size),
        "full_column_rank": bool(rank == A.shape[1]),
        "s_max": smax, "s_min": smin,
        "effective_cond_smax_over_s_rank": float(smax / s_sorted[rank - 1]),
        "singular_values": [float(v) for v in s_sorted],
        "theta0_norm": float(np.linalg.norm(theta0)),
        "theta_star_norm": float(np.linalg.norm(theta_star)),
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_w_norm": float(np.linalg.norm(delta[:128])),
        "delta_b_abs": float(abs(delta[128])),
        "stationarity": {"abs_norm_AT_resid": float(np.linalg.norm(gvec)),
                         "inf_norm_AT_resid": float(np.abs(gvec).max()),
                         "rel_vs_AT_d": float(np.linalg.norm(gvec) / (np.linalg.norm(A.T @ d_vec) + 1e-300)),
                         "residual_norm": float(np.linalg.norm(resid_vec)),
                         "truncation_used": bool(rank < A.shape[1])},
        "repeatability": {"bitwise_identical": bool(np.array_equal(delta, delta2)),
                          "max_abs_delta_diff": float(np.abs(delta - delta2).max()),
                          "rank2": int(rank2)},
    }

    # ---------- 7) train matched-object objective across the four heads ----------
    heads = {"E": theta0, "R38_matched_WLS": theta_star}
    paths = {}
    for name, path in (("AdamW_refit_ep10", args.refit_head), ("R37_allmask_WLS", args.r37_head),
                       ("L_ema_ep40", args.l_ckpt)):
        if path and Path(path).exists():
            w, b = head_from_pt(path)
            heads[name] = np.concatenate([w, b])
            paths[name] = {"path": path, "present": True, "md5": md5_file(Path(path))}
        else:
            paths[name] = {"path": path, "present": False}
    res["aux_heads"] = paths
    res["train_matched_objective"] = {
        name: {"J_match_mean_sq": J_match(th), **resid_stats(20.3 + bm.astype(np.float64) + A @ th, ym)}
        for name, th in heads.items()}
    J_refit = res["train_matched_objective"].get("AdamW_refit_ep10", {}).get("J_match_mean_sq")
    J_r37 = res["train_matched_objective"].get("R37_allmask_WLS", {}).get("J_match_mean_sq")

    # ---------- 8) write back to deployment dtype ----------
    set_head(net, theta_star[:128], theta_star[128:])
    w_dep, b_dep = get_head(net)
    theta_dep = np.concatenate([w_dep.astype(np.float64), b_dep.astype(np.float64)])
    res["deployment"] = {
        "head_dtype_written": "float32",
        "head_rounding_max_abs": float(np.abs(theta_star - theta_dep).max()),
        "J_match_f32_arithmetic": float(((x32 @ theta_dep.astype(np.float32)
                                          - r64.astype(np.float32)).astype(np.float64) ** 2).mean()),
        "J_match_f64_on_rounded_params": J_match(theta_dep),
        "prediction_max_abs_diff_f64_vs_rounded": float(np.abs(A @ theta_star - A @ theta_dep).max()),
    }

    # ---------- 9) distribution-shift diagnostic: all-mask vs matched ----------
    t0 = time.time()
    F_all, b_all, y_all = extract_allmask(net, train_ds, device)
    res["allmask_extraction"] = {"n_pixels": int(F_all.shape[0]), "seconds": time.time() - t0}
    mu_m, mu_a = Fm.astype(np.float64).mean(0), F_all.astype(np.float64).mean(0)
    v_m, v_a = Fm.astype(np.float64).var(0), F_all.astype(np.float64).var(0)
    smd = np.abs(mu_m - mu_a) / np.sqrt(np.maximum((v_m + v_a) / 2.0, 1e-30))
    res["distribution_shift"] = {
        "feature_abs_smd_mean": float(smd.mean()), "median": float(np.median(smd)),
        "max": float(smd.max()), "argmax_dim": int(np.argmax(smd)),
        "n_features": int(smd.size),
        "base_lognhi": {"matched_mean": float(bm.mean()), "matched_std": float(bm.std()),
                        "allmask_mean": float(b_all.mean()), "allmask_std": float(b_all.std()),
                        "matched_q": [float(v) for v in np.percentile(bm, [5, 25, 50, 75, 95])],
                        "allmask_q": [float(v) for v in np.percentile(b_all, [5, 25, 50, 75, 95])]},
        "truth_lognhi": {"matched_mean": float(ym.mean()), "matched_std": float(ym.std()),
                         "allmask_mean": float(y_all.mean()), "allmask_std": float(y_all.std()),
                         "matched_q": [float(v) for v in np.percentile(ym, [5, 25, 50, 75, 95])],
                         "allmask_q": [float(v) for v in np.percentile(y_all, [5, 25, 50, 75, 95])]},
        "note": "descriptive only; no sample weights were derived from these statistics",
    }

    # ---------- 10) parameter distance ----------
    def l2(u, v):
        return float(np.linalg.norm(u - v))

    def cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))

    dist = {}
    for name, th in heads.items():
        if name in ("E", "R38_matched_WLS"):
            continue
        dist[f"l2_R38_minus_{name}"] = l2(theta_star, th)
        dist[f"l2_E_minus_{name}"] = l2(theta0, th)
        dist[f"cosine_weight_R38_vs_{name}"] = cos(theta_star[:128], th[:128])
    res["parameter_distance"] = dist

    # ---------- 11) validation (never used for fitting) ----------
    def val_preds():
        ep = eval_pass(net, val_loader, device)
        cat, m = decode_and_score({"heatmap": ep["heatmap"], "offset": ep["offset"],
                                   "count_logits": ep["count_logits"], "lognhi": ep["lognhi"]},
                                  val_ds, wavelength, truth_val)
        return ep, cat, m

    set_head(net, wE32, bE32)
    epE, catE, mE = val_preds()
    res["baseline_E_E"] = {k: v for k, v in mE.items() if not k.startswith("_")}
    ref = {"42": 0.6697988593920631, "43": 0.6753101573065918, "44": 0.6769657906469371}[seed]
    res["baseline_reproduced"] = {"reference": ref, "here": mE["final"],
                                  "pass": abs(mE["final"] - ref) < 1e-12}
    geo_base = {k: epE[k].copy() for k in GEO_KEYS}

    cat_store = {}

    def one_head_eval(th, label):
        set_head(net, th[:128], th[128:])
        ep, cat, m = val_preds()
        cat_store[label] = cat
        geo = {k: bool(np.array_equal(ep[k], geo_base[k])) for k in GEO_KEYS}
        return {
            "label": label,
            "metrics": {k: v for k, v in m.items() if not k.startswith("_")},
            "Delta_vs_E": m["final"] - mE["final"],
            "geometry_bitwise_identical": geo,
            "all_geometry_identical": bool(all(geo.values())),
            "rows_identical": bool(np.array_equal(ep["rows"], epE["rows"])),
            "catalog_fields_identical": {f: bool(np.array_equal(np.asarray(catE[f]),
                                                                np.asarray(cat[f])))
                                         for f in CAT_FIELDS},
            "matching_pairs_equal": bool(m["_pi"] == mE["_pi"] and m["_ti"] == mE["_ti"]),
            "n_pred_nmatch_P_R_scorez_identical": bool(
                m["n_pred"] == mE["n_pred"] and m["n_match"] == mE["n_match"]
                and m["precision"] == mE["precision"] and m["recall"] == mE["recall"]
                and m["score_z"] == mE["score_z"]),
            "fixed_set": m["_fixed"],
            "final_identity_0.6D_0.4P": 0.6 * m["detection"] + 0.4 * m["parameter"],
            "parameter_identity_0.5_z_nhi": 0.5 * m["score_z"] + 0.5 * m["score_nhi"],
        }

    res["val_heads"] = {}
    for name in ("AdamW_refit_ep10", "R37_allmask_WLS", "R38_matched_WLS"):
        if name in heads:
            res["val_heads"][name] = one_head_eval(heads[name], name)
    res["baseline_identity_0.6D_0.4P"] = 0.6 * mE["detection"] + 0.4 * mE["parameter"]

    mW = res["val_heads"]["R38_matched_WLS"]["metrics"]
    res["main"] = {"Final_E_E": mE["final"], "Final_R38_matched_WLS": mW["final"],
                   "Delta_matchWLS_vs_E": mW["final"] - mE["final"]}

    # ---------- 12) R33 full-late donor (auxiliary, s42/s43 only) ----------
    if args.r33_json and Path(args.r33_json).exists():
        r33 = json.loads(Path(args.r33_json).read_text(encoding="utf-8"))
        fL = r33["fixed_set_residuals"]["E"]
        res["r33_donor"] = {"signed_bias": fL["lognhi_signed_bias_B"],
                            "abs_bias": abs(fL["lognhi_signed_bias_B"]),
                            "std": fL["lognhi_std_B"], "mae": fL["lognhi_mae_B"],
                            "rmse": fL["lognhi_rmse_B"], "score_nhi": fL["score_nhi_B"],
                            "n_matched": fL["n_matched_fixed"]}
        res["r33_crosscheck"] = {"here_E_E_final": mE["final"],
                                 "r33_E_E_final": r33["cells"]["E/E"]["final"],
                                 "match": abs(mE["final"] - r33["cells"]["E/E"]["final"]) == 0.0}
    else:
        res["r33_donor"] = None

    # ---------- 13) gates ----------
    v38 = res["val_heads"]["R38_matched_WLS"]
    gates = {
        "head_only_trainable": True,
        "decoder_replica_matches_official": bool(res["decoder_replica_check"]["all_equal"]),
        "sanity_within_declared_tol": bool(res["sanity_readout"]["within_declared_tol"]),
        "design_rows_eq_train_n_match": bool(res["design_matrix"]["rows_equal_train_n_match"]),
        "design_cols_eq_129": bool(res["design_matrix"]["cols"] == 129),
        "J_match_WLS_le_J_E": bool(J_R38 <= J_E + 1e-12),
        "J_match_WLS_le_J_refit": (None if J_refit is None else bool(J_R38 <= J_refit + 1e-12)),
        "J_match_WLS_le_J_R37": (None if J_r37 is None else bool(J_R38 <= J_r37 + 1e-12)),
        "solve_repeatable": bool(res["solver"]["repeatability"]["bitwise_identical"]),
        "baseline_reproduced": bool(res["baseline_reproduced"]["pass"]),
        "geometry_all_identical": bool(v38["all_geometry_identical"]),
        "rows_identical": bool(v38["rows_identical"]),
        "catalog_fields_identical": bool(all(v38["catalog_fields_identical"].values())),
        "matching_pairs_equal": bool(v38["matching_pairs_equal"]),
        "n_pred_nmatch_P_R_scorez_identical": bool(v38["n_pred_nmatch_P_R_scorez_identical"]),
    }

    # ---------- 14) products ----------
    np.savez_compressed(out_dir / f"r38_catalogs_seed{seed}.npz",
                        **{f"E_E_{f}": catE[f] for f in CAT_FIELDS + ("LOG_NHI",)},
                        **{f"R38_{f}": cat_store["R38_matched_WLS"][f]
                           for f in CAT_FIELDS + ("LOG_NHI",)})
    dep_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"model_state": dep_state,
                "score": {"epoch": "head_matched_wls_on_frozen_ema_ep10",
                          "weight_kind": "head_matched_wls_on_frozen_ema_ep10"},
                "config": ckE.get("config"), "threshold": ckE.get("threshold"),
                "base_e_ckpt": args.e_ckpt, "seed": seed},
               out_dir / f"r38_deployable_seed{seed}.pt")
    torch.save({"lognhi_delta_weight": torch.as_tensor(theta_star[:128], dtype=torch.float64).reshape(1, 128, 1),
                "lognhi_delta_bias": torch.as_tensor(theta_star[128:], dtype=torch.float64),
                "lognhi_delta_weight_f32": torch.as_tensor(w_dep, dtype=torch.float32).reshape(1, 128, 1),
                "lognhi_delta_bias_f32": torch.as_tensor(b_dep, dtype=torch.float32),
                "seed": seed, "weight_kind": "head_matched_wls_on_frozen_ema_ep10"},
               out_dir / f"r38_matched_wls_head_seed{seed}.pt")
    ck_dep = torch.load(out_dir / f"r38_deployable_seed{seed}.pt", map_location="cpu",
                        weights_only=False)
    sd = ck_dep["model_state"]
    gates["deployable_reload_ok"] = bool(
        torch.equal(sd["lognhi_delta.weight"], net.lognhi_delta.weight.detach().cpu())
        and torch.equal(sd["lognhi_delta.bias"], net.lognhi_delta.bias.detach().cpu())
        and all(torch.equal(sd[n].cpu(), ckE["model_state"][n].cpu())
                for n in sd if n not in HEAD_KEYS))
    gates["all_pass"] = all(v is True for v in gates.values() if v is not None)
    res["gates"] = gates
    res["fit_scope"] = {"design_source": "train split, E-decoder-selected matched predictions only",
                        "objective": "mean_j (x_j theta - r_j)^2, equal weight per matched object",
                        "no_mask_or_high_lognhi_weighting": True,
                        "val_used_in_fit": False, "test_used": False,
                        "cond_rule_fixed_before_validation": True, "no_sweep": True}
    res["cost"] = {"cpu_fit_seconds": fit_seconds, "wall_seconds": time.time() - t_start,
                   "train_forward_batches": n_batch_tr,
                   "matched_feature_batches": n_batch_feat,
                   "gpu_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
                   "process_peak_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    res["note"] = ("one pre-registered policy: train-split decoder-matched selected-pixel OLS; "
                   "train truth used only after prediction, for matching and targets; "
                   "no TEST, no validation calibration, no hyper-parameter sweep")

    Path(out_dir / f"r38_matched_wls_seed{seed}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "seed": seed, "n_truth_train": n_truth_tr, "n_pred_train": n_pred_tr,
        "n_match_train": n_match_tr, "matrix": list(x64.shape), "rank": int(rank),
        "J_E": J_E, "J_refit": J_refit, "J_R37": J_r37, "J_R38": J_R38,
        "E_Final": mE["final"],
        "R38_Final": mW["final"], "Delta_matchWLS_vs_E": mW["final"] - mE["final"],
        "R37_Final": (res["val_heads"].get("R37_allmask_WLS") or {}).get("metrics", {}).get("final"),
        "AdamW_Final": (res["val_heads"].get("AdamW_refit_ep10") or {}).get("metrics", {}).get("final"),
        "gates_all_pass": gates["all_pass"],
        "failing": {k: v for k, v in gates.items() if v is not True},
        "smd_mean": res["distribution_shift"]["feature_abs_smd_mean"],
    }, default=str), flush=True)
    print("wrote", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
