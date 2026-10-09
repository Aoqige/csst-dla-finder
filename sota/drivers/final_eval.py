#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""final_eval.py -- FINAL: frozen final-method manifest + one-shot TEST evaluation.

Stages (run in this order, each a separate process invocation):
  manifest : write final_method_manifest.json + .sha256 BEFORE any TEST access
  valcheck : reload each final deployment checkpoint and reproduce the frozen
             unified-validation Final endpoints; abort if they do not reproduce
  predict  : run each frozen deployment checkpoint on TEST exactly once and freeze
             the prediction artifacts (hash them).  Reads NO DLA label column.
  score    : read the frozen prediction artifacts + TEST truth and score them.

No TEST-specific tuning of any kind.
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
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402
from astropy.io import fits  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

REPO = Path("/home/heruihua/csst-dla-finder")
for _p in (str(REPO / "src"), str(REPO / "hybrid_ensemble"), "/home/heruihua"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from csst_dla.scoring import labels_to_truth, score_catalog, greedy_match, C_KMS  # noqa: E402
from decode import decode_validation_catalog  # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
from feature_fusion import (DualFusionTrainDataset, DualFusionTestDataset,  # noqa: E402
                            DualTowerFusionNet)
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint  # noqa: E402

B = Path("/home/heruihua/csst_dla_runs")
R12 = B / "20261003_r12"
R14 = B / "20261003_r14"
OUT = B / "20261005_final"
TRAIN_FITS = "/data/heruihua/newer/train_500k_GU_qlf.fits"
TEST_FITS = "/data/heruihua/newer/test_100k_GU.fits"
TEST_TRUTH = "/data/heruihua/newer/test_truth_100k_GU.fits"
NPZ = str(R12 / "cnn_targets_unified_seed42_sig15.npz")
DCK = str(R12 / "tower_grow_ctrl_u/model.pt")
WCK = str(R14 / "tower_flat_cons_u_r14/best_model.pt")

DEC_ARGS = dict(threshold=0.45, min_distance=10, min_z_dla=1.10,
                count_bias=[0.0, 0.0, 0.0], count_min_prob=0.0,
                lognhi_min=20.3, lognhi_max=22.5)
BATCH = 512
CAT_FIELDS = ("TARGETID", "Z_QSO", "Z_DLA", "CONFIDENCE", "SNR")

SEEDS = {
    "42": {"e": B / "20261004_r32/lr_step10_seed42/ema_ep10.pt",
           "dep": B / "20261005_r38/r38_deployable_seed42.pt",
           "val_final": 0.6825514434, "source": "R38"},
    "43": {"e": B / "20261004_r32/lr_step10_seed43/ema_ep10.pt",
           "dep": B / "20261005_r38/r38_deployable_seed43.pt",
           "val_final": 0.6831270401, "source": "R38"},
    "44": {"e": B / "20261004_r36/replay_seed44/ema_ep10.pt",
           "dep": B / "20261005_r38/r38_deployable_seed44.pt",
           "val_final": 0.6829074029, "source": "R38"},
    "45": {"e": B / "20261005_r39/seed45/ema_ep10.pt",
           "dep": B / "20261005_r39/r39_deployable_seed45.pt",
           "val_final": 0.6870866739, "source": "R39"},
}
SEED_ORDER = ("42", "43", "44", "45")
PREDECLARED_COMPANION = "45"

SOURCE_FILES = {
    "training_driver": "/home/heruihua/r32/run_fusion_ema_lrstep.py",
    "ema_implementation": "/home/heruihua/r30/ema.py",
    "matched_wls_script_seed42_44": "/home/heruihua/r38/r38_matched_wls.py",
    "matched_wls_script_seed45": "/home/heruihua/r39/r39_matched_wls.py",
    "decoder": str(REPO / "hybrid_ensemble/decode.py"),
    "scorer": str(REPO / "src/csst_dla/scoring.py"),
    "feature_fusion": str(REPO / "hybrid_ensemble/feature_fusion.py"),
    "data_module": str(REPO / "hybrid_ensemble/data.py"),
    "evaluate_hybrid": str(REPO / "hybrid_ensemble/evaluate_hybrid.py"),
    "targets_maker": str(REPO / "scripts/make_cnn_targets.py"),
    "unified_split_maker": "/home/heruihua/r12/make_unified_split.py",
    "final_eval_script": "/home/heruihua/final/final_eval.py",
}


def md5_file(p, chunk=1 << 20):
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_bytes(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def build_net(device):
    g, gcfg = load_checkpoint(DCK, device)
    f, fcfg = load_model_from_checkpoint(WCK, device)
    net = DualTowerFusionNet(g.model, f, merge_mode="residual_dilated", width=128,
                             depth=3).to(device)
    return net, str(gcfg.get("input_mode", "flux")), str(fcfg.get("feature_mode", "flux"))


def load_deployment(net, path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    miss, unexp = net.load_state_dict(ck["model_state"], strict=True)
    assert not miss and not unexp, (miss, unexp)
    net.eval()
    return ck


@torch.no_grad()
def eval_pass(net, loader, device):
    net.eval()
    acc = {k: [] for k in ("heatmap", "lognhi", "offset", "count_logits", "rows")}
    for b in loader:
        o = net(b[0].to(device), b[1].to(device), b[2].to(device))
        acc["heatmap"].append(torch.sigmoid(o["center_logits"]).float().cpu().numpy())
        acc["lognhi"].append((20.3 + o["lognhi_raw"]).float().cpu().numpy())
        acc["offset"].append(o["offset_raw"].float().cpu().numpy())
        acc["count_logits"].append(o["count_logits"].float().cpu().numpy())
        rows = b[-1]
        acc["rows"].append(np.asarray(rows, dtype=np.int64) if not torch.is_tensor(rows)
                           else rows.numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def metrics_from(cat, truth):
    s = score_catalog(truth, cat)
    m = greedy_match(truth, cat)
    ti = np.asarray([x[0] for x in m], dtype=int)
    pi = np.asarray([x[1] for x in m], dtype=int)
    dv = C_KMS * (cat["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
    dl = cat["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
    zb, zs = float(dv.mean()), float(dv.std())
    nb, ns = float(dl.mean()), float(dl.std())
    return {
        "final": float(s.final_score), "detection": float(s.detection_score),
        "parameter": float(s.parameter_score), "precision": float(s.purity),
        "recall": float(s.completeness), "n_pred": int(s.n_pred), "n_match": int(s.n_match),
        "n_truth": int(len(truth["TARGETID"])),
        "score_z": float(np.exp(-zs / 300.0) * np.exp(-abs(zb) / 150.0)),
        "score_nhi": float(np.exp(-ns / 0.25) * np.exp(-abs(nb) / 0.1)),
        "z_signed_bias": zb, "z_abs_bias": abs(zb), "z_std": zs,
        "z_mae": float(np.abs(dv).mean()), "z_rmse": float(np.sqrt((dv ** 2).mean())),
        "lognhi_signed_bias": nb, "lognhi_abs_bias": abs(nb), "lognhi_std": ns,
        "lognhi_mae": float(np.abs(dl).mean()), "lognhi_rmse": float(np.sqrt((dl ** 2).mean())),
        "final_identity_0.6D_0.4P": 0.6 * float(s.detection_score) + 0.4 * float(s.parameter_score),
        "parameter_identity_0.5_z_nhi": 0.5 * float(np.exp(-zs / 300.0) * np.exp(-abs(zb) / 150.0))
                                        + 0.5 * float(np.exp(-ns / 0.25) * np.exp(-abs(nb) / 0.1)),
    }


def read_truth_hdu(path, hdu):
    with fits.open(path) as h:
        d = h[hdu].data
        return {c: np.asarray(d[c]) for c in d.columns.names}


# --------------------------------------------------------------------------- stages
def stage_manifest(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with np.load(NPZ) as z:
        split = {"npz": NPZ, "npz_md5": md5_file(NPZ),
                 "train_idx_md5": md5_bytes(z["train_idx"]),
                 "val_idx_md5": md5_bytes(z["val_idx"]),
                 "n_train": int(z["train_idx"].size), "n_val": int(z["val_idx"].size),
                 "intersection": int(np.intersect1d(z["train_idx"], z["val_idx"]).size)}
    man = {
        "round": "FINAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_local": datetime.now().isoformat(),
        "created_epoch": time.time(),
        "frozen_before_test": True,
        "method": {
            "stage_A": {
                "backbones": "GrowNet conservative + FlatNet conservative, both frozen + eval",
                "merge_mode": "residual_dilated", "fusion_width": 128, "fusion_depth": 3,
                "loss_weights": {"region": 0.20, "lognhi": 0.05, "offset": 0.10, "count": 0.25},
                "count_loss": "ce (unweighted)", "count_gradient_path": "shared (no stop-gradient)",
                "optimizer": {"class": "AdamW", "lr": 5e-4, "weight_decay": 1e-4},
                "batch_size": 512, "epochs": 10, "scheduler": None,
                "ema": {"beta": 0.999, "update": "after every optimizer.step()",
                        "warmup": None, "bias_correction": None, "reset": None,
                        "n_ema_tensors": 40, "n_ema_params": 553479},
                "checkpoint": "fixed EMA epoch10",
            },
            "stage_B": {
                "frozen": "E (all parameters, buffers, eval state)",
                "sample": "unified TRAIN split; E-decoder-selected predictions that the official greedy_match judged matched",
                "objective": "J_match(theta) = mean_j (x_j theta - r_j)^2, equal weight per matched object",
                "solver": "scipy.linalg.lstsq(A, d, cond=max(A.shape)*eps64, lapack_driver='gelsd') CPU float64",
                "parametrisation": "theta = theta_E + delta (minimum-norm delta)",
                "fit_parameters": 129, "ridge": None, "other_weighting": None, "calibration": None,
            },
        },
        "towers": {"dilated": {"path": DCK, "md5": md5_file(DCK)},
                   "wzx": {"path": WCK, "md5": md5_file(WCK)}},
        "targets_split": split,
        "train_fits": {"path": TRAIN_FITS, "md5": md5_file(TRAIN_FITS)},
        "decoder": dict(DEC_ARGS),
        "seeds": {},
        "predeclared_single_model_companion": PREDECLARED_COMPANION,
        "test_data": {"fits": TEST_FITS, "md5": md5_file(TEST_FITS),
                      "truth_fits": TEST_TRUTH, "md5_truth": md5_file(TEST_TRUTH)},
        "source_hashes": {k: {"path": v, "md5": md5_file(v), "present": Path(v).exists()}
                          for k, v in SOURCE_FILES.items()},
        "forbidden_after_test": [
            "retraining", "changing epochs", "changing seed", "modifying head samples",
            "ridge", "bias correction", "threshold search", "decoder rescue", "new branch",
            "new EMA beta", "TEST-specific threshold/count-bias/NMS tuning",
            "head refit", "OLS on TEST", "calibration", "ensemble weighting",
            "checkpoint selection by TEST", "reusing TEST as untouched holdout",
        ],
    }
    for s in SEED_ORDER:
        cfg = SEEDS[s]
        r38 = B / "20261005_r38" / f"r38_matched_wls_seed{s}.json"
        r39 = B / "20261005_r39" / f"r39_matched_wls_seed45.json"
        jp = r39 if s == "45" else r38
        j = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
        so = j.get("solver", {})
        man["seeds"][s] = {
            "seed": s, "source_round": cfg["source"],
            "e_ckpt": str(cfg["e"]), "e_ckpt_md5": md5_file(cfg["e"]),
            "e_epoch": 10, "e_weight_kind": "ema",
            "deployment_ckpt": str(cfg["dep"]), "deployment_ckpt_md5": md5_file(cfg["dep"]),
            "unified_val_final": cfg["val_final"],
            "matched_wls": {
                "objective": "mean_j (x_j theta - r_j)^2 (equal weight per matched object)",
                "solver": so.get("call"), "cond": so.get("cond"), "rank": so.get("rank"),
                "n_singular_values": so.get("n_singular_values"),
                "s_max": so.get("s_max"), "s_min": so.get("s_min"),
                "effective_cond": so.get("effective_cond_smax_over_s_rank"),
                "truncation_used": (so.get("stationarity") or {}).get("truncation_used"),
                "train_n_match": (j.get("train_catalog") or {}).get("train_n_match"),
                "design_shape": (j.get("design_matrix") or {}).get("shape"),
                "json": str(jp), "json_md5": md5_file(jp),
            },
        }
    mp = OUT / "final_method_manifest.json"
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    h = hashlib.sha256(mp.read_bytes()).hexdigest()
    (OUT / "final_method_manifest.sha256").write_text(f"{h}  final_method_manifest.json\n",
                                                      encoding="utf-8")
    print(json.dumps({"manifest": str(mp), "sha256": h,
                      "created_utc": man["created_utc"], "created_local": man["created_local"],
                      "sizes": {k: (md5_file(SEEDS[k]["dep"]) is not None) for k in SEED_ORDER}},
                     ensure_ascii=False), flush=True)
    return 0


def stage_valcheck(args):
    device = resolve_device(args.device)
    net, d_in, w_in = build_net(device)
    val_ds = DualFusionTrainDataset(NPZ, TRAIN_FITS, "val", None,
                                    dilated_input_mode=d_in, wzx_feature_mode=w_in)
    wavelength = np.asarray(val_ds.wavelength, dtype=np.float64)
    truth = labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=20.3)
    loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
    res = {}
    ok_all = True
    for s in SEED_ORDER:
        cfg = SEEDS[s]
        ck = load_deployment(net, cfg["dep"], device)
        ep = eval_pass(net, loader, device)
        cat = decode_validation_catalog({"heatmap": ep["heatmap"], "offset": ep["offset"],
                                         "count_logits": ep["count_logits"],
                                         "lognhi": ep["lognhi"]},
                                        val_ds.indices, val_ds.labels, wavelength, **DEC_ARGS)
        m = metrics_from(cat, truth)
        ref = cfg["val_final"]
        ok = abs(m["final"] - ref) < 1e-9
        ok_all = ok_all and ok
        res[s] = {"weight_kind": (ck.get("score") or {}).get("weight_kind"),
                  "reproduced_final": m["final"], "reference_final": ref, "pass": bool(ok),
                  "detection": m["detection"], "parameter": m["parameter"],
                  "score_nhi": m["score_nhi"], "n_pred": m["n_pred"], "n_match": m["n_match"]}
        print(json.dumps({"stage": "valcheck", "seed": s, **res[s]}), flush=True)
    (OUT / "valcheck_results.json").write_text(
        json.dumps({"all_pass": bool(ok_all), "per_seed": res,
                    "tolerance": 1e-9}, ensure_ascii=False, indent=2), encoding="utf-8")
    if not ok_all:
        raise SystemExit("FATAL: final checkpoint did not reproduce the frozen validation endpoint")
    return 0


def stage_predict(args):
    device = resolve_device(args.device)
    net, d_in, w_in = build_net(device)
    test_ds = DualFusionTestDataset(TEST_FITS, dilated_input_mode=d_in, wzx_feature_mode=w_in)
    n = len(test_ds)
    # metadata only (NOT a DLA label column): per-object SNR_GU, used solely to fill the
    # catalog SNR field which the official scorer uses for binning.  It does not enter
    # peak selection, z, offset or logNHI.
    tmeta = read_truth_hdu(TEST_TRUTH, "TRUTH")
    snr_gu = np.asarray(tmeta["SNR_GU"]).astype(np.float32)
    labels = {"Z_QSO": np.asarray(test_ds.zq, dtype=np.float32), "SNR_GU": snr_gu}
    indices = np.arange(n, dtype=np.int64)
    wavelength = np.asarray(test_ds.wavelength, dtype=np.float64)
    loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=False)
    manifest_hashes = {}
    for s in SEED_ORDER:
        load_deployment(net, SEEDS[s]["dep"], device)
        t0 = time.time()
        ep = eval_pass(net, loader, device)
        cat = decode_validation_catalog({"heatmap": ep["heatmap"], "offset": ep["offset"],
                                         "count_logits": ep["count_logits"],
                                         "lognhi": ep["lognhi"]},
                                        indices, labels, wavelength, **DEC_ARGS)
        p = OUT / f"test_predictions_seed{s}.npz"
        np.savez_compressed(p, **{f: cat[f] for f in CAT_FIELDS},
                            **{f"LOG_NHI": cat["LOG_NHI"]},
                            rows=ep["rows"])
        h = md5_file(p)
        manifest_hashes[s] = {"path": str(p), "md5": h, "n_pred_rows": int(len(cat["TARGETID"])),
                              "seconds": time.time() - t0}
        print(json.dumps({"stage": "predict", "seed": s, "n_pred": int(len(cat["TARGETID"])),
                          "md5": h}), flush=True)
    (OUT / "test_predictions_manifest.json").write_text(
        json.dumps({"note": "frozen before any TEST scoring; only SNR_GU metadata was read "
                            "from the truth file, never N_DLA/Z_DLA/LOGNHI/HAS_DLA",
                    "predictions": manifest_hashes}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


def stage_score(args):
    truth = labels_to_truth(read_truth_hdu(TEST_TRUTH, "TRUTH"),
                            np.arange(100000), min_lognhi=20.3)
    pm = json.loads((OUT / "test_predictions_manifest.json").read_text(encoding="utf-8"))
    res, rows_csv = {}, []
    for s in SEED_ORDER:
        p = OUT / f"test_predictions_seed{s}.npz"
        cur = md5_file(p)
        frozen = pm["predictions"][s]["md5"]
        assert cur == frozen, f"prediction artifact changed for seed {s}"
        with np.load(p) as z:
            cat = {f: z[f] for f in CAT_FIELDS + ("LOG_NHI",)}
        m = metrics_from(cat, truth)
        m["prediction_md5"] = cur
        res[s] = m
        print(json.dumps({"stage": "score", "seed": s, "final": m["final"],
                          "detection": m["detection"], "parameter": m["parameter"],
                          "score_z": m["score_z"], "score_nhi": m["score_nhi"],
                          "n_pred": m["n_pred"], "n_match": m["n_match"],
                          "P": m["precision"], "R": m["recall"]}), flush=True)
    finals = [res[s]["final"] for s in SEED_ORDER]
    dets = [res[s]["detection"] for s in SEED_ORDER]
    pars = [res[s]["parameter"] for s in SEED_ORDER]
    summary = {
        "n_seeds": len(finals),
        "final": {"mean": float(np.mean(finals)), "median": float(np.median(finals)),
                  "std_ddof0": float(np.std(finals, ddof=0)),
                  "min": float(np.min(finals)), "max": float(np.max(finals)),
                  "range": float(np.max(finals) - np.min(finals))},
        "detection": {"mean": float(np.mean(dets)), "std_ddof0": float(np.std(dets, ddof=0))},
        "parameter": {"mean": float(np.mean(pars)), "std_ddof0": float(np.std(pars, ddof=0))},
        "predeclared_single_model_companion": {
            "seed": PREDECLARED_COMPANION,
            **{k: res[PREDECLARED_COMPANION][k] for k in
               ("final", "detection", "parameter", "score_z", "score_nhi",
                "precision", "recall", "n_pred", "n_match")}},
        "no_ensemble": True,
        "no_post_test_tuning": True,
    }
    out = {"round": "FINAL", "per_seed": res, "summary": summary,
           "test_data": {"fits": TEST_FITS, "md5": md5_file(TEST_FITS),
                         "truth": TEST_TRUTH, "md5_truth": md5_file(TEST_TRUTH)},
           "manifest_sha256": (OUT / "final_method_manifest.sha256").read_text().split()[0]}
    (OUT / "final_test_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["seed,unified_val_Final,TEST_Final,TEST_Detection,TEST_Parameter,TEST_score_z,"
             "TEST_score_nhi,TEST_Precision,TEST_Recall,TEST_n_pred,TEST_n_match,"
             "TEST_lognhi_signed_bias,TEST_lognhi_std,TEST_lognhi_mae,TEST_lognhi_rmse,"
             "TEST_z_signed_bias,TEST_z_std,TEST_z_mae,TEST_z_rmse"]
    for s in SEED_ORDER:
        m = res[s]
        lines.append(",".join(str(v) for v in [
            s, SEEDS[s]["val_final"], m["final"], m["detection"], m["parameter"], m["score_z"],
            m["score_nhi"], m["precision"], m["recall"], m["n_pred"], m["n_match"],
            m["lognhi_signed_bias"], m["lognhi_std"], m["lognhi_mae"], m["lognhi_rmse"],
            m["z_signed_bias"], m["z_std"], m["z_mae"], m["z_rmse"]]))
    (OUT / "final_test_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["manifest", "valcheck", "predict", "score"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    return {"manifest": stage_manifest, "valcheck": stage_valcheck,
            "predict": stage_predict, "score": stage_score}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
