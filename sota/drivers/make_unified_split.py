#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_unified_split.py —— R12-D：建**两塔 + fusion 完全一致**的共同 index split。

背景（为什么必须做）
------------------------------------------------------------------
R10 侦察发现：两塔训练用的 val 划分**是两套独立随机切分**
（cnn npz val_idx 99,999 ∩ wzx splits val_idx 100,000 = 20,043，Jaccard 0.111）
⇒ **FlatNet 在训练时见过本次评测集里 79.8% 的真值 DLA**。
已证无记忆效应（z-MAE 差 1.2%），但**论文要引用的数字必须来自干净口径**。

做法
------------------------------------------------------------------
1. 把现有 npz 的 train/val 切片**还原成全量 500,000 行**的目标张量
   （`full[train_idx] = train_*`、`full[val_idx] = val_*`），并**自检**还原正确。
2. 用**统一的**切分（seed 42、val_size 0.2、**按打分口径 count 分层**）生成新的 train/val index。
3. 按新切分重新切片，落盘：
     `cnn_targets_unified_seed42_sig15.npz`  （GrowNet 与 fusion 直接可用，无需改代码）
     `splits_unified.npz`                    （`train_idx` / `val_idx`，给 wzx 塔用）
4. 落盘一个对照表：新旧切分的重叠统计（用于报告口径）。

★ 只改**切分**，不改任何标签值 / 输入通道 / 目标张量内容。
"""
from __future__ import annotations

import os

import json
import sys
import argparse
from pathlib import Path

import numpy as np

DEFAULT_SRC = Path(os.environ.get(
    "CSST_UNIFIED_SRC",
    str(Path.home() / "csst_dla_runs/20260929_tftune/common/cnn_targets_seed42_sig15.npz")))
DEFAULT_OUTDIR = Path(os.environ.get(
    "CSST_UNIFIED_OUTDIR", str(Path.home() / "csst_dla_runs/20261003_r12")))
SEED = 42
VAL_SIZE = 0.2


def parse_args() -> "argparse.Namespace":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="input targets npz to re-split (its train_idx/val_idx are reconstructed)")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="output directory")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-size", type=float, default=VAL_SIZE)
    return ap.parse_args()

TARGET_KEYS = ["center", "region", "lognhi", "mask"]


def main() -> int:
    args = parse_args()
    SRC = Path(args.src)
    OUTDIR = Path(args.outdir)
    OUT_NPZ = OUTDIR / "cnn_targets_unified_seed42_sig15.npz"
    OUT_SPLITS = OUTDIR / "splits_unified.npz"
    SEED = int(args.seed)
    VAL_SIZE = float(args.val_size)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with np.load(SRC) as s:
        keys = list(s.files)
        tr = s["train_idx"].astype(np.int64)
        va = s["val_idx"].astype(np.int64)
        n_total = int(s["x"].shape[0])
        wave = s["wavelength"].astype(np.float32)
        # 其余标量/常量原样带过去
        consts = {k: s[k] for k in keys
                  if k not in ("x", "snr_proxy", "snr_gu", "wavelength")
                  and not k.startswith("train_") and not k.startswith("val_")
                  and k not in ("train_idx", "val_idx")}

        print(json.dumps({"src": str(SRC), "n_total": n_total,
                          "old_train": int(tr.size), "old_val": int(va.size)}), flush=True)

        # ---- 1. 还原全量目标张量 ----
        full = {}
        for k in TARGET_KEYS:
            arr = np.zeros((n_total, wave.size), dtype=np.float32)
            arr[tr] = s[f"train_{k}"].astype(np.float32)
            arr[va] = s[f"val_{k}"].astype(np.float32)
            full[k] = arr
        x_full = s["x"].astype(np.float32)
        snr_p = s["snr_proxy"].astype(np.float32)
        snr_g = s["snr_gu"].astype(np.float32)

    # 还原自检：full[k][va] 必须与 val_k **逐位相同**（同一顺序 ⇒ 可直接比）
    with np.load(SRC) as s:
        ok = all(np.array_equal(full[k][va], s["val_" + k].astype(np.float32))
                 for k in TARGET_KEYS)
        ok = ok and all(np.array_equal(full[k][tr], s["train_" + k].astype(np.float32))
                        for k in TARGET_KEYS)
    print(json.dumps({"stage": "reconstruct_check", "pass": bool(ok)}), flush=True)
    if not ok:
        raise SystemExit("FATAL: 全量还原自检失败，拒绝继续")

    # ---- 2. 统一切分：按**打分口径 count** 分层 ----
    # 打分口径 count = 该谱中 LOGNHI>=20.3 的 DLA 个数（上限 2）
    from csst_dla.fits_utils import read_labels
    labels = read_labels(os.environ.get("CSST_TRAIN_FITS", str(Path.home() / "data" / "train_500k_GU_qlf.fits")))
    scored = np.zeros(n_total, dtype=np.int64)
    nd = np.asarray(labels["N_DLA"]).astype(np.int64)
    for slot in (1, 2):
        ln = np.asarray(labels[f"LOGNHI{slot}"]).astype(np.float64)
        scored += ((nd >= slot) & np.isfinite(ln) & (ln >= 20.3)).astype(np.int64)
    scored = np.clip(scored, 0, 2)
    print(json.dumps({"stage": "scored_count_dist",
                      "counts": {str(k): int((scored == k).sum()) for k in (0, 1, 2)}}), flush=True)

    from sklearn.model_selection import train_test_split
    idx = np.arange(n_total)
    new_tr, new_va = train_test_split(idx, test_size=VAL_SIZE, random_state=SEED,
                                      stratify=scored)
    new_tr = np.sort(new_tr.astype(np.int64))
    new_va = np.sort(new_va.astype(np.int64))

    # ---- 3. 落盘 ----
    np.savez_compressed(
        OUT_NPZ, wavelength=wave, x=x_full, snr_proxy=snr_p, snr_gu=snr_g,
        train_idx=new_tr, val_idx=new_va,
        **{f"train_{k}": full[k][new_tr] for k in TARGET_KEYS},
        **{f"val_{k}": full[k][new_va] for k in TARGET_KEYS},
        **{k: v for k, v in consts.items()})
    np.savez(OUT_SPLITS, train_idx=new_tr, val_idx=new_va)

    # ---- 4. 新旧切分对照 ----
    inter = np.intersect1d(va, new_va)
    summary = {
        "out_npz": str(OUT_NPZ), "out_splits": str(OUT_SPLITS),
        "new_train": int(new_tr.size), "new_val": int(new_va.size),
        "old_val_vs_new_val": {
            "old_n": int(va.size), "new_n": int(new_va.size),
            "intersect": int(inter.size),
            "jaccard": round(float(inter.size / np.union1d(va, new_va).size), 4)},
        "new_val_scored_dist": {str(k): int((scored[new_va] == k).sum()) for k in (0, 1, 2)},
        "old_val_scored_dist": {str(k): int((scored[va] == k).sum()) for k in (0, 1, 2)},
        "new_val_frac_pos": round(float((scored[new_va] >= 1).mean()), 5),
        "old_val_frac_pos": round(float((scored[va] >= 1).mean()), 5),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    (OUTDIR / "unified_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT_NPZ, "and", OUT_SPLITS, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
