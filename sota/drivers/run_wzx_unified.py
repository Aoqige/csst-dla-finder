#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_wzx_unified.py —— R12-D：让 FlatNet 塔用**统一 split** 训练。

wzx 的 `train.py` 自己用 `make_split()` 现切（seed/split_seed 决定），
所以不能靠换 npz 生效。这里把 `T.make_split` 换成"直接返回统一 split"，
其余（模型/损失/lr/epochs/seed/特征模式）**全部原样**。

用法
  UNIFIED_SPLITS=~/csst_dla_runs/20261003_r12/splits_unified.npz \
    python run_wzx_unified.py --train_fits ... --output_dir ... （其余原样透传）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT / "hybrid_ensemble"), str(_ROOT / "vendor"),
           str(Path(__file__).resolve().parent)):
    sys.path.insert(0, _p)

from csst_dla_wzx_pkg import train as T  # noqa: E402

SPLITS = os.environ.get("UNIFIED_SPLITS")
if not SPLITS:
    raise SystemExit("需要 UNIFIED_SPLITS=<splits_unified.npz>")

with np.load(SPLITS) as s:
    TR = s["train_idx"].astype(np.int64)
    VA = s["val_idx"].astype(np.int64)
print(f"[unified] 载入统一切分 train={TR.size} val={VA.size} from {SPLITS}", flush=True)

# 自检：两集合不相交、并集 = 全集
assert len(np.intersect1d(TR, VA)) == 0, "统一 split 自检失败：train/val 有交集"
print("[unified] 自检 PASS（train ∩ val = 空）", flush=True)


def _make_split(labels_n_dla, val_size, split_seed, max_samples):
    """覆盖 wzx 的现切逻辑：直接返回统一 split（忽略其 seed/val_size 参数）。"""
    print("[unified] make_split 已覆盖 ⇒ 使用统一 split", flush=True)
    return TR, VA


T.make_split = _make_split

if __name__ == "__main__":
    T.main()
