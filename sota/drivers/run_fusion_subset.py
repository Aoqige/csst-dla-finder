#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_fusion_subset.py —— R11：G-only / F-only / dual 三个融合头的**重训对照**。

为什么需要重训对照
------------------------------------------------------------------
零训练诊断（`probe_feature_ablation.py`）里"置零一半 feature"是**分布外**操作
（融合头从没见过 0），所以它只能给"方向"，不能给"上限"。
要回答「只用一侧 feature 能到多少分」，必须**从头训一个只用那侧 feature 的头**。

设计（保持其余全部不变）
------------------------------------------------------------------
`DualTowerFusionNet` 的 `fuse[0]` 输入 = concat[d(96), w(96)] = 192。
本驱动把 **被排除的那半整个去掉**（不是置零）：
    subset=both  → fuse[0] in=192，输入 concat[d, w]      （= 原版，用现有 `flat_ctrl` 当对照）
    subset=grow  → fuse[0] in=96 ，输入 d                  （G-only）
    subset=flat  → fuse[0] in=96 ，输入 w                  （F-only）
其余（width 128 / depth 3 / merge_mode residual_dilated / 五头结构 / lr / epochs / seed / decoder）
**一字不动**。★ 两侧 feature 都是 96 维 ⇒ G-only 与 F-only 的**架构完全相同**，可直接对比。

★ `merge_mode` 仍是 `residual_dilated` ⇒ **输出 = GrowNet 自己的头(base) + delta**
  ⇒ 三个臂共享同一个 base ⇒ 差异**只来自 delta 那一侧**，可比。

用法
  FUS_SUBSET=grow python run_fusion_subset.py --targets ... --dilated-checkpoint ... \
      --wzx-checkpoint ... --merge-mode residual_dilated ... --epochs 40 --out-dir ...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch import nn

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT / "hybrid_ensemble"), str(_ROOT / "vendor"),
           str(Path(__file__).resolve().parent)):
    sys.path.insert(0, _p)

SUBSET = os.environ.get("FUS_SUBSET", "both")   # both | grow | flat
print(f"[subset] FUS_SUBSET={SUBSET}", flush=True)

from feature_fusion import DualTowerFusionNet as _Base   # noqa: E402


class SubsetFusionNet(_Base):
    """只把选中一侧的 feature 喂给融合头；其余结构与父类逐字相同。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.subset = SUBSET
        d_ch = int(self.dilated_backbone.refine[0].out_channels)
        w_ch = int(self.wzx_backbone.stem[0].out_channels)
        if d_ch != w_ch:
            raise SystemExit(f"两侧通道不等 ({d_ch} vs {w_ch})，本驱动假设相等")
        keep = d_ch if SUBSET == "grow" else w_ch
        if SUBSET == "both":
            keep = d_ch + w_ch
        first = self.fuse[0]
        new_first = nn.Conv1d(keep, first.out_channels, kernel_size=first.kernel_size[0],
                              padding=first.padding[0])
        self.fuse = nn.Sequential(new_first, *list(self.fuse)[1:])
        if self.merge_mode != "plain":
            for head in (self.center_delta, self.region_delta,
                         self.lognhi_delta, self.offset_delta):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            nn.init.zeros_(self.count_delta[-1].weight)
            nn.init.zeros_(self.count_delta[-1].bias)
        print(f"[subset] fuse[0] in_channels = {keep}", flush=True)

    def forward(self, hybrid, wzx, z_qso):
        if self.freeze_backbones:
            with torch.no_grad():
                d_features = self._dilated_features(hybrid)
                w_features = self._wzx_features(wzx, z_qso)
        else:
            d_features = self._dilated_features(hybrid)
            w_features = self._wzx_features(wzx, z_qso)

        if self.subset == "grow":
            feats = d_features
        elif self.subset == "flat":
            feats = w_features
        else:
            feats = torch.cat([d_features, w_features], dim=1)

        shared = self.fuse(feats)
        center_delta = self.center_delta(shared).squeeze(1)
        region_delta = self.region_delta(shared).squeeze(1)
        lognhi_delta = self.lognhi_delta(shared).squeeze(1)
        offset_delta = self.offset_delta(shared).squeeze(1)
        pooled = torch.cat(
            [torch.nn.functional.adaptive_avg_pool1d(shared, 1).flatten(1),
             torch.nn.functional.adaptive_max_pool1d(shared, 1).flatten(1),
             z_qso.view(-1, 1)], dim=1)
        count_delta = self.count_delta(pooled)

        if self.merge_mode == "plain":
            return {"center_logits": center_delta, "region_logits": region_delta,
                    "lognhi_raw": lognhi_delta, "offset_raw": torch.tanh(offset_delta),
                    "count_logits": count_delta}
        base = (self._dilated_base(d_features) if self.merge_mode == "residual_dilated"
                else self._wzx_base(w_features, z_qso))
        return {"center_logits": base["center_logits"] + center_delta,
                "region_logits": base["region_logits"] + region_delta,
                "lognhi_raw": base["lognhi_raw"] + lognhi_delta,
                "offset_raw": torch.clamp(base["offset_raw"] + 0.5 * torch.tanh(offset_delta),
                                          -1.0, 1.0),
                "count_logits": base["count_logits"] + count_delta}


import train_feature_fusion as TFF   # noqa: E402

TFF.DualTowerFusionNet = SubsetFusionNet   # train_feature_fusion 是 from ... import

if __name__ == "__main__":
    TFF.main()
