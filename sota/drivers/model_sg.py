#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_sg.py —— R26：把 count head 的输入改为 `shared.detach()`。

唯一改动：count 支路读 **detached** 的 shared fusion feature。
forward 数值与 baseline **完全一致**（detach 不改 value），只是反向图上少一条路径。

- `stopgrad_count=False` → 与 `~/r11/run_fusion_subset.py` 的 `SubsetFusionNet` **逐字等价**（baseline）
- `stopgrad_count=True`  → 仅 count 支路 detach（treatment）

★ 无新增 module / 无新增参数 / 无 RNG 消耗。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("FUS_SUBSET", "both")
_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "src"), str(_ROOT / "hybrid_ensemble"), str(_ROOT / "vendor"),
           str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch                                            # noqa: E402
import torch.nn.functional as F                         # noqa: E402

import run_fusion_subset as RS                          # noqa: E402

DEFAULT_SG = (os.environ.get("SG_MODE", "on") == "on")


class StopGradFusionNet(RS.SubsetFusionNet):
    """`SubsetFusionNet` 的逐字副本，只把 count 支路的输入换成 `shared.detach()`。"""

    def __init__(self, *a, stopgrad_count: bool = False, **k):
        super().__init__(*a, **k)
        self.stopgrad_count = bool(stopgrad_count)
        print(f"[sg] stopgrad_count = {self.stopgrad_count}", flush=True)

    def forward(self, hybrid, wzx, z_qso):
        # ---- 与 SubsetFusionNet.forward 逐字相同 ----
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
        # ★ 唯一改动：count 支路读 detached shared（forward value 不变）
        shared_for_count = shared.detach() if self.stopgrad_count else shared
        pooled = torch.cat(
            [torch.nn.functional.adaptive_avg_pool1d(shared_for_count, 1).flatten(1),
             torch.nn.functional.adaptive_max_pool1d(shared_for_count, 1).flatten(1),
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


class TrainNet(StopGradFusionNet):
    """给 `train_feature_fusion.main()` 用的构造入口（mode 由环境变量 SG_MODE 决定）。"""

    def __init__(self, *a, **k):
        super().__init__(*a, stopgrad_count=DEFAULT_SG, **k)
