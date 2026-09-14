"""Smoke test for the engineered channels: shapes, ranges, redundancy, cost."""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/home/heruihua/csst-dla-finder/src")
sys.path.insert(0, "/home/heruihua/csst-dla-finder/hybrid_ensemble")

from data import HybridTestDataset, channel_count

TESTS = "/data/aoqige/test.fits"
N = 300
ref = None

for mode in ["flux", "flux_sig", "flux_feat", "flux_aug"]:
    ds = HybridTestDataset(TESTS, mode)
    start = time.time()
    xs = np.stack([ds[i][0].numpy() for i in range(N)], axis=0)
    per_spec_ms = (time.time() - start) / N * 1000.0
    print(
        f"{mode:10s} ch={channel_count(mode):2d} shape={tuple(xs[0].shape)} "
        f"nan={int(np.isnan(xs).sum())} inf={int(np.isinf(xs).sum())} "
        f"cost={per_spec_ms:.1f} ms/spec  -> 400K cache ~{per_spec_ms * 400000 / 60000:.1f} min"
    )
    for c in range(xs.shape[1]):
        col = xs[:, c]
        print(
            f"    ch{c}: min={col.min():9.3f} max={col.max():9.3f} "
            f"mean={col.mean():8.3f} std={col.std():7.3f}"
        )
    if mode == "flux":
        ref = xs
    else:
        same = np.array_equal(ref, xs[:, :6])
        print(f"    first 6 channels == flux mode: {same}")
        if not same:
            print(f"    max abs diff: {np.abs(ref - xs[:, :6]).max():.6f}")
    print()
