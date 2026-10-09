#!/usr/bin/env python3
"""Regenerate the seed42 train/val split for the new 500k desisim train set.

Same 400001/99999 split ratio as the 0903 era split_seed42.npz, fresh
default_rng(42) permutation (documented here; not bit-identical to the old
split because the old generator script was not preserved).
"""
import numpy as np

N = 500000
N_VAL = 99999

rng = np.random.default_rng(42)
perm = rng.permutation(N).astype(np.int64)
train_idx = perm[: N - N_VAL]
val_idx = perm[N - N_VAL :]
np.savez_compressed(
    "/home/heruihua/csst_dla_runs/20260924_newdata/common/split_seed42.npz",
    train_idx=train_idx,
    val_idx=val_idx,
    seed=np.int64(42),
)
print("train_idx", train_idx.shape, train_idx[:5])
print("val_idx", val_idx.shape, val_idx[:5])
