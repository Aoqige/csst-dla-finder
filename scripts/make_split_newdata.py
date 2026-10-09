#!/usr/bin/env python3
"""Regenerate the seed42 train/val split for the new 500k desisim train set.

Same 400001/99999 split ratio as the 0903 era split_seed42.npz, fresh
default_rng(42) permutation (documented here; not bit-identical to the old
split because the old generator script was not preserved).
"""
import argparse
from pathlib import Path

import numpy as np

N = 500000
N_VAL = 99999

rng = np.random.default_rng(42)
perm = rng.permutation(N).astype(np.int64)
train_idx = perm[: N - N_VAL]
val_idx = perm[N - N_VAL :]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True, help="output .npz path")
args = parser.parse_args()
out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_path,
    train_idx=train_idx,
    val_idx=val_idx,
    seed=np.int64(42),
)
print("train_idx", train_idx.shape, train_idx[:5])
print("val_idx", val_idx.shape, val_idx[:5])
