#!/usr/bin/env python3
"""Reproduce the shipped ensemble_4way_v3 decode on the val split.

Anchor for the verifier comparison: the pool npz holds candidates from four
members, but the compl/purity numbers we must beat are those of the *shipped
pipeline* (per-member threshold 0.45 + per-member count_bias, then union merge).
This script rebuilds exactly that on val from the cached member predictions, so
the verifier table has an apples-to-apples baseline.

Run: /home/dingjch/anaconda3/envs/ML_env/bin/python -u val_union_baseline.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
for p in (HB, os.path.expanduser("~/csst-dla-finder/src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from csst_dla.scoring import C_KMS, greedy_match, labels_to_truth   # noqa: E402
from csst_dla.fits_utils import read_labels                          # noqa: E402
from decode import pick_peaks, pixel_to_wavelength, softmax          # noqa: E402

CACHE = "/tmp/verifier/member_preds"
TARGETS = os.path.expanduser("~/csst_dla_runs/20260903/common/cnn_targets_seed42.npz")
TRAIN_FITS = "/data/aoqige/train_5e5.fits"
LYA = 1215.67
DEDUP_DV = 1500.0
MEMBERS = {                       # per-member decode, exactly as shipped
    "v3c": dict(threshold=0.45, count_bias=[0.0, 0.75, 0.75]),
    "v5": dict(threshold=0.45, count_bias=[0.0, 0.75, 0.75]),
    "v6": dict(threshold=0.45, count_bias=[0.0, 1.5, 1.5]),
    "sota": dict(threshold=0.45, count_bias=[0.0, 0.0, 0.0]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grp", type=int, default=-1,
                    help="restrict to val rows with row%%3==grp (-1 = whole val split)")
    ap.add_argument("--out", default="/tmp/verifier/val_union_baseline.json")
    args = ap.parse_args()

    with np.load(TARGETS) as f:
        val_idx = f["val_idx"].astype(np.int64)
        wave = f["wavelength"].astype(np.float32)
    if args.grp >= 0:
        sel = np.arange(len(val_idx)) % 3 == args.grp
        keep_rows = np.flatnonzero(sel)
        val_idx_g = val_idx[keep_rows]
    else:
        keep_rows = np.arange(len(val_idx))
        val_idx_g = val_idx
    labels = read_labels(TRAIN_FITS)
    truth = labels_to_truth(labels, val_idx_g, min_lognhi=20.3)
    zq = labels["Z_QSO"][val_idx_g].astype(np.float32)
    n_rows = len(keep_rows)

    rows = []          # (tid, z, lognhi, member)
    for name, dec in MEMBERS.items():
        with np.load(os.path.join(CACHE, f"{name}.npz")) as f:
            p = {k: f[k] for k in f.files}
        assert np.array_equal(p["rows"], val_idx)
        cp = softmax(p["count_logits"] + np.asarray(dec["count_bias"], np.float32)[None, :])
        n_pick_all = np.argmax(cp, axis=1)
        n_only = 0
        for i, r in enumerate(keep_rows):
            peaks = pick_peaks(p["heatmap"][r], wave, float(zq[i]),
                               n_pick=int(n_pick_all[r]), threshold=dec["threshold"],
                               min_distance=10, min_z_dla=1.10)
            for pix in peaks:
                lam = pixel_to_wavelength(pix + float(p["offset"][r, pix]), wave)
                rows.append((int(val_idx[r]), float(lam / LYA - 1.0),
                             float(np.clip(p["lognhi"][r, pix], 19.5, 22.5)), name))
                n_only += 1
        print(f"  {name}: {n_only} raw candidates", flush=True)

    tid = np.asarray([r[0] for r in rows], dtype=np.int64)
    z = np.asarray([r[1] for r in rows], dtype=np.float32)
    logn = np.asarray([r[2] for r in rows], dtype=np.float32)

    # union merge: rank by predicted logNHI, <=2 per spectrum, 1500 km/s dedup
    order = np.argsort(-logn, kind="stable")
    taken: dict[int, list[float]] = {}
    keep: list[int] = []
    for i in order:
        t = int(tid[i])
        lst = taken.setdefault(t, [])
        if len(lst) >= 2:
            continue
        zz = float(z[i])
        if any(C_KMS * abs(zz - o) / (1.0 + o) < DEDUP_DV for o in lst):
            continue
        lst.append(zz)
        keep.append(int(i))
    keep = np.asarray(keep)

    pred = {"TARGETID": tid[keep], "Z_DLA": z[keep],
            "LOG_NHI": logn[keep], "SNR": np.zeros(len(keep), np.float32)}
    m = greedy_match(truth, pred)
    n_pred, n_truth = len(keep), len(truth["TARGETID"])
    compl = len(m) / n_truth
    purity = len(m) / n_pred
    ti = np.asarray([x[0] for x in m])
    tz = truth["Z_DLA"][ti]
    pz = z[keep[np.asarray([x[1] for x in m])]]
    dv = C_KMS * (pz - tz) / (1.0 + tz)
    out = dict(grp=args.grp, n_rows=int(n_rows), n_pred=int(n_pred), n_truth=int(n_truth),
               n_match=int(len(m)), completeness=float(compl), purity=float(purity),
               f1=float(2 * compl * purity / (compl + purity)),
               n_cand_total=int(len(rows)),
               mean_dv=float(np.mean(dv)), std_dv=float(np.std(dv)))
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
