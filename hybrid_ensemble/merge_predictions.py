#!/usr/bin/env python3
"""Merge two predictions.csv by velocity-matching DLA candidates.

Matched pairs: average Z_DLA and LOG_NHI.
Union mode: keep single-side detections too.
Intersection mode: keep only matched (both models agree) candidates.
"""
import csv, sys
from pathlib import Path

C = 299792.458


def read_preds(path):
    d = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tid = int(row["id"])
            cands = []
            for slot in (1, 2):
                z = float(row[f"Z_DLA{slot}"])
                ln = float(row[f"LOGNHI{slot}"])
                if z > 0 and ln >= 20.3:
                    cands.append((z, ln))
            d[tid] = cands
    return d


def dv(z1, z2):
    return abs(z1 - z2) / (1.0 + min(z1, z2)) * C


def merge(cands_a, cands_b, dv_thresh=1500.0, mode="union"):
    used_b = set()
    out = []
    for za, la in cands_a:
        best_j, best_dv = -1, 1e18
        for j, (zb, lb) in enumerate(cands_b):
            if j in used_b:
                continue
            d = dv(za, zb)
            if d < best_dv:
                best_dv, best_j = d, j
        if best_j >= 0 and best_dv < dv_thresh:
            zb, lb = cands_b[best_j]
            used_b.add(best_j)
            out.append(((za + zb) / 2.0, (la + lb) / 2.0))
        elif mode != "intersection":
            out.append((za, la))
    if mode != "intersection":
        for j, (zb, lb) in enumerate(cands_b):
            if j not in used_b:
                out.append((zb, lb))
    out.sort(key=lambda x: -x[1])
    return out[:2]


def main():
    a = sys.argv[1]
    b = sys.argv[2]
    out = sys.argv[3]
    mode = sys.argv[4] if len(sys.argv) > 4 else "union"
    dv_thresh = float(sys.argv[5]) if len(sys.argv) > 5 else 1500.0
    pa = read_preds(a)
    pb = read_preds(b)
    tids = sorted(set(pa) | set(pb))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"])
        n_cand = 0
        for tid in tids:
            merged = merge(pa.get(tid, []), pb.get(tid, []), dv_thresh, mode)
            n_cand += len(merged)
            while len(merged) < 2:
                merged.append((-1.0, 0.0))
            (z1, l1), (z2, l2) = merged[0], merged[1]
            w.writerow([tid, f"{z1:.6f}", f"{l1:.6f}", f"{z2:.6f}", f"{l2:.6f}"])
    print(f"wrote {out} mode={mode} dv_thresh={dv_thresh} rows={len(tids)} total_cands={n_cand}")


if __name__ == "__main__":
    main()
