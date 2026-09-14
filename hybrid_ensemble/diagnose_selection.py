#!/usr/bin/env python3
"""Where does the submission budget actually go?

The verifier experiment showed the frontier barely moves, so the question is
whether the loss is a *ranking* problem at all.  At the shipped operating point
(~0.89 predictions per truth) we walk the permissive pool in rank order and
classify every accepted candidate:

  matched   : paired with a truth by greedy_match (600 km/s)
  dup       : within 600 km/s of a truth another candidate already claimed
  far-empty : >600 km/s from any truth AND its spectrum has no DLA at all
  far-DLA   : >600 km/s from any truth, but the spectrum does contain a DLA

If most "far" picks land in spectra that contain no DLA, the heatmap is not
mis-ranking within candidate sets -- it is firing on whole empty spectra, and the
lever is a line-level gate (the count head), not a better candidate ranker.

Run: /home/dingjch/anaconda3/envs/ML_env/bin/python -u diagnose_selection.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

for p in (os.path.expanduser("~/csst-dla-finder/hybrid_ensemble"),
          os.path.expanduser("~/csst-dla-finder/src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from csst_dla.scoring import C_KMS, greedy_match   # noqa: E402

NPZ = "/tmp/verifier/val_pool.npz"
SHIP_RATIO = 1470.0 / 1644.0
DEDUP_DV = 1500.0
MAX_PER_LINE = 2


def select(tid, z, score, budget):
    order = np.argsort(-np.nan_to_num(score, nan=-1e9), kind="stable")
    taken: dict[int, list[float]] = {}
    keep: list[int] = []
    for i in order:
        t = int(tid[i])
        lst = taken.setdefault(t, [])
        if len(lst) >= MAX_PER_LINE:
            continue
        zz = float(z[i])
        if any(C_KMS * abs(zz - o) / (1.0 + o) < DEDUP_DV for o in lst):
            continue
        lst.append(zz)
        keep.append(int(i))
        if len(keep) >= budget:
            break
    return np.asarray(keep, dtype=np.int64)


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    grp = d["grp"]
    te = np.flatnonzero(grp == 2)
    row = d["row"].astype(np.int64)
    tid = d["targetid"]
    z = d["z"].astype(np.float32)
    hm = d["hm"].astype(np.float32)
    logn = d["logn"].astype(np.float32)
    y = d["y"].astype(bool)
    dup = d["dup"].astype(bool)
    scalars = d["scalars"].astype(np.float32)
    names = [str(x) for x in d["scalar_names"]]
    cprob = scalars[:, names.index("cprob_at")]

    truth = {"TARGETID": d["truth_targetid"].astype(np.int64),
             "Z_DLA": d["truth_z"].astype(np.float32),
             "LOG_NHI": d["truth_lognhi"].astype(np.float32),
             "SNR": d["truth_snr"].astype(np.float32)}
    # restrict truth to the eval spectra
    val_idx = d["val_idx"].astype(np.int64)
    m = np.isin(truth["TARGETID"], val_idx[2::3])
    truth_te = {k: v[m] for k, v in truth.items()}
    n_truth = len(truth_te["TARGETID"])

    # per-spectrum DLA presence, indexed by val row position
    n_dla = d["labels_N_DLA"]                       # (n_val,)
    ln1 = d["labels_LOGNHI1"]
    ln2 = d["labels_LOGNHI2"]
    has_dla = ((np.maximum(ln1, np.where(n_dla > 1, ln2, -99.0)) >= 20.3)
               & (n_dla > 0))

    budget = int(SHIP_RATIO * n_truth)
    print(json.dumps({"n_truth_eval": int(n_truth), "budget": budget,
                      "n_pool_eval": int(len(te)),
                      "frac_spectra_with_dla": round(float(has_dla[row[te]].mean()), 4)}))
    print(f"  in-pool candidates near a truth : {(y | dup)[te].sum()}")
    print(f"  candidates in spectra WITH a DLA: {has_dla[row[te]].sum()}")

    rows = {}
    for name, sc in {"hm": hm, "hm_x_cprob": hm * cprob, "lognhi": logn}.items():
        keep = select(tid[te], z[te], sc[te], budget)
        g = te[keep]
        matched = y[g].sum()
        dups = (dup[g] & ~y[g]).sum()
        far = ~(y[g] | dup[g])
        far_dla = int((far & has_dla[row[g]]).sum())
        far_empty = int((far & ~has_dla[row[g]]).sum())
        lines = np.unique(tid[g])
        lines_with_truth = int(np.unique(tid[g][y[g] | dup[g]]).size)
        rows[name] = dict(
            n_pred=int(len(keep)), n_matched=int(matched), n_dup=int(dups),
            far_in_dla_line=far_dla, far_in_empty_line=far_empty,
            distinct_lines=int(lines.size), lines_carrying_a_hit=lines_with_truth,
            matched_per_line=round(float(matched / max(1, lines.size)), 3),
        )
        print(f"  {name}: {json.dumps(rows[name])}")

    # How much would a PERFECT line-level gate be worth?  Restrict the pool to
    # candidates whose spectrum really contains a DLA and re-run the same walk.
    # (Oracle: uses the truth N_DLA.  Upper bound for a calibrated count head.)
    line_ok = has_dla[row[te]]
    for name, sc in {"hm": hm, "hm_x_cprob": hm * cprob}.items():
        sub = np.flatnonzero(line_ok)
        keep = select(tid[te][sub], z[te][sub], sc[te][sub], budget)
        g = te[sub][keep]
        print(f"  ORACLE line gate + {name}: n_pred={len(keep)} "
              f"matched={int(y[g].sum())} dup={int((dup[g] & ~y[g]).sum())} "
              f"far_empty={int((~(y[g] | dup[g]) & ~has_dla[row[g]]).sum())} "
              f"lines={int(np.unique(tid[g]).size)}")

    # truths the pool never even offered
    offered = set(truth_te["TARGETID"][np.isin(truth_te["TARGETID"], tid[te][y[te] | dup[te]])].tolist())
    all_t = set(np.unique(truth_te["TARGETID"]).tolist())
    print(f"  truths in eval spectra           : {len(all_t)}")
    print(f"  truths with a candidate in pool   : {len(offered)}"
          f"  ({len(offered) / max(1, len(all_t)):.1%})")
    print(f"  truths with NO candidate at all   : {len(all_t) - len(offered)}")

    with open("/tmp/verifier/selection_diagnosis.json", "w") as f:
        json.dump(dict(n_truth=n_truth, budget=budget, rankers=rows,
                       truths_in_eval=len(all_t), truths_offered=len(offered)), f, indent=2)


if __name__ == "__main__":
    main()
