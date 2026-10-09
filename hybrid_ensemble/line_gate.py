#!/usr/bin/env python3
"""Line-level DLA gate: decide *which spectra* deserve submission slots.

diagnose_selection.py showed where the budget actually goes at the shipped
operating point (grp2 of val, 2448 slots):

    matched (a real DLA)          1338
    far, spectrum has NO DLA       946   <-- 39% of the whole budget
    far, spectrum has a DLA        164

and that an ORACLE line gate + heatmap ranking would lift the matched count to
1609 of the 2005 truths the pool contains at all.  So the loss is not
candidate ranking -- it is that the towers fire confidently on spectra that
contain no DLA.

This script trains that gate for real: a per-spectrum classifier
P(spectrum contains >=1 DLA with logNHI >= 20.3) from the four members' own
signals (peak heights, count logits, peak counts, SNR, z_QSO).  Small, CPU-fast,
trained on val rows with row%3==0, early-stopped on row%3==1, reported on
row%3==2 -- the same three disjoint spectrum sets the verifier used.

Then it re-runs the shipped selection walk (<=2 per spectrum, 1500 km/s dedup)
with and without the gate and reports matched / compl / purity at matched budget.

Run: /home/dingjch/anaconda3/envs/ML_env/bin/python -u line_gate.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
for p in (HB, os.path.expanduser("~/csst-dla-finder/src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from csst_dla.scoring import C_KMS, greedy_match   # noqa: E402

CACHE = "/tmp/verifier/member_preds"
NPZ = "/tmp/verifier/val_pool.npz"
MEMBERS = ["v3c", "v5", "v6", "sota"]
DEDUP_DV = 1500.0
MAX_PER_LINE = 2
SHIP_RATIO = 1470.0 / 1644.0
THR_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def log(msg, **kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kw}), flush=True)


def auc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


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
        if budget is not None and len(keep) >= budget:
            break
    return np.asarray(keep, dtype=np.int64)


def evaluate(keep, tid, z, truth):
    if len(keep) == 0:
        return 0, 0.0, 0.0, 0
    pred = {"TARGETID": tid[keep].astype(np.int64), "Z_DLA": z[keep].astype(np.float32),
            "LOG_NHI": np.zeros(len(keep), np.float32)}
    m = greedy_match(truth, pred)
    return len(keep), len(m) / len(truth["TARGETID"]), len(m) / len(keep), len(m)


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    n_val = len(d["val_idx"])
    row = d["row"].astype(np.int64)
    tid_c = d["targetid"]
    z_c = d["z"].astype(np.float32)
    hm_c = d["hm"].astype(np.float32)
    logn_c = d["logn"].astype(np.float32)
    scalars = d["scalars"].astype(np.float32)
    sname = [str(x) for x in d["scalar_names"]]
    grp_c = d["grp"]
    snr_row = d["labels_SNR_GU"].astype(np.float32)
    zq_row = d["labels_Z_QSO"].astype(np.float32)
    n_dla = d["labels_N_DLA"]
    has_dla = ((np.maximum(d["labels_LOGNHI1"],
                           np.where(n_dla > 1, d["labels_LOGNHI2"], -99.0)) >= 20.3)
               & (n_dla > 0))

    # ---------- per-spectrum features from the four members ----------
    feats, names = [], []

    def add(n, v):
        feats.append(np.asarray(v, np.float32))
        names.append(n)

    add("snr", snr_row)
    add("zq", zq_row)
    hmmax_list = []
    for m in MEMBERS:
        with np.load(os.path.join(CACHE, f"{m}.npz")) as f:
            hm = f["heatmap"].astype(np.float32)
            ln = f["lognhi"].astype(np.float32)
            cl = f["count_logits"].astype(np.float32)
            offs = f["offset"].astype(np.float32)
        assert hm.shape[0] == n_val
        add(f"{m}_hmmax", hm.max(axis=1))
        hmmax_list.append(hm.max(axis=1))
        top5 = np.partition(hm, -5, axis=1)[:, -5:]
        add(f"{m}_hmtop5", top5.mean(axis=1))
        add(f"{m}_hmfrac", (hm > 0.45).mean(axis=1))
        add(f"{m}_hmfrac2", (hm > 0.20).mean(axis=1))
        mid = hm[:, 1:-1]
        npk = ((mid > 0.45) & (mid >= hm[:, :-2]) & (mid >= hm[:, 2:])).sum(axis=1)
        add(f"{m}_npeaks", npk.astype(np.float32))
        add(f"{m}_npeaks2", ((mid > 0.20) & (mid >= hm[:, :-2]) & (mid >= hm[:, 2:])).sum(axis=1))
        for j in range(3):
            add(f"{m}_cl{j}", cl[:, j])
        am = hm.argmax(axis=1)
        add(f"{m}_logn_at_max", ln[np.arange(n_val), am])
        add(f"{m}_off_at_max", offs[np.arange(n_val), am])
        del hm, ln, cl, offs
    H = np.stack(hmmax_list, axis=1)
    add("hmmax_ens_mean", H.mean(axis=1))
    add("hmmax_ens_min", H.min(axis=1))
    add("hmmax_ens_std", H.std(axis=1))
    X = np.stack(feats, axis=1)
    log("features", shape=X.shape)

    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    y = has_dla.astype(np.float32)

    tr = np.flatnonzero(np.arange(n_val) % 3 == 0)
    va = np.flatnonzero(np.arange(n_val) % 3 == 1)
    te = np.flatnonzero(np.arange(n_val) % 3 == 2)
    log("rows", train=int(len(tr)), early=int(len(va)), test=int(len(te)),
        base_rate=round(float(y.mean()), 4))

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(X.shape[1], 128), nn.ReLU(), nn.Dropout(0.1),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = torch.tensor([float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()))])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.from_numpy(Xs[tr]); yt = torch.from_numpy(y[tr])
    Xv = torch.from_numpy(Xs[va])

    best = dict(auc=-1, state=None, ep=-1)
    bs = 4096
    for ep in range(1, 61):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv).squeeze(1)).numpy()
        a = auc(y[va], pv)
        if a > best["auc"]:
            best = dict(auc=a, ep=ep,
                        state={k: v.clone() for k, v in model.state_dict().items()})
        if ep - best["ep"] >= 10:
            break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        P = torch.sigmoid(model(torch.from_numpy(Xs).float()).squeeze(1)).numpy()
    log("gate", best_epoch=best["ep"], auc_train=round(auc(y[tr], P[tr]), 4),
        auc_early=round(best["auc"], 4), auc_test=round(auc(y[te], P[te]), 4),
        base_rate=round(float(y[te].mean()), 4))

    # ---------- gated selection on the test third ----------
    truth = {"TARGETID": d["truth_targetid"].astype(np.int64),
             "Z_DLA": d["truth_z"].astype(np.float32),
             "LOG_NHI": d["truth_lognhi"].astype(np.float32),
             "SNR": d["truth_snr"].astype(np.float32)}
    m = np.isin(truth["TARGETID"], d["val_idx"].astype(np.int64)[2::3])
    truth_te = {k: v[m] for k, v in truth.items()}
    n_truth = len(truth_te["TARGETID"])
    budget = int(SHIP_RATIO * n_truth)

    cand_te = np.flatnonzero(grp_c == 2)
    cprob = scalars[:, sname.index("cprob_at")]
    hmx = (hm_c * cprob)[cand_te]
    log("budget", n_truth=int(n_truth), budget=budget, n_candidates=int(len(cand_te)))

    # pick the gate threshold / style on the EARLY-STOP third, report on test third
    cand_va = np.flatnonzero(grp_c == 1)
    m_va = np.isin(truth["TARGETID"], d["val_idx"].astype(np.int64)[1::3])
    truth_va = {k: v[m_va] for k, v in truth.items()}
    budget_va = int(SHIP_RATIO * len(truth_va["TARGETID"]))

    out = {}
    gate_c = P[row[cand_te]]
    gate_v = P[row[cand_va]]
    # threshold chosen on the early third across BOTH gate styles
    best_cfg, best_f1v = None, -1
    for thr in THR_GRID:
        selv = gate_v >= thr if thr > 0 else np.ones(len(cand_va), bool)
        for style in ("pow1", "pow2"):
            g = np.where(gate_v[selv] > 1e-6, gate_v[selv], 1e-6) ** (1 if style == "pow1" else 2)
            scv = (hm_c * cprob)[cand_va][selv] * g
            keep = select(tid_c[cand_va][selv], z_c[cand_va][selv], scv, budget_va)
            _, c_, p_, _ = evaluate(keep, tid_c[cand_va][selv], z_c[cand_va][selv], truth_va)
            f1 = 2 * c_ * p_ / (c_ + p_) if (c_ + p_) else 0.0
            if f1 > best_f1v:
                best_f1v, best_cfg = f1, (thr, style)
            log("thr sweep (early third)", thr=thr, style=style, n_pred=int(keep.size),
                compl=round(c_, 4), purity=round(p_, 4), f1=round(f1, 4))
    chosen_thr, chosen_style = best_cfg
    log("chosen", thr=chosen_thr, style=chosen_style, f1_early=round(best_f1v, 4))

    variants = {
        "hm": (hm_c[cand_te], None, 0.0),
        "hm_x_cprob": (hmx, None, 0.0),
        "gate_x_hm": (hm_c[cand_te], "pow1", 0.0),
        "gate_x_hm_x_cprob": (hmx, "pow1", 0.0),
        "gate2_x_hm_x_cprob": (hmx, "pow2", 0.0),
        "ORACLE_gate_x_hm": (hm_c[cand_te], "oracle", 0.0),
    }
    budgets = sorted({int(SHIP_RATIO * n_truth * f) for f in (0.5, 0.75, 1.0, 1.25, 1.6)})
    out = {}
    for tag, (base, style, thr) in variants.items():
        if style is None:
            sel = np.ones(len(cand_te), bool)
            sc = base
        elif style == "pick":
            sel = gate_c >= thr
            sc = base[sel]
        elif style == "oracle":
            sel = has_dla[row[cand_te]]
            sc = base[sel]
        else:
            sel = np.ones(len(cand_te), bool)
            ex = 1 if style == "pow1" else 2
            sc = base * np.maximum(gate_c, 1e-6) ** ex
        tt, zz = tid_c[cand_te][sel], z_c[cand_te][sel]
        for b in budgets:
            keep = select(tt, zz, sc, b)
            n_pred, compl, pur, n_match = evaluate(keep, tt, zz, truth_te)
            f1 = 2 * compl * pur / (compl + pur) if (compl + pur) else 0.0
            out[f"{tag}@{b}"] = dict(n_cand_pool=int(sel.sum()), n_pred=n_pred,
                                     n_match=n_match, compl=compl, purity=pur, f1=f1,
                                     lines=int(np.unique(tt[keep]).size))
            log("RESULT", tag=tag, budget=b, n_cand=int(sel.sum()), n_pred=n_pred,
                matched=n_match, compl=round(compl, 4), purity=round(pur, 4), f1=round(f1, 4))

    with open("/tmp/verifier/line_gate_report.json", "w") as f:
        json.dump(dict(args=dict(budget=budget, n_truth=n_truth, chosen_thr=chosen_thr,
                                 chosen_style=chosen_style,
                                 gate_auc_test=auc(y[te], P[te]),
                                 gate_auc_train=auc(y[tr], P[tr])),
                       results=out, feature_names=names), f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
