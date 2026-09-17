#!/usr/bin/env python3
"""Candidate-level verifier (cascade stage 2), trained and scored on val.

Reads /tmp/verifier/val_pool.npz (build_verifier_val.py).  Three disjoint
spectrum sets, split by val-row parity:
  grp 0 -> train        grp 1 -> early stop        grp 2 -> final score
Towers never *trained* on val (only selected their checkpoint on it), so both
the verifier and the numbers below inherit that one selection step of leakage.

Reported on grp 2:
  * AUC of P(candidate is real), over the whole pool and over the top-30% of the
    pool by heatmap (the region that actually decides the submission)
  * compl / purity of a rank-ordered selection (<=2 per spectrum, 1500 km/s
    dedup) at budgets around the shipped operating point, versus hm, lognhi,
    hm*cprob and the |dv| oracle, all normalised by that group's truth count
  * sigma_dv of matched pairs before/after the verifier's velocity head

Run:
  CUDA_VISIBLE_DEVICES=0 /home/dingjch/anaconda3/envs/ML_env/bin/python \
      -u train_verifier.py --npz /tmp/verifier/val_pool.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
SRC = os.path.expanduser("~/csst-dla-finder/src")
for p in (HB, SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

from csst_dla.scoring import C_KMS, greedy_match   # noqa: E402

DEDUP_DV = 1500.0
MAX_PER_LINE = 2
SHIP_RATIO = 1470.0 / 1644.0     # shipped operating point: preds per truth


def log(msg, **kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kw}), flush=True)


class Verifier(nn.Module):
    def __init__(self, n_scalars: int, width: int = 32, hidden: int = 96, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(5, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_scalars, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 64), nn.ReLU(),
        )
        self.cls = nn.Linear(64, 1)
        self.dv = nn.Linear(64, 1)

    def forward(self, win, sc):
        h = self.conv(win)
        h = torch.cat([h.amax(dim=2), h.mean(dim=2)], dim=1)
        z = self.head(torch.cat([h, self.mlp(sc)], dim=1))
        return self.cls(z).squeeze(1), self.dv(z).squeeze(1)


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
        lst = taken.get(t)
        if lst is None:
            lst = []
            taken[t] = lst
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


def score_selection(keep, tid, z, truth):
    if len(keep) == 0:
        return 0, 0.0, 0.0, 0, []
    pred = {"TARGETID": tid[keep].astype(np.int64), "Z_DLA": z[keep].astype(np.float32),
            "LOG_NHI": np.zeros(len(keep), dtype=np.float32)}
    m = greedy_match(truth, pred)
    n_pred = len(keep)
    return n_pred, len(m) / len(truth["TARGETID"]), len(m) / n_pred, len(m), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/tmp/verifier/val_pool.npz")
    ap.add_argument("--out-dir", default="/tmp/verifier")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--dv-weight", type=float, default=0.2)
    ap.add_argument("--test-grp", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    d = np.load(args.npz, allow_pickle=True)
    window = d["window"].astype(np.float32)
    scalars = d["scalars"].astype(np.float32)
    names = [str(x) for x in d["scalar_names"]]
    y = d["y"].astype(np.float32)
    dv = d["dv"].astype(np.float32)
    dv_near = d["dv_near"].astype(np.float32)
    dup = d["dup"].astype(bool)
    grp = d["grp"]
    row = d["row"].astype(np.int64)
    tid = d["targetid"]
    z = d["z"].astype(np.float32)
    logn = d["logn"].astype(np.float32)
    hm = d["hm"].astype(np.float32)
    val_idx = d["val_idx"].astype(np.int64)
    meta = json.loads(str(d["meta"]))
    truth_all = {
        "TARGETID": d["truth_targetid"].astype(np.int64),
        "Z_DLA": d["truth_z"].astype(np.float32),
        "LOG_NHI": d["truth_lognhi"].astype(np.float32),
        "SNR": d["truth_snr"].astype(np.float32),
    }
    log("loaded", n_cand=int(len(y)), n_pos=int(y.sum()), n_scalars=len(names), meta=meta)

    def truth_of_grp(g):
        tids = val_idx[g::3]
        m = np.isin(truth_all["TARGETID"], tids)
        return {k: v[m] for k, v in truth_all.items()}

    gtr, gva, gte = 0, 1, args.test_grp
    tr = np.flatnonzero(grp == gtr)
    va = np.flatnonzero(grp == gva)
    te = np.flatnonzero(grp == gte)
    truth_te = truth_of_grp(gte)
    obs = ~dup          # 'dup' = a second candidate for an already-matched truth
    log("split", n_train=int(len(tr)), n_early=int(len(va)), n_test=int(len(te)),
        n_truth_test=int(len(truth_te["TARGETID"])),
        pos_train=int(y[tr].sum()), pos_test=int(y[te].sum()), dup_total=int(dup.sum()))

    mu, sd = scalars[tr].mean(0), scalars[tr].std(0)
    sd[sd < 1e-6] = 1.0
    wmu = window[tr].mean(axis=(0, 2))[:, None]
    wsd = window[tr].std(axis=(0, 2))[:, None]
    wsd[wsd < 1e-6] = 1.0

    def pack(idx):
        w = torch.from_numpy((window[idx] - wmu) / wsd)
        s = torch.from_numpy((scalars[idx] - mu) / sd)
        return w, s

    # velocity target: nearest-truth residual, available for positives AND dups
    dvt = np.where(np.isfinite(dv_near), dv_near, 0.0).astype(np.float32)
    has_dvt = np.isfinite(dv_near)

    w_tr, s_tr = pack(tr)
    tr_loader = DataLoader(
        TensorDataset(w_tr, s_tr, torch.from_numpy(y[tr]), torch.from_numpy(dvt[tr]),
                      torch.from_numpy(obs[tr].astype(np.float32)),
                      torch.from_numpy(has_dvt[tr].astype(np.float32))),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    w_va, s_va = pack(va)
    w_te, s_te = pack(te)

    model = Verifier(len(names), dropout=args.dropout).to(device)
    log("model", params=sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(tr_loader)), pct_start=0.15)
    pos_w = torch.tensor([float(((y[tr] == 0) & obs[tr]).sum() / max(1, (y[tr] == 1).sum()))],
                         device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduction="none")
    DV_SCALE = 1000.0

    @torch.no_grad()
    def infer(w, s, bs=65536):
        model.eval()
        sc = np.empty(len(w), dtype=np.float32)
        dh = np.empty(len(w), dtype=np.float32)
        for i in range(0, len(w), bs):
            lg, dvv = model(w[i:i + bs].to(device), s[i:i + bs].to(device))
            sc[i:i + bs] = torch.sigmoid(lg).cpu().numpy()
            dh[i:i + bs] = dvv.cpu().numpy() * DV_SCALE
        return sc, dh

    best = dict(auc=-1.0, epoch=-1, state=None)
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for w, s, yy, dd, ob, hd in tr_loader:
            w, s, yy, dd = w.to(device), s.to(device), yy.to(device), dd.to(device)
            ob, hd = ob.to(device), hd.to(device)
            logit, dv_pred = model(w, s)
            per = bce(logit, yy) * ob
            loss = per.sum() / ob.sum().clamp_min(1.0)
            if hd.any():
                loss = loss + args.dv_weight * (((dv_pred - dd / DV_SCALE) ** 2) * hd).sum() / hd.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss.detach())
            nb += 1
        sc_va, _ = infer(w_va, s_va)
        keep_va = obs[va]
        a_va = auc(y[va][keep_va] > 0, sc_va[keep_va])
        if a_va > best["auc"]:
            best = dict(auc=a_va, epoch=ep,
                        state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if ep % 5 == 0 or ep == args.epochs:
            log("epoch", epoch=ep, loss=round(tot / max(1, nb), 4), auc_early=round(a_va, 4),
                best_epoch=best["epoch"], best_auc=round(best["auc"], 4))
        if ep - best["epoch"] >= args.patience:
            log("early stop", epoch=ep, best_epoch=best["epoch"])
            break

    model.load_state_dict(best["state"])
    sc_tr, _ = infer(w_tr, s_tr)
    sc_te, dv_te = infer(w_te, s_te)
    ob_te = obs[te]
    pos_te = (y[te] > 0) & ob_te
    a_te = auc(pos_te, sc_te)
    a_hm = auc(pos_te, hm[te])
    a_logn = auc(pos_te, logn[te])
    top30 = hm[te] >= np.quantile(hm[te], 0.7)
    log("AUC", test_verifier=round(a_te, 4), test_hm=round(a_hm, 4), test_lognhi=round(a_logn, 4),
        train=round(auc((y[tr] > 0) & obs[tr], sc_tr), 4),
        top30_verifier=round(auc((y[te] > 0)[top30], sc_te[top30]), 4),
        top30_hm=round(auc((y[te] > 0)[top30], hm[te][top30]), 4))

    n_truth_te = len(truth_te["TARGETID"])
    budgets = sorted({int(SHIP_RATIO * n_truth_te * f) for f in (0.5, 0.75, 1.0, 1.25, 1.6, 2.0)})
    tid_te, z_te = tid[te], z[te]
    cprob_at = scalars[:, names.index("cprob_at")]
    rankers = {
        "verifier": sc_te,
        "verifier_x_hm": sc_te * hm[te],
        "hm": hm[te],
        "hm_x_cprob": (hm * cprob_at)[te],
        "lognhi": logn[te],
    }
    dv_abs = np.abs(dv[te])
    rankers["oracle"] = -np.where(np.isnan(dv_abs), 1e9, dv_abs)

    report = {}
    for name, sc in rankers.items():
        rows = []
        for b in budgets:
            keep = select(tid_te, z_te, sc, b)
            n_pred, compl, pur, n_match, _ = score_selection(keep, tid_te, z_te, truth_te)
            rows.append(dict(budget=b, n_pred=n_pred, n_match=n_match, compl=compl, purity=pur,
                             f1=(2 * compl * pur / (compl + pur)) if (compl + pur) else 0.0))
        report[name] = rows
        log("frontier", ranker=name,
            **{str(r["budget"]): (round(r["compl"], 4), round(r["purity"], 4), round(r["f1"], 4))
               for r in rows})

    # composition of the selected set: how much budget goes to candidates that are
    # genuinely near a DLA (matched or duplicate) versus pure noise
    useful = ((y > 0) | dup)[te]
    comp = {}
    for name in ("verifier", "verifier_x_hm", "hm", "hm_x_cprob"):
        keep = select(tid_te, z_te, rankers[name], int(SHIP_RATIO * n_truth_te))
        comp[name] = dict(n_pred=int(len(keep)),
                          n_matched=int(useful[keep].sum()),
                          n_dup=int(dup[te][keep].sum()),
                          n_far=int((~useful[keep]).sum()))
    log("composition", **comp)

    # velocity head: does it shrink sigma_dv on matched pairs?
    b0 = int(SHIP_RATIO * n_truth_te)
    keep = select(tid_te, z_te, sc_te, b0)
    _, _, _, _, matches = score_selection(keep, tid_te, z_te, truth_te)
    zfix = {}
    if matches:
        ti = np.asarray([m[0] for m in matches])
        gi = te[keep[np.asarray([m[1] for m in matches])]]     # <- global candidate index
        tz = truth_te["Z_DLA"][ti]
        dv_raw = C_KMS * (z[gi] - tz) / (1.0 + tz)
        dv_hat = dv_te[keep[np.asarray([m[1] for m in matches])]]
        shift = dv_hat / C_KMS * (1.0 + z[gi])
        dv_fix = C_KMS * (z[gi] + shift - tz) / (1.0 + tz)
        zfix = dict(n_match=int(len(matches)),
                    std_dv_raw=float(np.std(dv_raw)), mean_dv_raw=float(np.mean(dv_raw)),
                    std_dv_fix=float(np.std(dv_fix)), mean_dv_fix=float(np.mean(dv_fix)),
                    corr=float(np.corrcoef(dv_raw, dv_hat)[0, 1]),
                    frac_improved=float(np.mean(np.abs(dv_fix) < np.abs(dv_raw))))
        log("velocity head", **{k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in zfix.items()})

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f"verifier_eval_{args.tag}.npz"),
             te=te, scores=sc_te, dv=dv_te)
    with open(os.path.join(args.out_dir, f"verifier_report_{args.tag}.json"), "w") as f:
        json.dump(dict(args=vars(args), auc=dict(test=a_te, hm=a_hm, lognhi=a_logn,
                                                train=auc(y[tr] > 0, sc_tr)),
                       top30=dict(verifier=auc(y[te][top30] > 0, sc_te[top30]),
                                  hm=auc(y[te][top30] > 0, hm[te][top30])),
                       best_epoch=best["epoch"], budgets=budgets,
                       report=report, composition=comp, velocity=zfix,
                       scalar_names=names), f, indent=2)
    log("DONE", out=args.out_dir)


if __name__ == "__main__":
    main()
