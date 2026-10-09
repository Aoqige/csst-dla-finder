#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the three R38 per-seed JSONs into the report tables."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

OUT = Path("/home/heruihua/csst_dla_runs/20261005_r38")
SEEDS = ("42", "43", "44")
D = {}
for s in SEEDS:
    p = OUT / f"r38_matched_wls_seed{s}.json"
    D[s] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

KEYS = ("final", "detection", "parameter", "score_z", "score_nhi", "precision", "recall",
        "n_pred", "n_match")


def vh(d, name):
    return (d.get("val_heads") or {}).get(name)


print("=== core table ===")
print("seed | E | AdamW-refit | all-mask WLS(R37) | matched-WLS(R38) | dR38_vs_E | dR38_vs_R37 | dR38_vs_AdamW")
rows = []
for s in SEEDS:
    d = D[s]
    if d is None:
        print(f"{s} | MISSING")
        continue
    e = d["baseline_E_E"]["final"]
    a = (vh(d, "AdamW_refit_ep10") or {}).get("metrics", {}).get("final")
    r37 = (vh(d, "R37_allmask_WLS") or {}).get("metrics", {}).get("final")
    r38 = (vh(d, "R38_matched_WLS") or {}).get("metrics", {}).get("final")
    rows.append((s, e, a, r37, r38))
    print(f"{s} | {e:.10f} | {a:.10f} | {r37:.10f} | {r38:.10f} | {r38-e:+.10f} | "
          f"{r38-r37:+.10f} | {r38-a:+.10f}")

for tag, idx in (("dR38_vs_E", 4), ("dR38_vs_R37", 4), ("dR38_vs_AdamW", 4)):
    pass
dE = [r[4] - r[1] for r in rows]
dR37 = [r[4] - r[3] for r in rows]
dA = [r[4] - r[2] for r in rows]
for name, vals in (("Delta_R38_vs_E", dE), ("Delta_R38_vs_R37_allmask", dR37),
                   ("Delta_R38_vs_AdamW", dA)):
    print(f"{name:28s} mean {st.mean(vals):+.10f} median {st.median(vals):+.10f} "
          f"min {min(vals):+.10f} max {max(vals):+.10f} pos/neg/zero "
          f"{sum(1 for v in vals if v>0)}/{sum(1 for v in vals if v<0)}/{sum(1 for v in vals if v==0)}")

print("\n=== train matched-readout sample ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    tc = d["train_catalog"]
    print(f"seed {s}: truth {tc['train_truth_dla']} n_pred {tc['train_n_pred']} "
          f"n_match {tc['train_n_match']} | matrix {d['design_matrix']['shape']} "
          f"rows==n_match {d['design_matrix']['rows_equal_train_n_match']} "
          f"cols {d['design_matrix']['cols']}")

print("\n=== train matched objective J_match (mean squared residual) ===")
names = ("E", "AdamW_refit_ep10", "R37_allmask_WLS", "R38_matched_WLS", "L_ema_ep40")
print("seed | " + " | ".join(names))
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    t = d["train_matched_objective"]
    print(f"{s} | " + " | ".join(f"{t[n]['J_match_mean_sq']:.8e}" if n in t else "-" for n in names))
print("\n=== train matched residual stats (E / AdamW / R37 / R38) ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    print(f"seed {s}:")
    for n in ("E", "AdamW_refit_ep10", "R37_allmask_WLS", "R38_matched_WLS"):
        if n not in d["train_matched_objective"]:
            continue
        v = d["train_matched_objective"][n]
        print(f"   {n:20s} mse {v['mse']:.8e} bias {v['signed_bias']:+.6f} std {v['std']:.6f} "
              f"mae {v['mae']:.6f} rmse {v['rmse']:.6f}")

print("\n=== solver ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    so = d["solver"]
    print(f"seed {s}: rank {so['rank']}/{so['n_singular_values']} full={so['full_column_rank']} "
          f"s_max {so['s_max']:.6f} s_min {so['s_min']:.6e} cond_eff {so['effective_cond_smax_over_s_rank']:.3f}")
    print(f"   stationarity abs {so['stationarity']['abs_norm_AT_resid']:.3e} "
          f"rel_vs_AT_d {so['stationarity']['rel_vs_AT_d']:.3e} "
          f"trunc {so['stationarity']['truncation_used']} | repeatable "
          f"{so['repeatability']['bitwise_identical']} | cpu_fit {so['cpu_fit_seconds']:.3f}s")
    print(f"   theta0 {so['theta0_norm']:.6f} theta* {so['theta_star_norm']:.6f} "
          f"delta {so['delta_norm']:.6f} | sanity maxdiff np64 "
          f"{d['sanity_readout']['max_abs_diff_numpy64_vs_official']:.3e} "
          f"torch {d['sanity_readout']['max_abs_diff_torch_f32_vs_official']:.3e}")
    print(f"   deploy rounding {d['deployment']['head_rounding_max_abs']:.3e} "
          f"J_f32 {d['deployment']['J_match_f32_arithmetic']:.8e} "
          f"J_f64_rounded {d['deployment']['J_match_f64_on_rounded_params']:.8e}")

print("\n=== validation official layer (E / AdamW / R37 all-mask / R38 matched) ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    e = d["baseline_E_E"]
    a = (vh(d, "AdamW_refit_ep10") or {}).get("metrics")
    r37 = (vh(d, "R37_allmask_WLS") or {}).get("metrics")
    r38 = (vh(d, "R38_matched_WLS") or {}).get("metrics")
    print(f"seed {s}:")
    for k in KEYS:
        fa = f"{a[k]:.6f}" if a else "-"
        f37 = f"{r37[k]:.6f}" if r37 else "-"
        print(f"   {k:10s} E {e[k]:.6f} | AdamW {fa} | R37 {f37} | R38 {r38[k]:.6f}")

print("\n=== fixed val matching set ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    print(f"seed {s} (n_matched={d['baseline_E_E']['n_match']}):")
    print(f"   E_E              : {json.dumps({k: round(v,6) for k,v in d['baseline_E_E'].items() if k.startswith('lognhi_')})}")
    for n in ("AdamW_refit_ep10", "R37_allmask_WLS", "R38_matched_WLS"):
        v = vh(d, n)
        if v:
            print(f"   {n:17s}: {json.dumps({k: round(x,6) for k,x in v['fixed_set'].items()})}")
    if d.get("r33_donor"):
        print(f"   R33_full_late    : {json.dumps({k: round(x,6) for k,x in d['r33_donor'].items()})}")

KEYMAP = {"abs_bias": "lognhi_abs_bias", "std": "lognhi_std", "mae": "lognhi_mae",
          "rmse": "lognhi_rmse", "score_nhi": "score_nhi"}


def base_val(d, metric):
    return d["baseline_E_E"][KEYMAP[metric]]


def fixed_val(h, metric):
    k = "signed_bias" if metric == "abs_bias" else metric
    v = h["fixed_set"][k]
    return abs(v) if metric == "abs_bias" else v


print("\n=== direction summary: R38 vs E on fixed val matching set ===")
for metric in ("abs_bias", "std", "mae", "rmse", "score_nhi"):
    better = 0
    detail = []
    for s in SEEDS:
        d = D[s]
        if d is None:
            continue
        e = base_val(d, metric)
        r = fixed_val(vh(d, "R38_matched_WLS"), metric)
        ok = (r > e) if metric == "score_nhi" else (r < e)
        better += int(ok)
        detail.append(f"s{s} {e:.6f}->{r:.6f}{'+' if ok else '-'}")
    print(f"{metric:10s} improved {better}/3  ({'; '.join(detail)})")

print("\n=== direction summary: R38 vs R37 all-mask WLS on fixed val matching set ===")
for metric in ("abs_bias", "std", "mae", "rmse", "score_nhi"):
    better = 0
    detail = []
    for s in SEEDS:
        d = D[s]
        if d is None:
            continue
        r37 = vh(d, "R37_allmask_WLS")
        if not r37:
            continue
        a = fixed_val(r37, metric)
        r = fixed_val(vh(d, "R38_matched_WLS"), metric)
        ok = (r > a) if metric == "score_nhi" else (r < a)
        better += int(ok)
        detail.append(f"s{s} {a:.6f}->{r:.6f}{'+' if ok else '-'}")
    print(f"{metric:10s} improved {better}/3  ({'; '.join(detail)})")

print("\n=== direction summary: R38 vs AdamW refit on fixed val matching set ===")
for metric in ("abs_bias", "std", "mae", "rmse", "score_nhi"):
    better = 0
    detail = []
    for s in SEEDS:
        d = D[s]
        if d is None:
            continue
        a0 = vh(d, "AdamW_refit_ep10")
        if not a0:
            continue
        a = fixed_val(a0, metric)
        r = fixed_val(vh(d, "R38_matched_WLS"), metric)
        ok = (r > a) if metric == "score_nhi" else (r < a)
        better += int(ok)
        detail.append(f"s{s} {a:.6f}->{r:.6f}{'+' if ok else '-'}")
    print(f"{metric:10s} improved {better}/3  ({'; '.join(detail)})")

print("\n=== distribution shift (R37 all-mask vs R38 matched) ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    ds = d["distribution_shift"]
    print(f"seed {s}: feature |SMD| mean {ds['feature_abs_smd_mean']:.6f} "
          f"median {ds['median']:.6f} max {ds['max']:.6f} (dim {ds['argmax_dim']})")
    print(f"   base_lognhi  matched {ds['base_lognhi']['matched_mean']:.6f}+-{ds['base_lognhi']['matched_std']:.6f} "
          f"| allmask {ds['base_lognhi']['allmask_mean']:.6f}+-{ds['base_lognhi']['allmask_std']:.6f}")
    print(f"   truth_lognhi matched {ds['truth_lognhi']['matched_mean']:.6f}+-{ds['truth_lognhi']['matched_std']:.6f} "
          f"| allmask {ds['truth_lognhi']['allmask_mean']:.6f}+-{ds['truth_lognhi']['allmask_std']:.6f}")

print("\n=== parameter distance ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    print(f"seed {s}: {json.dumps({k: round(v,6) for k,v in d['parameter_distance'].items()})}")

print("\n=== gates ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    print(f"seed {s}: all_pass={d['gates']['all_pass']} failing="
          f"{json.dumps({k: v for k, v in d['gates'].items() if v is not True})}")

print("\n=== cost ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    print(f"seed {s}: {json.dumps(d['cost'])}")

# combined CSV
lines = ["seed,E_Final,AdamW_Final,R37_allmask_Final,R38_matched_Final,"
         "dR38_vs_E,dR38_vs_R37,dR38_vs_AdamW,"
         "J_match_E,J_match_AdamW,J_match_R37,J_match_R38,n_match_train,rank,cond_eff,"
         "fixed_E_absbias,fixed_AdamW_absbias,fixed_R37_absbias,fixed_R38_absbias,fixed_R33_absbias,"
         "fixed_E_std,fixed_AdamW_std,fixed_R37_std,fixed_R38_std,fixed_R33_std,"
         "fixed_E_mae,fixed_AdamW_mae,fixed_R37_mae,fixed_R38_mae,fixed_R33_mae,"
         "fixed_E_rmse,fixed_AdamW_rmse,fixed_R37_rmse,fixed_R38_rmse,fixed_R33_rmse,"
         "fixed_E_scorenhi,fixed_AdamW_scorenhi,fixed_R37_scorenhi,fixed_R38_scorenhi,fixed_R33_scorenhi,"
         "smd_mean,smd_median,smd_max"]


def g(dic, k, sub=None):
    if dic is None:
        return ""
    if sub is not None:
        return dic.get(sub, {}).get(k, "")
    return dic.get(k, "")


for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    a = vh(d, "AdamW_refit_ep10"); r37 = vh(d, "R37_allmask_WLS"); r38 = vh(d, "R38_matched_WLS")
    t = d["train_matched_objective"]
    fs = lambda h, k: (h["fixed_set"][k] if h else "")
    r33 = d.get("r33_donor") or {}
    lines.append(",".join(str(v) for v in [
        s, d["baseline_E_E"]["final"], g(a, "final", "metrics"), g(r37, "final", "metrics"),
        g(r38, "final", "metrics"),
        r38["Delta_vs_E"], r38["metrics"]["final"] - r37["metrics"]["final"],
        r38["metrics"]["final"] - a["metrics"]["final"],
        t["E"]["J_match_mean_sq"], t.get("AdamW_refit_ep10", {}).get("J_match_mean_sq", ""),
        t.get("R37_allmask_WLS", {}).get("J_match_mean_sq", ""), t["R38_matched_WLS"]["J_match_mean_sq"],
        d["train_catalog"]["train_n_match"], d["solver"]["rank"],
        d["solver"]["effective_cond_smax_over_s_rank"],
        d["baseline_E_E"]["lognhi_abs_bias"], abs(g(a, "signed_bias", "fixed_set") or 0) if a else "",
        abs(fs(r37, "signed_bias") or 0) if r37 else "", abs(fs(r38, "signed_bias")), r33.get("abs_bias", ""),
        d["baseline_E_E"]["lognhi_std"], fs(a, "std"), fs(r37, "std"), fs(r38, "std"), r33.get("std", ""),
        d["baseline_E_E"]["lognhi_mae"], fs(a, "mae"), fs(r37, "mae"), fs(r38, "mae"), r33.get("mae", ""),
        d["baseline_E_E"]["lognhi_rmse"], fs(a, "rmse"), fs(r37, "rmse"), fs(r38, "rmse"), r33.get("rmse", ""),
        d["baseline_E_E"]["score_nhi"], fs(a, "score_nhi"), fs(r37, "score_nhi"), fs(r38, "score_nhi"), r33.get("score_nhi", ""),
        d["distribution_shift"]["feature_abs_smd_mean"], d["distribution_shift"]["median"],
        d["distribution_shift"]["max"],
    ]) + "\n")
(OUT / "r38_summary_all.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nwrote", OUT / "r38_summary_all.csv")
