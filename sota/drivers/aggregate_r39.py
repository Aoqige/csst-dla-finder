#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R39: merge seed42/43/44 (R38) with the prospective seed45 (R39) endpoint."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

R38 = Path(os.environ.get("CSST_R38_DIR", str(Path.home() / "csst_dla_runs" / "20261005_r38")))
R39 = Path(os.environ.get("CSST_R39_DIR", str(Path.home() / "csst_dla_runs" / "20261005_r39")))
SEEDS = ("42", "43", "44", "45")


def load(s):
    p = (R38 / f"r38_matched_wls_seed{s}.json") if s != "45" else (R39 / f"r39_matched_wls_seed45.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


D = {s: load(s) for s in SEEDS}

print("=== seed42-45 core table ===")
print("seed | E Final | matched-WLS Final | dFinal | Delta_vs_E")
rows = []
for s in SEEDS:
    d = D[s]
    if d is None:
        print(f"{s} | MISSING")
        continue
    e = d["baseline_E_E"]["final"]
    w = d["val_heads"]["R38_matched_WLS"]["metrics"]["final"]
    rows.append((s, e, w, w - e))
    print(f"{s} | {e:.10f} | {w:.10f} | {w-e:+.10f} | {d['val_heads']['R38_matched_WLS']['Delta_vs_E']:+.10f}")

ds = [r[3] for r in rows]
print(f"\nDelta: mean {st.mean(ds):+.10f} median {st.median(ds):+.10f} "
      f"min {min(ds):+.10f} max {max(ds):+.10f} pos/neg/zero "
      f"{sum(1 for v in ds if v>0)}/{sum(1 for v in ds if v<0)}/{sum(1 for v in ds if v==0)}")

print("\n=== seed45 detail ===")
d = D["45"]
if d is None:
    print("seed45 MISSING")
else:
    print("train_catalog:", json.dumps(d["train_catalog"]))
    print("design:", json.dumps(d["design_matrix"]))
    so = d["solver"]
    print(f"solver: rank {so['rank']}/{so['n_singular_values']} full={so['full_column_rank']} "
          f"s_max {so['s_max']:.6f} s_min {so['s_min']:.6e} cond_eff {so['effective_cond_smax_over_s_rank']:.3f}")
    print(f"  stationarity abs {so['stationarity']['abs_norm_AT_resid']:.3e} "
          f"rel {so['stationarity']['rel_vs_AT_d']:.3e} trunc {so['stationarity']['truncation_used']}")
    print(f"  repeatable {so['repeatability']['bitwise_identical']} cpu_fit {so['cpu_fit_seconds']:.3f}s")
    print("  J_match:", json.dumps({k: v["J_match_mean_sq"] for k, v in d["train_matched_objective"].items()}))
    print("  train resid:", json.dumps({k: {kk: round(vv, 6) for kk, vv in v.items() if kk != "J_match_mean_sq"}
                                        for k, v in d["train_matched_objective"].items()}))
    print("  sanity:", json.dumps(d["sanity_readout"]))
    print("  deploy:", json.dumps(d["deployment"]))
    print("  decoder_replica all_equal:", d["decoder_replica_check"]["all_equal"])
    print("  gates:", json.dumps(d["gates"]))
    print("  detection_bins identity:", d["detection_bins"]["identity_holds"],
          "sum_norm", d["detection_bins"]["delta_detection_sum_of_bins_normalised"],
          "from_scores", d["detection_bins"]["delta_detection_from_scores"])
    print("  E/E metrics:", json.dumps({k: d["baseline_E_E"][k] for k in
                                        ("final", "detection", "parameter", "score_z", "score_nhi",
                                         "precision", "recall", "n_pred", "n_match")}))
    print("  R39 metrics:", json.dumps({k: d["val_heads"]["R38_matched_WLS"]["metrics"][k] for k in
                                        ("final", "detection", "parameter", "score_z", "score_nhi",
                                         "precision", "recall", "n_pred", "n_match")}))
    print("  fixed E:", json.dumps(d["baseline_E_E"] and
                                   {k: round(d["baseline_E_E"][k], 6) for k in
                                    ("lognhi_signed_bias", "lognhi_std", "lognhi_mae", "lognhi_rmse", "score_nhi")}))
    print("  fixed R39:", json.dumps({k: round(v, 6) for k, v in d["val_heads"]["R38_matched_WLS"]["fixed_set"].items()}))
    print("  distribution_shift:", json.dumps({k: v for k, v in d["distribution_shift"].items()
                                               if k in ("feature_abs_smd_mean", "median", "max")}))
    print("  cost:", json.dumps(d["cost"]))

print("\n=== four-seed residual direction (R39/R38 matched-WLS vs E) ===")
KEY = {"abs_bias": "lognhi_abs_bias", "std": "lognhi_std", "mae": "lognhi_mae",
       "rmse": "lognhi_rmse", "score_nhi": "score_nhi"}
for metric in ("abs_bias", "std", "mae", "rmse", "score_nhi"):
    better, detail = 0, []
    for s in SEEDS:
        d = D[s]
        if d is None:
            continue
        e = d["baseline_E_E"][KEY[metric]]
        fs = d["val_heads"]["R38_matched_WLS"]["fixed_set"]
        r = abs(fs["signed_bias"]) if metric == "abs_bias" else fs[metric]
        ok = (r > e) if metric == "score_nhi" else (r < e)
        better += int(ok)
        detail.append(f"s{s} {e:.6f}->{r:.6f}{'+' if ok else '-'}")
    print(f"{metric:10s} improved {better}/4  ({'; '.join(detail)})")

print("\n=== fixed val matching set, all four seeds ===")
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    fs = d["val_heads"]["R38_matched_WLS"]["fixed_set"]
    print(f"seed {s}: E bias {d['baseline_E_E']['lognhi_signed_bias']:+.6f} std {d['baseline_E_E']['lognhi_std']:.6f} "
          f"mae {d['baseline_E_E']['lognhi_mae']:.6f} rmse {d['baseline_E_E']['lognhi_rmse']:.6f} "
          f"score_nhi {d['baseline_E_E']['score_nhi']:.6f}")
    print(f"          R39 bias {fs['signed_bias']:+.6f} std {fs['std']:.6f} "
          f"mae {fs['mae']:.6f} rmse {fs['rmse']:.6f} score_nhi {fs['score_nhi']:.6f}")

lines = ["seed,E_Final,matched_WLS_Final,Delta_vs_E,absbias_E,absbias_R39,std_E,std_R39,"
         "mae_E,mae_R39,rmse_E,rmse_R39,scorenhi_E,scorenhi_R39,n_match_train,rank,cond_eff"]
for s in SEEDS:
    d = D[s]
    if d is None:
        continue
    fs = d["val_heads"]["R38_matched_WLS"]["fixed_set"]
    lines.append(",".join(str(v) for v in [
        s, d["baseline_E_E"]["final"], d["val_heads"]["R38_matched_WLS"]["metrics"]["final"],
        d["val_heads"]["R38_matched_WLS"]["Delta_vs_E"],
        d["baseline_E_E"]["lognhi_abs_bias"], abs(fs["signed_bias"]),
        d["baseline_E_E"]["lognhi_std"], fs["std"],
        d["baseline_E_E"]["lognhi_mae"], fs["mae"],
        d["baseline_E_E"]["lognhi_rmse"], fs["rmse"],
        d["baseline_E_E"]["score_nhi"], fs["score_nhi"],
        d["train_catalog"]["train_n_match"], d["solver"]["rank"],
        d["solver"]["effective_cond_smax_over_s_rank"],
    ]) + "\n")
(R39 / "r39_summary_all.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\nwrote", R39 / "r39_summary_all.csv")
