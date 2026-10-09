#!/usr/bin/env python3
"""R48-SH -- aggregate the 64 matched-WLS candidate results into the ranking table.

Reads  <root>/cand/seed{S}_ep{E}/r38_matched_wls_seed{S}.json  for
S in 46..53 and E in {6,8,10,12,14,16,18,20}, checks the integrity gates, and writes:

  r48sh_leaderboard.csv   full 64-row table sorted by VAL Final (desc)
  r48sh_summary.json      highlights required by section 8
  r48sh_report.md         fact report

Primary selection: max Final over all candidates.  Ties -> earlier EMA epoch, then smaller seed.
No TEST scoring, no threshold sweep, no parameter ensemble.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, os
from pathlib import Path

SEEDS = [46, 47, 48, 49, 50, 51, 52, 53]
EPOCHS = [6, 8, 10, 12, 14, 16, 18, 20]
HIST_BEST = 0.687357          # R21 count_loss_weight=0.10 best
TARGET = 0.700000
HIST_REF = [("42", "R38 matched-WLS", 0.6825514434232297),
            ("43", "R38 matched-WLS", 0.6831270401484937),
            ("44", "R38 matched-WLS", 0.6829074029348012),
            ("45", "R39 matched-WLS", 0.6870866739276652),
            ("21", "count_loss_weight=0.10 best", 0.687357)]

METRIC_KEYS = ("final", "detection", "parameter", "precision", "recall", "n_pred", "n_match",
               "score_z", "score_nhi", "z_signed_bias", "z_std", "lognhi_signed_bias",
               "lognhi_abs_bias", "lognhi_std", "lognhi_mae", "lognhi_rmse")


def sha256_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="R48-SH run root (contains cand/)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", default=",".join(str(e) for e in EPOCHS))
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x]
    epochs = [int(x) for x in args.epochs.split(",") if x]

    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root
    out.mkdir(parents=True, exist_ok=True)
    cand_root = root / "cand"

    rows, missing = [], []
    for seed in seeds:
        for ep in epochs:
            d = cand_root / f"seed{seed}_ep{ep}"
            j = d / f"r38_matched_wls_seed{seed}.json"
            if not j.exists():
                missing.append({"seed": seed, "ema_epoch": ep, "dir": str(d)})
                continue
            r = json.loads(j.read_text(encoding="utf-8"))
            m = r["val_heads"]["R38_matched_WLS"]["metrics"]
            g = r.get("gates", {})
            dep = d / f"r38_deployable_seed{seed}.pt"
            hd = d / f"r38_matched_wls_head_seed{seed}.pt"
            row = {"seed": seed, "ema_epoch": ep, "cand_dir": str(d)}
            for k in METRIC_KEYS:
                row[k] = m.get(k)
            row["baseline_E_E_final"] = r.get("baseline_E_E", {}).get("final")
            row["delta_vs_E_E"] = r.get("main", {}).get("Delta_matchWLS_vs_E")
            row["e_ckpt"] = r.get("e_ckpt")
            row["e_ckpt_md5"] = r.get("e_ckpt_md5")
            row["wls_sample_count"] = r.get("train_catalog", {}).get("wls_sample_count")
            row["solver_rank"] = r.get("solver", {}).get("rank")
            row["solver_cond"] = r.get("solver", {}).get("cond")
            row["gates_all_pass"] = bool(g.get("all_pass"))
            row["gates_failed"] = [k for k, v in g.items() if v is False]
            row["deployable_sha256"] = sha256_file(dep) if dep.exists() else None
            row["head_sha256"] = sha256_file(hd) if hd.exists() else None
            row["fatal"] = r.get("fatal")
            rows.append(row)

    rows.sort(key=lambda x: (-(x["final"] if x["final"] is not None else -1.0),
                             x["ema_epoch"], x["seed"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    hdr = (["rank", "seed", "ema_epoch", "final", "detection", "parameter", "precision", "recall",
            "n_pred", "n_match", "score_z", "score_nhi", "z_signed_bias", "z_std",
            "lognhi_signed_bias", "lognhi_std", "lognhi_mae", "lognhi_rmse",
            "baseline_E_E_final", "delta_vs_E_E", "wls_sample_count", "solver_rank", "solver_cond",
            "gates_all_pass", "gates_failed", "e_ckpt", "e_ckpt_md5",
            "deployable_sha256", "head_sha256", "cand_dir", "fatal"])
    with (out / "r48sh_leaderboard.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for r in rows:
            w.writerow([r.get(k) for k in hdr])

    finals = [r["final"] for r in rows if r["final"] is not None]
    best = rows[0] if rows else None
    n_all_gates = sum(1 for r in rows if r["gates_all_pass"])
    summary = {
        "round": "R48-SH",
        "n_candidates_expected": len(seeds) * len(epochs),
        "n_candidates_scored": len(rows),
        "n_missing": len(missing),
        "missing": missing,
        "n_completed_all_gates_pass": n_all_gates,
        "n_candidates_all_gates_pass": n_all_gates,
        "all_candidates_complete_and_passed": bool(len(rows) == len(seeds) * len(epochs)
                                                   and n_all_gates == len(rows)),
        "selection_rule": "max VAL Final over all candidates; tie -> earlier EMA epoch, then smaller seed",
        "selection_status": "repeated-development selection on unified VAL (not an unbiased generalization claim)",
        "best": None if best is None else {
            "seed": best["seed"], "ema_epoch": best["ema_epoch"], "final": best["final"],
            "detection": best["detection"], "parameter": best["parameter"],
            "e_ckpt": best["e_ckpt"], "e_ckpt_md5": best["e_ckpt_md5"],
            "deployable_sha256": best["deployable_sha256"], "head_sha256": best["head_sha256"],
            "cand_dir": best["cand_dir"]},
        "final_max": max(finals) if finals else None,
        "final_min": min(finals) if finals else None,
        "final_median": sorted(finals)[len(finals) // 2] if finals else None,
        "beats_history_0.687357": [{"seed": r["seed"], "ema_epoch": r["ema_epoch"], "final": r["final"]}
                                   for r in rows if r["final"] is not None and r["final"] > HIST_BEST],
        "n_beats_0.687357": sum(1 for v in finals if v > HIST_BEST),
        "n_above_0.690000": sum(1 for v in finals if v > 0.690000),
        "n_above_0.695000": sum(1 for v in finals if v > 0.695000),
        "n_above_0.700000": sum(1 for v in finals if v > 0.700000),
        "gap_to_0.700000": (TARGET - max(finals)) if finals else None,
        "gap_best_to_history": (max(finals) - HIST_BEST) if finals else None,
        "max_final_per_ema_epoch": {str(e): (max([r["final"] for r in rows
                                                  if r["ema_epoch"] == e and r["final"] is not None],
                                                 default=None)) for e in epochs},
        "n_candidates_per_ema_epoch": {str(e): sum(1 for r in rows if r["ema_epoch"] == e)
                                       for e in epochs},
        "historical_reference_readonly": [
            {"seed": s, "source": src, "val_final": v} for s, src, v in HIST_REF],
        "notes": [
            "TEST was not scored, not tuned and not used for selection in this round.",
            "All numbers come from the official float32 scoring path of the frozen scorer.",
            "The VAL maximum is a repeated-development selection over 64 candidates."],
    }
    (out / "r48sh_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    L = ["# R48-SH — Unified VAL Single-Checkpoint Score Hunt", "",
         "## 1. 实验改了什么", "",
         "在 unified TRAIN 上从头训练 8 个新 seed（46–53），每个 20 epoch，EMA beta=0.999；",
         "对 8 个 EMA endpoint（6/8/10/12/14/16/18/20）各跑一次 R38 式 matched-WLS 读出，共 64 个候选；",
         "只在 unified VAL 上按 Final 取最大。TEST 未评分、未调参、未用于选择。", "",
         "## 2. 结果", "",
         f"- 候选总数（预期 / 实际评分）：{summary['n_candidates_expected']} / {summary['n_candidates_scored']}",
         f"- 缺失候选数：{summary['n_missing']}",
         f"- 全部 gate 通过的候选数：{n_all_gates}",
         f"- 全部候选完成且通过完整性核查：{summary['all_candidates_complete_and_passed']}",
         f"- 最高单模型 VAL Final：{summary['final_max']}",
         f"- 最低 / 中位：{summary['final_min']} / {summary['final_median']}",
         f"- 超过 0.687357 的候选数：{summary['n_beats_0.687357']}",
         f"- 超过 0.690000 / 0.695000 / 0.700000：{summary['n_above_0.690000']} / "
         f"{summary['n_above_0.695000']} / {summary['n_above_0.700000']}",
         f"- 距 0.700000：{summary['gap_to_0.700000']}",
         f"- 最高点相对历史 0.687357：{summary['gap_best_to_history']}", ""]
    if best is not None:
        L += [f"最高点：seed **{best['seed']}**，EMA epoch **{best['ema_epoch']}**，",
              f"Final {best['final']} = 0.6×{best['detection']} + 0.4×{best['parameter']}；",
              f"checkpoint `{best['e_ckpt']}`（md5 `{best['e_ckpt_md5']}`）。", ""]
    L += ["### 各 EMA epoch 的最高分", "",
          "| EMA epoch | 候选数 | 该 epoch 最高 Final |", "|---:|---:|---:|"]
    for e in epochs:
        L.append(f"| {e} | {summary['n_candidates_per_ema_epoch'][str(e)]} | "
                 f"{summary['max_final_per_ema_epoch'][str(e)]} |")
    L += ["", "### 历史只读参考（不进入 64 候选预算）", "",
          "| seed | 来源 | VAL Final |", "|---|---|---:|"]
    for s, src, v in HIST_REF:
        L.append(f"| {s} | {src} | {v} |")
    L += ["", "## 3. 实验有效性", "",
          "- 评分沿用 R42/R47b 已核实的官方 float32 口径；未在评分前把物理预测字段转 float64。",
          "- 每个候选的完整性 gate（两塔权重未变、WLS 只改 lognhi_delta、几何与 matched 集合不变、",
          "  checkpoint 可重载、gate 全过）由 R38 Stage-B 脚本自身记录在 `gates` 字段。",
          "- 未做 threshold sweep、count-bias search、LR/EMA sweep、WLS ridge/weight search、",
          "  seed 扩展、epoch 追加、参数 ensemble、split 更换；未做 TEST 评分。",
          "- 最高分属 repeated-development selection（unified VAL），不构成无偏泛化成绩。", "",
          "## 4. 尚缺的数据", "",
          "- 本表只覆盖 8 seed × 8 EMA endpoint；未覆盖其它 epoch 或其它 seed。",
          "- 未评估所选最高点在其它 split 上的行为（本轮禁止 TEST 评分）。",
          "- 未做多候选一致性/稳定性统计（本轮目标为 single-checkpoint maximum）。", ""]
    (out / "r48sh_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(json.dumps({"n_scored": len(rows), "n_missing": len(missing),
                      "n_gates_pass": n_all_gates, "final_max": summary["final_max"],
                      "best": summary["best"], "n_above_0.7": summary["n_above_0.700000"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
