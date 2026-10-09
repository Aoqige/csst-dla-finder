#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_fusion_ema_lrstep.py -- R32: passive parameter-EMA + ONE fixed LR drop.

This file is a **copy of ~/r30/run_fusion_ema.py** with exactly three additive
changes; the R30/R31 sources are untouched.

  1. `--lr-drop-epoch` (default 11) / `--lr-after-drop` (default 5e-5):
     at the top of training epoch `lr-drop-epoch` -- i.e. AFTER epoch 10's training
     and validation have fully finished, and BEFORE the first `optimizer.step()` of
     epoch 11 -- every EXISTING optimizer param_group's `lr` is set to `lr-after-drop`.
     Nothing else is touched: optimizer state, EMA state, EMA update count and
     global_step are all preserved.
  2. Per-epoch `lr_used` (the lr actually in force for that epoch's optimizer steps)
     is recorded in the history rows and in `ema_run_info.json -> lr_trace`.
  3. Epoch-10 checkpoints + an epoch-10 resume state, and a **front-10 regression
     gate** comparing epochs 0-10 against the same seed's R30 histories
     (`--front10-ref-online` / `--front10-ref-ema` / `--front10-ref-runinfo`).
     If that gate fails the process exits before training epoch 11.

Everything else (construction path, EMA protocol, decoder, scoring, evaluation
loop, saving) is byte-for-byte the R30 driver.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("FUS_SUBSET", "both")
EMA_BASE = os.environ.get("EMA_BASE", "subset")

for _p in ("/home/heruihua/csst-dla-finder/src",
           "/home/heruihua/csst-dla-finder/hybrid_ensemble",
           "/home/heruihua", "/home/heruihua/r11", "/home/heruihua/r27",
           "/home/heruihua/r30", "/home/heruihua/r32"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
from torch.utils.data import DataLoader                               # noqa: E402

if EMA_BASE == "sg_off":
    os.environ.setdefault("SG_MODE", "off")
    import model_sg                                                    # noqa: E402
    BASE_CLS = model_sg.TrainNet
else:
    import run_fusion_subset as RS                                     # noqa: E402
    BASE_CLS = RS.SubsetFusionNet

import train_feature_fusion as TFF                                     # noqa: E402
from csst_dla.scoring import greedy_match, C_KMS                       # noqa: E402
from ema import ParameterEMA                                           # noqa: E402

TFF.DualTowerFusionNet = BASE_CLS
BETA = 0.999


# --------------------------------------------------------------------------- build
def build(args):
    """Replicate `train_feature_fusion.main()` construction order exactly."""
    device = TFF.resolve_device(args.device)
    dilated_wrapper, dilated_config = TFF.load_checkpoint(args.dilated_checkpoint, device)
    wzx_model, wzx_config = TFF.load_model_from_checkpoint(args.wzx_checkpoint, device)
    args.dilated_input_mode = str(dilated_config.get("input_mode", "all"))
    args.wzx_feature_mode = str(wzx_config.get("feature_mode", "all"))
    print(json.dumps({"stage": "loading_datasets", "merge_mode": args.merge_mode,
                      "dilated_input_mode": args.dilated_input_mode,
                      "wzx_feature_mode": args.wzx_feature_mode}, ensure_ascii=False), flush=True)
    train_ds = TFF.DualFusionTrainDataset(
        args.targets, args.train_fits, "train", args.max_train_samples,
        dilated_input_mode=args.dilated_input_mode, wzx_feature_mode=args.wzx_feature_mode)
    val_ds = TFF.DualFusionTrainDataset(
        args.targets, args.train_fits, "val", args.max_val_samples,
        dilated_input_mode=args.dilated_input_mode, wzx_feature_mode=args.wzx_feature_mode)
    print(json.dumps({"stage": "datasets_ready", "train": len(train_ds), "val": len(val_ds)},
                     ensure_ascii=False), flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)
    model = BASE_CLS(
        dilated_wrapper.model, wzx_model,
        merge_mode=args.merge_mode, width=args.fusion_width, depth=args.fusion_depth,
        freeze_backbones=not args.train_backbones,
        freeze_backbone_norm=bool(args.freeze_bn),
    ).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=1e-4)
    print(json.dumps({
        "stage": "unfreeze_protocol", "train_backbones": bool(args.train_backbones),
        "freeze_bn": bool(args.freeze_bn), "backbone_lr": args.backbone_lr, "lr": args.lr,
        "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_total": int(sum(p.numel() for p in model.parameters())),
        "optimizer_groups": [{"lr": g["lr"], "n_params": int(sum(p.numel() for p in g["params"]))}
                             for g in optimizer.param_groups]}, ensure_ascii=False), flush=True)
    return dict(device=device, model=model, optimizer=optimizer, train_ds=train_ds, val_ds=val_ds,
                train_loader=train_loader, val_loader=val_loader,
                dilated_config=dilated_config, wzx_config=wzx_config)


# ------------------------------------------------------------------------ predict
@torch.inference_mode()
def predict_dual(model_online, model_ema, loader, device):
    """One pass over `loader`; both models see the *same* batch tensors."""
    model_online.eval()
    model_ema.eval()
    acc = {t: {"heatmap": [], "lognhi": [], "offset": [], "count_logits": []}
           for t in ("online", "ema")}
    maxdiff = 0.0
    for batch in loader:
        hy = batch[0].to(device, non_blocking=True)
        wz = batch[1].to(device, non_blocking=True)
        zq = batch[2].to(device, non_blocking=True)
        outs = {"online": model_online(hy, wz, zq), "ema": model_ema(hy, wz, zq)}
        for tag in ("online", "ema"):
            o = outs[tag]
            acc[tag]["heatmap"].append(torch.sigmoid(o["center_logits"]).float().cpu().numpy())
            acc[tag]["lognhi"].append((20.3 + o["lognhi_raw"]).float().cpu().numpy())
            acc[tag]["offset"].append(o["offset_raw"].float().cpu().numpy())
            acc[tag]["count_logits"].append(o["count_logits"].float().cpu().numpy())
        for key in ("center_logits", "region_logits", "lognhi_raw", "offset_raw", "count_logits"):
            d = float((outs["online"][key] - outs["ema"][key]).abs().max().item())
            if d > maxdiff:
                maxdiff = d
    pred = {}
    for tag in ("online", "ema"):
        pred[tag] = {k: np.concatenate(v) for k, v in acc[tag].items()}
    return pred, maxdiff


def score_pred(pred, val_ds, args, epoch, loss=None, weight_kind=""):
    """Official decoder + official scorer; residual metrics use ddof=0."""
    catalog = TFF.decode_validation_catalog(
        pred, val_ds.indices, val_ds.labels, val_ds.wavelength,
        threshold=args.threshold, min_distance=10, min_z_dla=args.min_z_dla,
        lognhi_min=20.3, lognhi_max=22.5)
    truth = TFF.labels_to_truth(val_ds.labels, val_ds.indices, min_lognhi=20.3)
    score = TFF.score_catalog(truth, catalog)

    # Signed residuals on the official matched set (same greedy_match as the scorer).
    matches = greedy_match(truth, catalog)
    if matches:
        ti = np.asarray([m[0] for m in matches], dtype=int)
        pi = np.asarray([m[1] for m in matches], dtype=int)
        dv = C_KMS * (catalog["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
        dl = catalog["LOG_NHI"][pi] - truth["LOG_NHI"][ti]
        z_bias = float(dv.mean()); z_std = float(dv.std()); z_mae = float(np.abs(dv).mean())
        z_rmse = float(np.sqrt((dv ** 2).mean()))
        dl_bias = float(dl.mean()); dl_std = float(dl.std()); dl_mae = float(np.abs(dl).mean())
        dl_rmse = float(np.sqrt((dl ** 2).mean()))
        score_z = float(np.exp(-z_std / 300.0) * np.exp(-abs(z_bias) / 150.0))
        score_nhi = float(np.exp(-dl_std / 0.25) * np.exp(-abs(dl_bias) / 0.1))
    else:
        z_bias = z_std = z_mae = z_rmse = 0.0
        dl_bias = dl_std = dl_mae = dl_rmse = 0.0
        score_z = score_nhi = 0.0
    # the recomputation must reproduce the official parameter score
    assert abs(0.5 * (score_z + score_nhi) - score.parameter_score) < 1e-9, \
        f"parameter recompute mismatch {score_z}+{score_nhi} vs {score.parameter_score}"

    return {"epoch": epoch, "loss": loss, "weight_kind": weight_kind,
            "final_score": score.final_score, "detection_score": score.detection_score,
            "parameter_score": score.parameter_score, "completeness": score.completeness,
            "purity": score.purity, "n_pred": score.n_pred, "n_match": score.n_match,
            "std_dv": score.std_dv, "std_dlognhi": score.std_dlognhi,
            "z_signed_bias": z_bias, "z_std": z_std, "z_mae": z_mae, "z_rmse": z_rmse,
            "lognhi_signed_bias": dl_bias, "lognhi_std": dl_std,
            "lognhi_mae": dl_mae, "lognhi_rmse": dl_rmse,
            "score_z": score_z, "score_nhi": score_nhi}


def save_ckpt(path: Path, module, args, row, dcfg, wcfg):
    TFF.save_checkpoint(path, module, args, row, dcfg, wcfg)


def save_resume(path: Path, args, model, ema, optimizer, global_step, epoch, lr_trace):
    """Complete resume state: online + EMA + optimizer + all RNG + counters + LR stage."""
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "ema_update_count": ema.n_updates,
        "beta": BETA,
        "online_model_state": model.state_dict(),
        "ema_shadow_state": ema.shadow.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_lrs": [float(g["lr"]) for g in optimizer.param_groups],
        "lr_trace": lr_trace,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "ema_whitelist": list(ema.whitelist),
        "args": vars(args),
    }, path)


# --------------------------------------------------------------- front-10 gate
def _leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from _leaves(obj[k], f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare_histories(cur, ref, max_epoch):
    """Bitwise comparison of every common numeric leaf for epochs 0..max_epoch."""
    worst, first, ncmp = 0.0, None, 0
    for a, b in zip(cur, ref):
        ea, eb = a.get("epoch"), b.get("epoch")
        if ea != eb:
            return {"pass": False, "reason": f"epoch label mismatch {ea} vs {eb}"}
        if ea > max_epoch:
            break
        da, db = dict(_leaves(a)), dict(_leaves(b))
        for k in sorted(set(da) & set(db)):
            va, vb = da[k], db[k]
            if isinstance(va, bool) or isinstance(vb, bool):
                continue
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = abs(float(va) - float(vb))
                ncmp += 1
                if d > worst:
                    worst = d
                if d > 0 and first is None:
                    first = {"epoch": ea, "field": k, "cur": va, "ref": vb, "absdiff": d}
    return {"pass": worst == 0.0, "n_compared": ncmp, "max_abs_diff": worst,
            "first_divergence": first, "fields_only_in_cur": None}


# ---------------------------------------------------------------------------- main
def main() -> None:
    # allow_abbrev=False is required: otherwise the pre-parser would treat the
    # trainer's own `--lr` as an ambiguous abbreviation of --lr-drop-epoch /
    # --lr-after-drop and refuse to start.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--lr-drop-epoch", type=int, default=11)
    pre.add_argument("--lr-after-drop", type=float, default=5e-5)
    pre.add_argument("--front10-ref-online", default=None)
    pre.add_argument("--front10-ref-ema", default=None)
    pre.add_argument("--front10-ref-runinfo", default=None)
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    args = TFF.parse_args()
    args.lr_drop_epoch = known.lr_drop_epoch
    args.lr_after_drop = known.lr_after_drop
    args.front10_ref_online = known.front10_ref_online
    args.front10_ref_ema = known.front10_ref_ema
    args.front10_ref_runinfo = known.front10_ref_runinfo

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = build(args)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_ds, val_ds = ctx["train_ds"], ctx["val_ds"]
    train_loader, val_loader = ctx["train_loader"], ctx["val_loader"]
    dcfg, wcfg = ctx["dilated_config"], ctx["wzx_config"]

    ema = ParameterEMA(model, beta=BETA)
    info = ema.info()
    print(json.dumps({"stage": "ema_init", **{k: v for k, v in info.items() if k != "whitelist"},
                      "whitelist_n": len(info["whitelist"])}, ensure_ascii=False), flush=True)
    (out_dir / "ema_whitelist.json").write_text(
        json.dumps({"beta": BETA, "whitelist": info["whitelist"],
                    "n_ema_tensors": info["n_ema_tensors"], "n_ema_params": info["n_ema_params"],
                    "n_frozen_params_copied": info["n_frozen_params_copied"]}, indent=2), encoding="utf-8")
    if info["n_ema_params"] != 553479:
        raise SystemExit(f"FATAL: EMA 覆盖参数数 {info['n_ema_params']} != 553479")

    history_online, history_ema = [], []
    best_online, best_ema = -float("inf"), -float("inf")
    rng_trace, lr_trace = [], []
    global_step = 0
    front10 = None

    def evaluate(epoch, loss):
        ema.sync_non_ema_state()          # exact copy of all non-EMA state
        pred, maxdiff = predict_dual(model, ema.shadow, val_loader, device)
        row_o = score_pred(pred["online"], val_ds, args, epoch, loss, "online")
        row_e = score_pred(pred["ema"], val_ds, args, epoch, loss, "ema")
        return row_o, row_e, maxdiff

    row_o0, row_e0, md0 = evaluate(0, None)
    history_online.append(row_o0)
    history_ema.append(row_e0)
    best_online, best_ema = row_o0["final_score"], row_e0["final_score"]
    save_ckpt(out_dir / "initial_model.pt", model, args, row_o0, dcfg, wcfg)
    save_ckpt(out_dir / "ema_initial_model.pt", ema.shadow, args, row_e0, dcfg, wcfg)
    print(json.dumps({"epoch": 0, "online": row_o0, "ema": row_e0,
                      "ema_vs_online_maxabs": md0}, ensure_ascii=False), flush=True)

    for epoch in range(1, args.epochs + 1):
        # ---- the ONLY schedule change: applied before epoch 11's first optimizer.step(),
        #      i.e. after epoch 10's training and validation are fully finished.
        if epoch == args.lr_drop_epoch:
            for g in optimizer.param_groups:
                g["lr"] = float(args.lr_after_drop)
            print(json.dumps({"stage": "lr_drop", "at_training_epoch": epoch,
                              "lr_after": float(args.lr_after_drop),
                              "group_lrs": [float(g["lr"]) for g in optimizer.param_groups],
                              "global_step": global_step,
                              "ema_update_count": ema.n_updates,
                              "note": "optimizer state / EMA state / counters untouched"},
                             ensure_ascii=False), flush=True)
        lr_used = float(optimizer.param_groups[0]["lr"])
        model.train()
        total_loss = 0.0
        head_sum = {k: 0.0 for k in TFF.HEAD_KEYS}
        progress_every = max(1, len(train_loader) // 5)
        h = hashlib.sha256()
        steps_this_epoch = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            loss, _, head_losses = TFF.loss_for_batch(model, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update()
            global_step += 1
            steps_this_epoch += 1
            total_loss += float(loss.detach().cpu()) * len(batch[0])
            for key in TFF.HEAD_KEYS:
                head_sum[key] += head_losses[key] * len(batch[0])
            h.update(np.asarray(batch[-1], dtype=np.int64).tobytes())
            if batch_index == 1 or batch_index % progress_every == 0:
                print(json.dumps({"progress": "train", "epoch": epoch, "batch": batch_index,
                                  "total_batches": len(train_loader)}, ensure_ascii=False), flush=True)
        row_o, row_e, maxdiff = evaluate(epoch, total_loss / len(train_ds))
        epoch_head_losses = {k: v / len(train_ds) for k, v in head_sum.items()}
        row_o["head_losses"] = epoch_head_losses
        row_o["head_budget"] = TFF.head_budget(epoch_head_losses, args)
        row_o["global_step"] = global_step
        row_o["lr_used"] = lr_used
        row_o["optimizer_steps_this_epoch"] = steps_this_epoch
        row_e["global_step"] = global_step
        row_e["ema_update_count"] = ema.n_updates
        row_e["lr_used"] = lr_used
        row_e["optimizer_steps_this_epoch"] = steps_this_epoch
        history_online.append(row_o)
        history_ema.append(row_e)
        rng_trace.append({"epoch": epoch, "roworder_sha256": h.hexdigest(),
                          "global_step": global_step, "ema_update_count": ema.n_updates,
                          "lr_used": lr_used, "optimizer_steps_this_epoch": steps_this_epoch})
        lr_trace.append({"epoch": epoch, "lr_used": lr_used,
                         "group_lrs": [float(g["lr"]) for g in optimizer.param_groups],
                         "optimizer_steps_this_epoch": steps_this_epoch,
                         "global_step": global_step, "ema_update_count": ema.n_updates})
        print(json.dumps({"epoch": epoch, "online": row_o, "ema": row_e,
                          "ema_vs_online_maxabs": maxdiff, "actual_lr": lr_used,
                          "optimizer_steps_this_epoch": steps_this_epoch,
                          "global_step": global_step,
                          "ema_update_count": ema.n_updates}, ensure_ascii=False), flush=True)

        if row_o["final_score"] > best_online:
            best_online = row_o["final_score"]
            save_ckpt(out_dir / "best_model.pt", model, args, row_o, dcfg, wcfg)
        if row_e["final_score"] > best_ema:
            best_ema = row_e["final_score"]
            save_ckpt(out_dir / "ema_best_model.pt", ema.shadow, args, row_e, dcfg, wcfg)
        if epoch >= 36:
            save_ckpt(out_dir / f"online_ep{epoch}.pt", model, args, row_o, dcfg, wcfg)
            save_ckpt(out_dir / f"ema_ep{epoch}.pt", ema.shadow, args, row_e, dcfg, wcfg)

        # ---- epoch-10 boundary: save + front-10 regression gate (BEFORE epoch 11) ----
        if epoch == args.lr_drop_epoch - 1:
            save_ckpt(out_dir / f"online_ep{epoch}.pt", model, args, row_o, dcfg, wcfg)
            save_ckpt(out_dir / f"ema_ep{epoch}.pt", ema.shadow, args, row_e, dcfg, wcfg)
            save_resume(out_dir / f"resume_state_ep{epoch}.pt", args, model, ema,
                        optimizer, global_step, epoch, lr_trace)
            (out_dir / f"history_ep{epoch}_partial.json").write_text(
                json.dumps({"online": history_online, "ema": history_ema,
                            "roworder_trace": rng_trace, "lr_trace": lr_trace}, indent=2),
                encoding="utf-8")
            front10 = {"epoch_boundary": epoch}
            if args.front10_ref_online and args.front10_ref_ema:
                ref_o = json.loads(Path(args.front10_ref_online).read_text(encoding="utf-8"))
                ref_e = json.loads(Path(args.front10_ref_ema).read_text(encoding="utf-8"))
                c_o = compare_histories(history_online, ref_o, epoch)
                c_e = compare_histories(history_ema, ref_e, epoch)
                c_h = None
                if args.front10_ref_runinfo:
                    ri = json.loads(Path(args.front10_ref_runinfo).read_text(encoding="utf-8"))
                    ref_tr = {t["epoch"]: t["roworder_sha256"] for t in ri.get("roworder_trace", [])}
                    cur_tr = {t["epoch"]: t["roworder_sha256"] for t in rng_trace}
                    bad = [e for e in range(1, epoch + 1) if cur_tr.get(e) != ref_tr.get(e)]
                    c_h = {"pass": not bad, "mismatched_epochs": bad,
                           "n_compared": len([e for e in range(1, epoch + 1) if e in ref_tr])}
                front10.update({"online_vs_ref": c_o, "ema_vs_ref": c_e, "roworder_vs_ref": c_h})
                front10["pass"] = bool(c_o["pass"] and c_e["pass"]
                                       and (c_h is None or c_h["pass"]))
                (out_dir / "front10_gate.json").write_text(
                    json.dumps(front10, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps({"stage": "front10_gate", **{
                    "pass": front10["pass"],
                    "online_max_abs_diff": c_o["max_abs_diff"],
                    "online_first_divergence": c_o["first_divergence"],
                    "ema_max_abs_diff": c_e["max_abs_diff"],
                    "ema_first_divergence": c_e["first_divergence"],
                    "roworder": c_h}}, ensure_ascii=False), flush=True)
                if not front10["pass"]:
                    raise SystemExit("FATAL: front-10 regression gate FAILED -> stop before epoch 11")

    save_ckpt(out_dir / "online_last_model.pt", model, args, history_online[-1], dcfg, wcfg)
    save_ckpt(out_dir / "ema_last_model.pt", ema.shadow, args, history_ema[-1], dcfg, wcfg)
    save_resume(out_dir / "resume_state.pt", args, model, ema, optimizer,
                global_step, args.epochs, lr_trace)
    (out_dir / "online_history.json").write_text(json.dumps(history_online, indent=2), encoding="utf-8")
    (out_dir / "ema_history.json").write_text(json.dumps(history_ema, indent=2), encoding="utf-8")
    (out_dir / "ema_run_info.json").write_text(json.dumps({
        "beta": BETA, "ema_base": EMA_BASE, "n_epochs": args.epochs, "seed": args.seed,
        "lr_drop_epoch": args.lr_drop_epoch, "lr_after_drop": args.lr_after_drop,
        "lr_initial": float(args.lr),
        "global_step": global_step, "ema_update_count": ema.n_updates,
        "n_ema_params": info["n_ema_params"], "n_frozen_params_copied": info["n_frozen_params_copied"],
        "roworder_trace": rng_trace, "lr_trace": lr_trace, "front10_gate": front10,
        "args": vars(args)}, indent=2), encoding="utf-8")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
