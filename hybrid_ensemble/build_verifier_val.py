#!/usr/bin/env python3
"""Build a candidate-level verifier dataset on the *validation* split.

Motivation (measured 2026-09-12): the shipped union sits far inside the real
compl/purity frontier.  With the SAME 1470-slot budget an oracle ranking of the
existing permissive pool would reach compl 0.763 / purity 0.854 versus the
shipped 0.581 / 0.650.  Nothing cheap recovers that gap: heatmap value,
heatmap x count-prob and member votes all land within 1 pp of each other, and
pool-level AUC is inflated by easy candidates.  The missing ingredient is a
discriminator trained *for the candidate-real-or-not task*, which is what this
dataset feeds.

Pool: for each of the 4 ensemble members, take every local heatmap maximum with
p > 0.20 on the val split with the count gate opened (count_bias [0,3,3]) --
the same "permissive pool" convention as frontier_compl_purity.py.

Per candidate we store
  window   (5, 41) : local raw-flux window, including an UNCLIPPED normalisation
                     so the verifier can see the deep troughs that
                     normalize_with_scale(clip 0..3) destroyed (~15% of DLA cores)
  scalars  (S,)    : per-member heatmap / logNHI / offset at the candidate
                     pixel + 5-px heatmap profile, count probabilities,
                     vote count, SNR, z_QSO, n_peaks, local sigma, ...
  label            : 1 if greedy_match(dv_limit=600 km/s) pairs it with truth
  dv               : signed velocity error for positives (regression target)

Rows are split by parity into two disjoint halves so the verifier can be
trained and evaluated without touching the same spectra:
  grp 0 -> verifier train      grp 1 -> verifier eval

Caveat to state in any report: towers never *trained* on val, but their
best_model.pt was SELECTED by val score, so both halves inherit that selection
bias.  It is the standard stacking compromise, not a fresh holdout.

Run:
  CUDA_VISIBLE_DEVICES=0 /home/dingjch/anaconda3/envs/ML_env/bin/python \
      -u build_verifier_val.py --out /tmp/verifier/val_pool.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
SRC = os.path.expanduser("~/csst-dla-finder/src")
for p in (HB, SRC, os.path.expanduser("~")):
    if p not in sys.path:
        sys.path.insert(0, p)

from csst_dla.fits_utils import read_image, read_labels          # noqa: E402
from csst_dla.scoring import C_KMS, greedy_match, labels_to_truth  # noqa: E402
from data import HybridTrainDataset, local_noise_sigma, smooth_flux  # noqa: E402
from decode import pick_peaks, pixel_to_wavelength, predict_member, softmax  # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device       # noqa: E402
from feature_fusion import DualFusionTrainDataset, DualTowerFusionNet  # noqa: E402
from csst_dla_wzx_pkg.inference import load_model_from_checkpoint  # noqa: E402

RUNS = os.path.expanduser("~/csst_dla_runs")
TARGETS = os.path.join(RUNS, "20260903/common/cnn_targets_seed42.npz")
TRAIN_FITS = "/data/aoqige/train_5e5.fits"
LYA = 1215.67

TRANSFORMERS = [
    ("v3c", os.path.join(RUNS, "20260911/transformer_v3c_lognhi/best_model.pt")),
    ("v5", os.path.join(RUNS, "20260911/transformer_conv_stem_v5_rope/best_model.pt")),
    ("v6", os.path.join(RUNS, "20260912/transformer_conv_stem_v6_alibi/best_model.pt")),
]
SOTA_CKPT = os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/best_model.pt")

# permissive pool convention (matches frontier_compl_purity.py)
POOL_THRESHOLD = 0.20
POOL_COUNT_BIAS = np.asarray([0.0, 3.0, 3.0], dtype=np.float32)
MIN_DISTANCE = 10
MIN_Z_DLA = 1.10
LOGN_LOW, LOGN_HIGH = 19.5, 22.5
HALF = 20            # window half-width in pixels -> 41 px
SIG_CLIP = 12.0


def log(msg: str, **kw) -> None:
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "msg": msg, **kw}), flush=True)


def load_fusion(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    dilated, _ = load_checkpoint(cfg["dilated_checkpoint"], device)
    wzx, _ = load_model_from_checkpoint(cfg["wzx_checkpoint"], device)
    model = DualTowerFusionNet(
        dilated.model, wzx,
        merge_mode=cfg["merge_mode"],
        width=int(cfg["fusion_width"]),
        depth=int(cfg["fusion_depth"]),
        freeze_backbones=bool(cfg.get("freeze_backbones", True)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.inference_mode()
def predict_fusion(model, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    heat, logn, off, cnt, rows = [], [], [], [], []
    for batch in loader:
        hybrid, wzx, zq = batch[0], batch[1], batch[2]
        out = model(hybrid.to(device), wzx.to(device), zq.to(device))
        heat.append(torch.sigmoid(out["center_logits"]).cpu().numpy())
        logn.append((20.3 + out["lognhi_raw"]).cpu().numpy())
        off.append(out["offset_raw"].cpu().numpy())
        cnt.append(out["count_logits"].cpu().numpy())
        rows.append(np.asarray(batch[10], dtype=np.int64))
    return {
        "heatmap": np.concatenate(heat),
        "lognhi": np.concatenate(logn),
        "offset": np.concatenate(off),
        "count_logits": np.concatenate(cnt),
        "rows": np.concatenate(rows),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/verifier/val_pool.npz")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ---- shared val metadata (also gives us the cached 'flux' channels) ----
    log("building val dataset (input_mode=flux)")
    val_ds = HybridTrainDataset(TARGETS, TRAIN_FITS, "val", "flux", cache_channels=True)
    n_val = len(val_ds)
    val_idx = val_ds.indices.astype(np.int64)
    wave = val_ds.wavelength.astype(np.float32)
    zq = val_ds.zq.astype(np.float32)
    snr = val_ds.snr.astype(np.float32)
    labels = val_ds.labels
    labels_slim = {k: labels[k][val_idx] for k in
                   ("Z_QSO", "N_DLA", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2", "SNR_GU")}
    truth = labels_to_truth(labels, val_idx, min_lognhi=20.3)
    log("val ready", n_val=int(n_val), n_truth=int(len(truth["TARGETID"])))
    del labels

    # ---- member predictions on val (cached: each member is a ~45 s forward pass) ----
    member_names = [n for n, _ in TRANSFORMERS] + ["sota"]
    cache_dir = os.path.join(os.path.dirname(args.out), "member_preds")
    os.makedirs(cache_dir, exist_ok=True)
    preds: dict[str, dict[str, np.ndarray]] = {}

    def cached(name, builder):
        path = os.path.join(cache_dir, f"{name}.npz")
        if os.path.exists(path):
            with np.load(path) as f:
                d = {k: f[k] for k in f.files}
            log("member cached", member=name)
            return d
        d = builder()
        np.savez(path, **d)
        return d

    for name, ckpt_path in TRANSFORMERS:
        t0 = time.time()
        def build(ckpt_path=ckpt_path):
            model, cfg = load_checkpoint(ckpt_path, device)
            loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            out = predict_member(model, loader, device)
            del model
            torch.cuda.empty_cache()
            return out
        preds[name] = cached(name, build)
        log("member ready", member=name, secs=round(time.time() - t0, 1))

    t0 = time.time()
    def build_sota():
        fusion, fpaste = load_fusion(SOTA_CKPT, device)
        fcfg = fpaste["config"]
        fds = DualFusionTrainDataset(
            TARGETS, TRAIN_FITS, "val",
            dilated_input_mode=fcfg.get("dilated_input_mode", "flux"),
            wzx_feature_mode=fcfg.get("wzx_feature_mode", "flux"),
        )
        assert len(fds) == n_val
        out = predict_fusion(fusion, fds, device, args.batch_size)
        del fusion, fds
        torch.cuda.empty_cache()
        return out
    preds["sota"] = cached("sota", build_sota)
    log("member ready", member="sota", secs=round(time.time() - t0, 1))

    # NOTE: HybridTrainDataset returns the GLOBAL fits row index (val_idx), not
    # the position inside the split, and decode_validation_catalog indexes
    # predictions positionally.  shuffle=False guarantees position r <-> val row r.
    for name in member_names:
        rows = preds[name]["rows"]
        assert rows.shape == val_idx.shape, f"{name}: row shape {rows.shape}"
        assert np.array_equal(rows, val_idx), f"{name}: row order mismatch"
        log("member rows ok", member=name)

    # ---- unclipped spectrum statistics over the whole val block ----
    log("loading raw FLUX and building unclipped view statistics")
    raw = read_image(TRAIN_FITS, "FLUX").astype(np.float32)[val_idx]     # (N,681)
    scale = np.percentile(raw, 75, axis=1, keepdims=True).astype(np.float32)
    scale[scale <= 0] = 1.0
    nf_unc = (raw / scale).astype(np.float32)                            # unclipped
    del raw
    nf_clip = val_ds.channels[:, 0, :].astype(np.float32)                # what towers saw
    sm_unc = np.empty_like(nf_unc)
    sig = np.empty_like(nf_unc)
    sig_local = np.empty_like(nf_unc)
    for i in range(n_val):
        s = smooth_flux(nf_unc[i], 15)
        sm_unc[i] = s
        r = nf_unc[i] - s
        sd = local_noise_sigma(r)
        sig_local[i] = sd
        sig[i] = np.clip(r / np.maximum(sd, 1e-6), -SIG_CLIP, SIG_CLIP)
    del val_ds.channels
    wave_norm = ((2.0 * (wave - wave.min()) / (wave.max() - wave.min())) - 1.0).astype(np.float32)
    blue = (wave[None, :] < (LYA * (1.0 + zq))[:, None]).astype(np.float32)
    log("views ready")

    # ---- permissive candidate pool, one pass per member ----
    counts = {k: [] for k in ("row", "pix", "cprob", "n_pick", "n_peaks", "rank", "member")}
    for mi, name in enumerate(member_names):
        p = preds[name]
        cp = softmax(p["count_logits"] + POOL_COUNT_BIAS[None, :])
        n_pick_all = np.argmax(cp, axis=1)
        cprob_all = cp[np.arange(n_val), n_pick_all]
        for row in range(n_val):
            peaks = pick_peaks(p["heatmap"][row], wave, float(zq[row]),
                               n_pick=int(n_pick_all[row]), threshold=POOL_THRESHOLD,
                               min_distance=MIN_DISTANCE, min_z_dla=MIN_Z_DLA)
            for k, pix in enumerate(peaks):
                counts["row"].append(row)
                counts["pix"].append(int(pix))
                counts["cprob"].append(float(cprob_all[row]))
                counts["n_pick"].append(int(n_pick_all[row]))
                counts["n_peaks"].append(len(peaks))
                counts["rank"].append(k)
                counts["member"].append(mi)
        log("pool", member=name, total=len(counts["row"]))

    row_c = np.asarray(counts["row"], dtype=np.int32)
    pix_c = np.asarray(counts["pix"], dtype=np.int32)
    member_c = np.asarray(counts["member"], dtype=np.int8)
    n_cand = len(row_c)

    # ---- candidate coordinates (offset-corrected, member-averaged) ----
    z_cand = np.zeros(n_cand, dtype=np.float32)
    logn_ens = np.zeros(n_cand, dtype=np.float32)
    hm_ens = np.zeros(n_cand, dtype=np.float32)
    for name in member_names:
        p = preds[name]
        pix_f = pix_c.astype(np.float64)
        off = p["offset"][row_c, pix_c]
        pix_shift = np.clip(pix_f + off, 0, len(wave) - 1)
        lam = np.interp(pix_shift, np.arange(len(wave), dtype=np.float64), wave.astype(np.float64))
        z_cand += (lam / LYA - 1.0).astype(np.float32)
        logn_ens += p["lognhi"][row_c, pix_c]
        hm_ens += p["heatmap"][row_c, pix_c]
        del lam, off
    k_mem = len(member_names)
    z_cand /= k_mem
    logn_ens = np.clip(logn_ens / k_mem, LOGN_LOW, LOGN_HIGH)
    hm_ens /= k_mem

    # ---- labels ----
    cand_cat = {
        "TARGETID": val_idx[row_c],
        "Z_DLA": z_cand,
        "LOG_NHI": logn_ens,
        "SNR": snr[row_c],
    }
    matches = greedy_match(truth, cand_cat)
    truth_of_cand = np.full(n_cand, -1, dtype=np.int32)
    dv_of_cand = np.full(n_cand, np.nan, dtype=np.float32)
    for ti, pi, dv in matches:
        truth_of_cand[pi] = ti
        dv_of_cand[pi] = C_KMS * (cand_cat["Z_DLA"][pi] - truth["Z_DLA"][ti]) / (1.0 + truth["Z_DLA"][ti])
    y = (truth_of_cand >= 0).astype(np.uint8)

    # Nearest-truth signed velocity for EVERY candidate, not just the matched ones.
    # A candidate 100 km/s from a truth that greedy_match already gave to a closer
    # candidate is not a false positive -- it is the same DLA.  Training on it as a
    # negative teaches the net to demote good candidates, so we flag it and mask it.
    z_by_tid: dict[int, list[float]] = {}
    for t, zz in zip(truth["TARGETID"].tolist(), truth["Z_DLA"].tolist()):
        z_by_tid.setdefault(int(t), []).append(float(zz))
    dv_near = np.full(n_cand, np.nan, dtype=np.float32)
    for i in range(n_cand):
        zs = z_by_tid.get(int(cand_cat["TARGETID"][i]))
        if not zs:
            continue
        zc = float(z_cand[i])
        best = min(zs, key=lambda zt: abs(zc - zt))
        dv_near[i] = C_KMS * (zc - best) / (1.0 + best)
    dup = ((y == 0) & (np.abs(dv_near) < 600.0)).astype(np.uint8)
    log("labels", n_cand=int(n_cand), n_pos=int(y.sum()),
        n_dup=int(dup.sum()), purity=round(float(y.mean()), 4))

    # ---- windows ----
    log("extracting windows")
    offs = np.arange(-HALF, HALF + 1)
    pix_w = np.clip(pix_c[:, None] + offs[None, :], 0, len(wave) - 1)   # (N, W)
    rows_w = np.repeat(row_c[:, None], len(offs), axis=1)
    window = np.empty((n_cand, 5, len(offs)), dtype=np.float32)
    window[:, 0, :] = nf_clip[rows_w, pix_w]
    window[:, 1, :] = nf_unc[rows_w, pix_w]
    window[:, 2, :] = sig[rows_w, pix_w]
    window[:, 3, :] = wave_norm[pix_w]
    window[:, 4, :] = blue[rows_w, pix_w]
    window -= np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)[None, :, None]
    log("windows done")

    # ---- scalars ----
    prof_r = np.clip(pix_c[:, None] + np.arange(-2, 3)[None, :], 0, len(wave) - 1)  # (N,5)
    cols, names = [], []

    def add(name, value):
        cols.append(np.asarray(value, dtype=np.float32))
        names.append(name)

    for name in member_names:
        p = preds[name]
        add(f"hm_{name}", p["heatmap"][row_c, pix_c])
        add(f"logn_{name}", p["lognhi"][row_c, pix_c])
        add(f"off_{name}", p["offset"][row_c, pix_c])
        hmprof = p["heatmap"][row_c[:, None], prof_r]
        for j in range(5):
            add(f"hmprof{name}_{j-2}", hmprof[:, j])
        # height of this candidate relative to the member's own strongest peak in
        # the spectrum -- makes "this member's #1 hit" distinguishable from its #7
        add(f"hmrel_{name}", p["heatmap"][row_c, pix_c] /
            np.maximum(p["heatmap"][row_c].max(axis=1), 1e-6))
    for j, name in enumerate(member_names):
        add(f"is_{name}", (member_c == j).astype(np.float32))
    cp_ens = np.mean([softmax(preds[n]["count_logits"] + POOL_COUNT_BIAS[None, :]) for n in member_names],
                     axis=0)
    for j in range(3):
        add(f"cprob_ens{j}", cp_ens[row_c, j])
    add("hm_ens", hm_ens)
    add("hmrel_ens", hm_ens / np.maximum(
        np.mean([preds[n]["heatmap"][row_c].max(axis=1) for n in member_names], axis=0), 1e-6))
    add("logn_ens", logn_ens)
    add("hm_std", np.std([preds[n]["heatmap"][row_c, pix_c] for n in member_names], axis=0))
    # how many members put a peak at (almost) the same pixel
    vote = np.zeros(n_cand, dtype=np.float32)
    for name in member_names:
        p = preds[name]
        prof = p["heatmap"][row_c[:, None], prof_r]       # (N,5)
        vote += (prof.max(axis=1) >= POOL_THRESHOLD).astype(np.float32)
    add("vote", vote)
    add("snr", snr[row_c])
    add("zq", zq[row_c])
    add("z_dla", z_cand)
    add("z_minus_zq", z_cand - zq[row_c])
    add("pix", pix_c)
    add("n_peaks", np.asarray(counts["n_peaks"], dtype=np.float32))
    add("peak_rank", np.asarray(counts["rank"], dtype=np.float32))
    add("cprob_at", np.asarray(counts["cprob"], dtype=np.float32))
    add("n_pick", np.asarray(counts["n_pick"], dtype=np.float32))
    add("sig_local", sig_local[row_c, pix_c])
    add("sig_min_win", np.min(sig[rows_w, pix_w], axis=1))
    add("nf_min_win", np.min(window[:, 1, :], axis=1))
    add("nf_med_win", np.median(window[:, 1, :], axis=1))
    # distance to the nearest other candidate in the same spectrum (px)
    order = np.lexsort((pix_c, row_c))
    gap = np.full(n_cand, 999.0, dtype=np.float32)
    prev_row, prev_pix = -1, -10 ** 9
    last_row, last_pix = -1, -10 ** 9
    for i in order:                      # forward pass over sorted rows
        if row_c[i] == prev_row:
            gap[i] = min(gap[i], pix_c[i] - prev_pix)
        prev_row, prev_pix = row_c[i], pix_c[i]
    for i in order[::-1]:
        if row_c[i] == last_row:
            gap[i] = min(gap[i], last_pix - pix_c[i])
        last_row, last_pix = row_c[i], pix_c[i]
    add("gap_px", gap)

    scalars = np.stack(cols, axis=1).astype(np.float32)
    # three-way split by spectrum parity so the verifier can be trained, early
    # stopped and finally scored on three disjoint sets of spectra
    grp = (row_c % 3).astype(np.uint8)

    log("saving", window=window.shape, scalars=scalars.shape, ncols=len(names))
    np.savez_compressed(
        args.out,
        window=window, scalars=scalars, scalar_names=np.asarray(names),
        y=y, dv=dv_of_cand, dv_near=dv_near, dup=dup, grp=grp,
        row=row_c, pix=pix_c, member=member_c,
        z=z_cand, logn=logn_ens, hm=hm_ens,
        targetid=val_idx[row_c], truth_idx=truth_of_cand,
        val_idx=val_idx,
        wave=wave,
        meta=np.asarray(json.dumps({
            "n_val": int(n_val), "n_cand": int(n_cand), "n_pos": int(y.sum()),
            "members": member_names, "pool_threshold": POOL_THRESHOLD,
            "pool_count_bias": POOL_COUNT_BIAS.tolist(),
            "n_truth": int(len(truth["TARGETID"])),
        })),
        truth_targetid=truth["TARGETID"], truth_z=truth["Z_DLA"], truth_lognhi=truth["LOG_NHI"],
        truth_snr=truth["SNR"],
        labels_Z_QSO=labels_slim["Z_QSO"], labels_N_DLA=labels_slim["N_DLA"],
        labels_Z_DLA1=labels_slim["Z_DLA1"], labels_LOGNHI1=labels_slim["LOGNHI1"],
        labels_Z_DLA2=labels_slim["Z_DLA2"], labels_LOGNHI2=labels_slim["LOGNHI2"],
        labels_SNR_GU=labels_slim["SNR_GU"],
    )
    log("DONE", out=args.out)


if __name__ == "__main__":
    main()
