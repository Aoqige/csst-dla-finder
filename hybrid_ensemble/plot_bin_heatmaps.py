"""SNR x logNHI purity / completeness heatmaps (paper-style two-panel figure).

Reproduces the layout of the reference figure:
  left  panel : Purity       = matched predictions / predicted        (binned by the PRED's own SNR, logNHI)
  right panel : Completeness = matched truth        / true DLAs       (binned by the TRUTH's own SNR, logNHI)
  x axis      : SNR_GU, 7 bins  [0,0.3) [0.3,0.6) [0.6,1) [1,2) [2,3) [3,4) [4,5)
  y axis      : log(N_HI), 5 bins [20.3,20.5) [20.5,21) [21,21.5) [21.5,22) [22,inf)
  gray cell   : empty bin (no truth for completeness, no prediction for purity)

Matching = src/csst_dla/scoring.py greedy_match (dv_limit 600 km/s), the official scorer.

Usage (on dataCop-119):
  P=/home/dingjch/anaconda3/envs/ML_env/bin/python
  $P plot_bin_heatmaps.py --models tf_sota,cnn_sota,fuse_sota \
      --outdir ~/csst_dla_runs/20260912/binfigs --combined
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

import score_test  # noqa: E402
from csst_dla.scoring import greedy_match  # noqa: E402

RUNS = os.path.expanduser("~/csst_dla_runs")
TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
SNR_FIELD = "SNR_GU"
MIN_LOGNHI = 20.3
DV_LIMIT = 600.0

SNR_EDGES = [0.0, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0]
SNR_LABELS = ["0-0.3", "0.3-0.6", "0.6-1", "1-2", "2-3", "3-4", "4-5"]
NHI_EDGES = [20.3, 20.5, 21.0, 21.5, 22.0, np.inf]
NHI_LABELS = ["[20.3,20.5)", "[20.5,21)", "[21,21.5)", "[21.5,22)", "[22,inf)"]

# Three-way SOTA taxonomy used throughout the report:
#   TF SOTA    = best SINGLE transformer tower        (v3c_lognhi + count_bias)
#   CNN SOTA   = best CNN-family system               (2 frozen CNN towers + fusion head)
#   Fusion SOTA= cross-architecture decision union    (3 TF towers + CNN SOTA member)
# key -> (short title, predictions.csv, "Final / Det / Param" for the caption)
MODELS = {
    "tf_sota": (
        "TF SOTA: v3c_lognhi + count_bias",
        os.path.join(RUNS, "20260912/transformer_v3c_countbias/predictions_countbias_075.csv"),
        "Final 0.5310  Det 0.5506  Param 0.5016  (single tower, decode-only count_bias)",
    ),
    "tf_raw": (
        "TF single tower: v3c_lognhi raw decode",
        os.path.join(RUNS, "20260911/transformer_v3c_lognhi/predictions.csv"),
        "Final 0.4980  Det 0.4835  Param 0.5197  (no count_bias)",
    ),
    "cnn_sota": (
        "CNN SOTA: feature_no_clean_flux_context_v1",
        os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv"),
        "Final 0.4955  Det 0.4815  Param 0.5166  (2 frozen CNN towers + fusion head)",
    ),
    "fuse_sota": (
        "Fusion SOTA: ensemble_4way_v3",
        os.path.join(RUNS, "20260912/ensemble_4way_v3/predictions.csv"),
        "Final 0.5455  Det 0.5721  Param 0.5056  (3 TF towers + CNN SOTA, union)",
    ),
    "cnn_base": (
        "CNN baseline: dilated_flux",
        os.path.join(RUNS, "20260903/dilated_flux/predictions.csv"),
        "Final 0.3625  Det 0.3259  Param 0.4174",
    ),
}


def load_prediction(csv_path: str, snr_by_target: dict[int, float]) -> dict[str, np.ndarray]:
    cat = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if cat.ndim == 0:
        cat = np.asarray([cat])
    cols = {k: [] for k in ("tid", "z", "lognhi", "snr")}
    for row in cat:
        tid = int(row["id"])
        for slot in (1, 2):
            z = float(row[f"Z_DLA{slot}"])
            ln = float(row[f"LOGNHI{slot}"])
            if z <= 0 or ln < MIN_LOGNHI:
                continue
            if not score_test._in_band(z, score_test.BAND_BY_SNR_FIELD[SNR_FIELD]):
                continue
            cols["tid"].append(tid)
            cols["z"].append(z)
            cols["lognhi"].append(ln)
            cols["snr"].append(snr_by_target.get(tid, np.nan))
    return {
        "TARGETID": np.asarray(cols["tid"], np.int64),
        "Z_DLA": np.asarray(cols["z"], np.float32),
        "LOG_NHI": np.asarray(cols["lognhi"], np.float32),
        "SNR": np.asarray(cols["snr"], np.float32),
    }


def bin_index(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Return bin index, -1 when out of range."""
    idx = np.digitize(values, edges[1:-1], right=False)
    idx = np.where(np.isnan(values), -1, idx)
    return idx.astype(int)


def compute_grids(truth: dict, pred: dict):
    matches = greedy_match(truth, pred, dv_limit=DV_LIMIT)
    ti = np.asarray([m[0] for m in matches], int)
    pi = np.asarray([m[1] for m in matches], int)

    t_nh = bin_index(truth["LOG_NHI"].astype(float), NHI_EDGES)
    t_sn = bin_index(truth["SNR"].astype(float), SNR_EDGES)
    p_nh = bin_index(pred["LOG_NHI"].astype(float), NHI_EDGES)
    p_sn = bin_index(pred["SNR"].astype(float), SNR_EDGES)

    shape = (len(NHI_LABELS), len(SNR_LABELS))
    n_truth = np.zeros(shape)
    n_pred = np.zeros(shape)
    m_truth = np.zeros(shape)
    m_pred = np.zeros(shape)

    for h, s in zip(t_nh, t_sn):
        if h >= 0 and s >= 0:
            n_truth[h, s] += 1
    for h, s in zip(p_nh, p_sn):
        if h >= 0 and s >= 0:
            n_pred[h, s] += 1
    for i in ti:
        h, s = t_nh[i], t_sn[i]
        if h >= 0 and s >= 0:
            m_truth[h, s] += 1
    for i in pi:
        h, s = p_nh[i], p_sn[i]
        if h >= 0 and s >= 0:
            m_pred[h, s] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        compl = np.where(n_truth > 0, m_truth / np.maximum(n_truth, 1), np.nan)
        purity = np.where(n_pred > 0, m_pred / np.maximum(n_pred, 1), np.nan)

    return dict(n_truth=n_truth, n_pred=n_pred, m_truth=m_truth, m_pred=m_pred,
                compl=compl, purity=purity, n_match=len(matches))


def draw_panel(ax, grid, title, ylabel, yticklabels, vmax=1.0):
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    masked = np.ma.masked_invalid(np.flipud(grid))
    im = ax.imshow(masked, cmap=cmap, norm=Normalize(0.0, vmax), aspect="auto")

    ax.set_xticks(np.arange(len(SNR_LABELS)))
    ax.set_xticklabels(SNR_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(NHI_LABELS)))
    ax.set_yticklabels(list(reversed(yticklabels)), fontsize=9)
    ax.set_xlabel("SNR", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    ax.set_xticks(np.arange(-0.5, len(SNR_LABELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(NHI_LABELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    flipped = np.flipud(grid)
    for r in range(flipped.shape[0]):
        for c in range(flipped.shape[1]):
            v = flipped[r, c]
            if np.isnan(v):
                continue
            ax.text(c, r, "%.3f" % v, ha="center", va="center", fontsize=8,
                    color="black" if v > 0.55 else "white", fontweight="bold")

    cb = ax.figure.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    cb.set_label(title.split(" (")[0], fontsize=9)
    cb.ax.tick_params(labelsize=8)
    return im


def make_figure(model_key: str, outdir: str, dpi: int = 200):
    truth = score_test.load_truth(TRUTH, min_lognhi=MIN_LOGNHI, snr_field=SNR_FIELD)
    snr_by_target = score_test.load_snr_by_target(TRUTH, SNR_FIELD)
    title, csv_path, caption = MODELS[model_key]
    pred = load_prediction(csv_path, snr_by_target)
    g = compute_grids(truth, pred)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=dpi)
    draw_panel(axes[0], g["purity"], "Purity (raw FLUX)", "predicted log(N$_{HI}$)", NHI_LABELS)
    draw_panel(axes[1], g["compl"], "Completeness (raw FLUX)", "true log(N$_{HI}$)", NHI_LABELS)
    fig.suptitle("%s\n%s   |   n_truth %d  n_pred %d  n_match %d   (gray = empty bin)"
                 % (title, caption, len(truth["TARGETID"]), len(pred["TARGETID"]), g["n_match"]),
                 fontsize=10.5, y=0.995)
    fig.text(0.012, 0.032,
             "-- : empty bin (no prediction for Purity / no true DLA for Completeness). "
             "Purity is binned by the prediction's own SNR and log(N_HI); "
             "Completeness by the truth's. Matching = greedy 1-to-1 within 600 km/s (official scorer).",
             fontsize=7.2, color="#555555")
    fig.text(0.012, 0.012,
             "Taxonomy: TF SOTA = best single transformer tower | CNN SOTA = best CNN-family system "
             "(2 frozen CNN towers + fusion head) | Fusion SOTA = cross-architecture union of 3 TF towers + CNN SOTA.  "
             "Caveat: only the TF towers carry a count_bias decode; the CNN and Fusion members do not.",
             fontsize=7.2, color="#993C1D")
    fig.subplots_adjust(left=0.070, right=0.945, top=0.815, bottom=0.205, wspace=0.50)
    out = os.path.join(outdir, "bins_%s.png" % model_key)
    fig.savefig(out)
    plt.close(fig)

    print("=== %s ===" % model_key)
    print("n_truth=%d n_pred=%d n_match=%d" % (len(truth["TARGETID"]), len(pred["TARGETID"]), g["n_match"]))
    hdr = "  %-14s" % "logNHI \\ SNR"
    for lab in SNR_LABELS:
        hdr += "%14s" % lab
    for name, grid in (("PURITY", g["purity"]), ("COMPLETENESS", g["compl"])):
        print("  -- %s --" % name)
        print(hdr)
        for r in range(len(NHI_LABELS) - 1, -1, -1):
            line = "  %-14s" % NHI_LABELS[r]
            for c in range(len(SNR_LABELS)):
                v = grid[r, c]
                line += "%14s" % ("--" if np.isnan(v) else "%.3f" % v)
            print(line)
    print("  saved -> %s" % out)
    return out, g


def make_combined(keys, outdir, dpi=200):
    truth = score_test.load_truth(TRUTH, min_lognhi=MIN_LOGNHI, snr_field=SNR_FIELD)
    snr_by_target = score_test.load_snr_by_target(TRUTH, SNR_FIELD)
    n = len(keys)
    fig, axes = plt.subplots(n, 2, figsize=(11.0, 4.8 * n), dpi=dpi, squeeze=False)
    for i, key in enumerate(keys):
        title, csv_path, caption = MODELS[key]
        pred = load_prediction(csv_path, snr_by_target)
        g = compute_grids(truth, pred)
        draw_panel(axes[i][0], g["purity"], "Purity (raw FLUX)", "predicted log(N$_{HI}$)", NHI_LABELS)
        draw_panel(axes[i][1], g["compl"], "Completeness (raw FLUX)", "true log(N$_{HI}$)", NHI_LABELS)
        axes[i][0].set_title("Purity  -  %s\n%s   |   n_pred %d"
                             % (title, caption, len(pred["TARGETID"])),
                             fontsize=10.0, fontweight="bold", pad=8)
        axes[i][1].set_title("Completeness  -  %s\n%s   |   n_match %d"
                             % (title, caption, g["n_match"]),
                             fontsize=10.0, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.080, right=0.945, top=0.925, bottom=0.065,
                        wspace=0.52, hspace=0.55)
    out = os.path.join(outdir, "bins_combined_%s.png" % "_".join(keys))
    fig.savefig(out)
    plt.close(fig)
    print("combined -> %s" % out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tf_sota,cnn_sota,fuse_sota")
    ap.add_argument("--outdir", default=os.path.join(RUNS, "20260912/binfigs"))
    ap.add_argument("--combined", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    for k in keys:
        if k not in MODELS:
            raise SystemExit("unknown model key %r; choices=%s" % (k, list(MODELS)))
        make_figure(k, args.outdir)
    if args.combined and len(keys) > 1:
        make_combined(keys, args.outdir)


if __name__ == "__main__":
    main()
