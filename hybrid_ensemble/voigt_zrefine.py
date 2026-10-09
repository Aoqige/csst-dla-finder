#!/usr/bin/env python3
"""Damping-wing (Voigt) template fit -> sub-pixel z refinement upper bound.

Stage A  Synthetic injection: inject a Voigt DLA at a *sub-pixel* position into
         real DLA-free spectra, then refit starting from the pixel centre.
         Measures how much sub-pixel information survives at 8 A/px.

Stage B  Real predictions: refit the shipped union predictions and report
         std_dv / mean_dv against truth, using the same greedy_match(600 km/s)
         pairing as the scorer.  Baseline (unrefined) numbers are reported
         side by side.

Compliance: raw FLUX only, never FLUX_CLEAN.  The fit solves a per-grid-point
linear continuum (amplitude + slope) on UNCLIPPED flux, so the ~15% of the
core destroyed by normalize_with_scale(clip 0..3) does not bias it.

Physics: for a damped system the core is fully saturated (tau ~ 1e6) and the
observable signature is the Lorentzian damping wing,

    tau(dlam_rest) = C_TAU * N / dlam_rest^2 ,
    C_TAU = (e^2/m_e c) f Gamma lam0^4 / (4 pi c^2)

which is independent of the Doppler parameter b (b cancels between the Voigt
normalisation and the wing expansion).  b only sets the softening scale where
the saturated core is clipped, so it is fixed at 25 km/s.

Run (server):
  /home/dingjch/anaconda3/envs/ML_env/bin/python -u voigt_zrefine.py --stage A
  /home/dingjch/anaconda3/envs/ML_env/bin/python -u voigt_zrefine.py --stage B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from astropy.io import fits

# ---------------------------------------------------------------- constants --
C_CMS = 2.99792458e10          # cm/s
C_KMS = 2.99792458e5           # km/s
R_E = 2.8179403262e-13         # cm, classical electron radius
LAM_LYA = 1215.67              # A
F_LYA = 0.4164                 # oscillator strength
GAMMA_LYA = 6.265e8            # s^-1
A_CM = 1e-8                    # cm per angstrom

# tau = C_TAU * N / dlam_rest_cm^2   (N in cm^-2)
C_TAU = (R_E * C_CMS) * F_LYA * GAMMA_LYA * (LAM_LYA * A_CM) ** 4 / (
    4.0 * np.pi * C_CMS**2
)

B_DOPPLER = 25.0               # km/s, only sets the saturated-core softening
EPS_REST = (B_DOPPLER / C_KMS) * LAM_LYA          # ~0.101 A

TEST_FITS = "/data/aoqige/test.fits"
TRUTH_FITS = "/data/aoqige/test_truth.fits"
DEFAULT_PRED = os.path.expanduser(
    "~/csst_dla_runs/20260912/ensemble_4way_v3/predictions.csv"
)

HALF_WIN = 25                  # full window: +-25 px (continuum lives here)
FIT_HALF = 8                   # absorption is fitted on the central +-8 px only
SUB = 16                       # sub-samples per pixel for pixel integration
EDGE_PX = 8                    # px at each end used for the continuum estimate
DZ_STEP = 0.05                 # px
DZ_MAX = 2.0                   # px
LOGN_GRID = np.arange(20.0, 22.41, 0.1)


# ------------------------------------------------------------------ fitting --
def _continuum(flux_win, wave_win):
    """Linear continuum fitted on the two ENDS of the wide window only.

    Kept fixed during the fit: a free continuum lets the model absorb the
    trough and the (dz, logN) grid then runs away.  The ends sit 17-25 px from
    the centre, well outside even a strong damping wing (~3 px).
    """
    left = np.median(flux_win[:EDGE_PX])
    right = np.median(flux_win[-EDGE_PX:])
    xl = wave_win[:EDGE_PX].mean()
    xr = wave_win[-EDGE_PX:].mean()
    return left + (right - left) * (wave_win - xl) / (xr - xl)


def fit_one(flux_win, wave_win, lam_center_guess, cont_win=None):
    """Return (lam_center_fit, lognhi_fit, sse_fit, hit_edge), or None when the
    continuum is degenerate (non-positive) - that happens in the Ly-a forest
    where the local flux is noise dominated.

    cont_win: optional externally supplied continuum over the *whole* window
    (used by the oracle-continuum diagnostic).
    """
    n_pix = len(wave_win)
    dpx = float(np.median(np.diff(wave_win)))
    ipc = n_pix // 2
    sl = slice(ipc - FIT_HALF, ipc + FIT_HALF + 1)
    cont = _continuum(flux_win, wave_win) if cont_win is None else np.asarray(cont_win)
    cont = cont[sl]
    if not np.all(cont > 0) or np.median(cont) <= 0:
        return None
    y = flux_win[sl] / cont
    wfit = wave_win[sl]
    n_fit = wfit.size

    lam_fine = wfit[0] + (np.arange(n_fit * SUB) + 0.5) * (dpx / SUB)

    dz = np.arange(-DZ_MAX, DZ_MAX + 1e-9, DZ_STEP)
    g_dz, g_n = np.meshgrid(dz, LOGN_GRID, indexing="ij")
    g_dz = g_dz.ravel()
    g_n = g_n.ravel()
    G = g_dz.size

    lam_c = lam_center_guess + g_dz * dpx                   # [G]
    z_c = lam_c / LAM_LYA - 1.0
    dlam_obs = lam_fine[None, :] - lam_c[:, None]           # [G, n_fine]
    dlam_rest_cm = (dlam_obs / (1.0 + z_c)[:, None]) * A_CM
    tau = C_TAU * (10.0 ** g_n)[:, None] / (
        dlam_rest_cm**2 + (EPS_REST * A_CM) ** 2
    )
    tpix = np.exp(-tau).reshape(G, n_fit, SUB).mean(axis=2)  # [G, n_fit]

    den = np.einsum("gi,gi->g", tpix, tpix)
    a = np.einsum("gi,i->g", tpix, y) / np.maximum(den, 1e-30)
    a = np.clip(a, 0.0, 3.0)
    resid = y[None, :] - a[:, None] * tpix
    sse = np.einsum("gi,gi->g", resid, resid)
    k = int(np.argmin(sse))
    hit_edge = bool(
        abs(g_dz[k]) >= DZ_MAX - 1e-9 or g_n[k] <= LOGN_GRID[0] + 1e-9
    )
    return lam_c[k], g_n[k], sse[k], hit_edge


# --------------------------------------------------------------- data utils --
def read_truth(path=TRUTH_FITS):
    with fits.open(path, memmap=True) as hdul:
        d = hdul["TRUTH"].data
        out = {
            "TARGETID": np.asarray(d["TARGETID"], dtype=np.int64),
            "N_DLA": np.asarray(d["N_DLA"], dtype=np.int32),
            "Z_QSO": np.asarray(d["Z_QSO"], dtype=np.float64),
            "Z_DLA1": np.asarray(d["Z_DLA1"], dtype=np.float64),
            "LOGNHI1": np.asarray(d["LOGNHI1"], dtype=np.float64),
            "Z_DLA2": np.asarray(d["Z_DLA2"], dtype=np.float64),
            "LOGNHI2": np.asarray(d["LOGNHI2"], dtype=np.float64),
            "SNR_GU": np.asarray(d["SNR_GU"], dtype=np.float64),
        }
    return out


def read_flux(path=TEST_FITS):
    with fits.open(path, memmap=True) as hdul:
        flux = np.asarray(hdul["FLUX"].data, dtype=np.float64)
        wave = np.asarray(hdul["WAVELENGTH"].data, dtype=np.float64)
    if wave.ndim == 1:
        wave = np.broadcast_to(wave, flux.shape).copy()
    return flux, wave


def dv_of(z_pred, z_true):
    return C_KMS * (z_pred - z_true) / (1.0 + z_true)


def greedy_match(z_true, z_pred, dv_limit=600.0):
    """Return (list of (ti, pi, dv)) greedy pairs, same rule as the scorer."""
    pairs = []
    used = set()
    order = np.argsort(np.abs(z_true[:, None] - z_pred[None, :]), axis=1)
    for ti in range(len(z_true)):
        for pi in order[ti]:
            if pi in used:
                continue
            dv = dv_of(z_pred[pi], z_true[ti])
            if abs(dv) <= dv_limit:
                pairs.append((ti, int(pi), dv))
                used.add(int(pi))
                break
    return pairs


# ----------------------------------------------------------------- stage A ---
def _smooth(y, w=31):
    k = np.ones(w) / w
    yp = np.pad(y, w // 2, mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def stage_a(flux, wave, truth, n_per_bin=400, seed=0, z_lo=1.15, z_hi=2.40,
            ideal=False, oracle_cont=False,
            sigmas=(0.0, 0.05, 0.10, 0.20, 0.35)):
    """Inject a Voigt DLA at a sub-pixel position; refit from the pixel centre.

    ideal=False -> inject into real DLA-free spectra (real noise + forest)
    ideal=True  -> flat continuum + white noise of a chosen sigma, so we can
                   separate "is the method right" from "is the data good enough"

    Noise convention in ideal mode: the continuum is 1 and the noise is ADDED
    AFTER the absorption is applied (f_inj = T + noise), i.e. constant sigma in
    observed flux.  The earlier multiplicative form (f_inj = (1+noise)*T) put
    almost no noise inside the saturated core and flattered the fit by 2-3x.
    """
    rng = np.random.default_rng(seed)
    clean_rows = np.flatnonzero(truth["N_DLA"] == 0)
    dpx = float(np.median(np.diff(wave[0])))
    lam_lo = LAM_LYA * (1.0 + z_lo)
    lam_hi = LAM_LYA * (1.0 + z_hi)

    results = []
    sig_list = list(sigmas) if ideal else [None]
    for sig in sig_list:
        for logn_inj in (20.3, 20.8, 21.3):
            rows = rng.choice(clean_rows, size=min(n_per_bin, len(clean_rows)),
                              replace=False)
            dvs, dvs_raw, lfit, n_edge = [], [], [], 0
            n_fail = 0
            for r in rows:
                w = wave[r]
                if ideal:
                    # continuum 1 with additive white noise of sigma `sig`;
                    # the noise is carried through the absorption unchanged
                    # (constant sigma in observed flux, as in the real data)
                    f_clean = np.ones_like(w)
                    f = f_clean + rng.normal(0.0, sig, size=w.shape)
                else:
                    f = flux[r]
                    f_clean = f
                cand = np.flatnonzero((w >= lam_lo) & (w <= lam_hi))
                cand = cand[(cand >= HALF_WIN + 1) & (cand <= len(w) - HALF_WIN - 2)]
                if cand.size == 0:
                    continue
                ip = int(rng.choice(cand))
                frac = rng.uniform(-0.5, 0.5)          # sub-pixel offset, px
                lam_true = w[ip] + frac * dpx
                z_true = lam_true / LAM_LYA - 1.0

                dlam_obs = w - lam_true
                dlam_rest_cm = (dlam_obs / (1.0 + z_true)) * A_CM
                tau = C_TAU * 10.0**logn_inj / (
                    dlam_rest_cm**2 + (EPS_REST * A_CM) ** 2
                )
                f_inj = f_clean * np.exp(-tau) + (f - f_clean)

                sl = slice(ip - HALF_WIN, ip + HALF_WIN + 1)
                cont_full = _smooth(f) if oracle_cont else None
                res = fit_one(
                    f_inj[sl], w[sl], w[ip],
                    cont_win=None if cont_full is None else cont_full[sl],
                )
                if res is None:
                    n_fail += 1
                    continue
                lam_fit, logn_fit, _, edge = res
                z_fit = lam_fit / LAM_LYA - 1.0
                dvs.append(dv_of(z_fit, z_true))
                dvs_raw.append(dv_of(w[ip] / LAM_LYA - 1.0, z_true))
                lfit.append(logn_fit)
                n_edge += int(edge)

            dvs = np.asarray(dvs)
            dvs_raw = np.asarray(dvs_raw)
            results.append({
                "mode": ("ideal" if ideal else ("oracle-cont" if oracle_cont else "real")),
                "sigma": sig,
                "lognhi_inj": logn_inj,
                "n": int(dvs.size),
                "edge_hit_rate": float(n_edge / max(dvs.size, 1)),
                "continuum_fail_rate": float(n_fail / max(dvs.size + n_fail, 1)),
                "median_lognhi_fit": float(np.median(lfit)),
                "std_dv_refined": float(np.std(dvs)),
                "mean_dv_refined": float(np.mean(dvs)),
                "std_dv_pixelcenter": float(np.std(dvs_raw)),
                "mad_dv_refined": float(
                    np.median(np.abs(dvs - np.median(dvs))) * 1.4826
                ),
            })
            mode = "ideal" if ideal else ("oracle" if oracle_cont else "real")
            print(f"  mode={mode} sig={sig} "
                  f"logNHI={logn_inj}: n={dvs.size} "
                  f"refined={np.std(dvs):7.1f}  pixel-centre={np.std(dvs_raw):7.1f} "
                  f"medlogN={np.median(lfit):.2f} edge={n_edge/max(dvs.size,1):.2f} "
                  f"contfail={n_fail/max(dvs.size+n_fail,1):.2f}", flush=True)
    return results


# ----------------------------------------------------------------- stage B ---
def stage_b(flux, wave, truth, pred_path, limit=None):
    import csv

    z_pred_list, row_list = [], []
    with open(pred_path) as fh:
        for row in csv.DictReader(fh):
            for slot in ("1", "2"):
                z = float(row[f"Z_DLA{slot}"])
                if z <= 0:
                    continue
                z_pred_list.append(z)
                row_list.append(int(row["id"]))
    z_pred = np.asarray(z_pred_list)
    rows = np.asarray(row_list)
    if limit:
        z_pred, rows = z_pred[:limit], rows[:limit]

    # truth catalogue (band + lognhi filtered, same as the scorer)
    tz, tr = [], []
    for r in range(len(truth["N_DLA"])):
        for slot in ("1", "2"):
            if truth["N_DLA"][r] < int(slot):
                continue
            ln = truth[f"LOGNHI{slot}"][r]
            zt = truth[f"Z_DLA{slot}"][r]
            if ln < 20.3 or not (1.098 <= zt < 2.455):
                continue
            tz.append(zt)
            tr.append(r)
    tz = np.asarray(tz)

    pairs = greedy_match(tz, z_pred)
    print(f"  predictions={len(z_pred)} truth={len(tz)} matched={len(pairs)}",
          flush=True)

    dv_base, dv_ref, logn_ref, keep_rows = [], [], [], []
    n_edge, n_fail = 0, 0
    dpx = float(np.median(np.diff(wave[0])))
    for k, (ti, pi, dv0) in enumerate(pairs):
        r = rows[pi]
        w = wave[r]
        f = flux[r]
        lam_pred = LAM_LYA * (1.0 + z_pred[pi])
        ip = int(np.argmin(np.abs(w - lam_pred)))
        if ip < HALF_WIN + 1 or ip > len(w) - HALF_WIN - 2:
            continue
        sl = slice(ip - HALF_WIN, ip + HALF_WIN + 1)
        res = fit_one(f[sl], w[sl], w[ip])
        if res is None:
            n_fail += 1
            continue
        lam_fit, logn_fit, _, edge = res
        z_fit = lam_fit / LAM_LYA - 1.0
        dv_base.append(dv0)
        dv_ref.append(dv_of(z_fit, tz[ti]))
        logn_ref.append(logn_fit)
        n_edge += int(edge)
        if (k + 1) % 200 == 0:
            print(f"    {k+1}/{len(pairs)}", flush=True)

    dv_base = np.asarray(dv_base)
    dv_ref = np.asarray(dv_ref)
    out = {
        "n_matched": int(len(pairs)),
        "n_fitted": int(dv_base.size),
        "baseline_std_dv": float(np.std(dv_base)),
        "baseline_mean_dv": float(np.mean(dv_base)),
        "refined_std_dv": float(np.std(dv_ref)),
        "refined_mean_dv": float(np.mean(dv_ref)),
        "refined_mad_dv": float(np.median(np.abs(dv_ref - np.median(dv_ref))) * 1.4826),
        "edge_hit_rate": float(n_edge / max(dv_base.size, 1)),
        "continuum_fail_rate": float(n_fail / max(dv_base.size + n_fail, 1)),
        "median_lognhi_fit": float(np.median(logn_ref)),
    }
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--pred", default=DEFAULT_PRED)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-inject", type=int, default=400)
    ap.add_argument("--ideal", action="store_true",
                    help="stage A: flat continuum + controlled white noise")
    ap.add_argument("--oracle-cont", action="store_true",
                    help="stage A: feed the fit a smoothed version of the "
                         "pre-injection spectrum as the continuum")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"C_TAU = {C_TAU:.6e}   eps_rest = {EPS_REST:.4f} A", flush=True)
    print("loading flux ...", flush=True)
    flux, wave = read_flux()
    truth = read_truth()
    print(f"  flux {flux.shape}  truth {len(truth['N_DLA'])}", flush=True)

    if args.stage == "A":
        res = stage_a(flux, wave, truth, n_per_bin=args.n_inject,
                      ideal=args.ideal, oracle_cont=args.oracle_cont)
        payload = {"stage": "A", "ideal": args.ideal,
                   "oracle_cont": args.oracle_cont, "results": res}
    else:
        res = stage_b(flux, wave, truth, args.pred, args.limit)
        payload = {"stage": "B", "results": res}

    out = args.out or f"/tmp/voigt_{args.stage}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
