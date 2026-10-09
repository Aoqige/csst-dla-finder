#!/usr/bin/env python3
"""Why does the Voigt z-refinement fail?  Two decisive measurements.

Part 1 - empirical wing geometry.  Stack the REAL truth DLAs (band + logNHI
         selected exactly like the scorer) around their true z, normalise each
         spectrum with the same continuum estimator the fit uses, and read off
         how many pixels the observable trough actually occupies and how deep
         it is relative to the per-pixel noise.

Part 2 - Cramer-Rao floor.  For a template fit with known continuum, the best
         achievable sigma_z is 1/sqrt( sum_px (dT/dz)^2 / sigma_px^2 ).
         Converted to km/s this is a HARD lower bound on std_dv for ANY
         damping-wing fit.  Compare with what the fit actually achieves.

Part 3 - ideal-mode sigma sweep (runs in background, see --stage C).

Compliance: raw FLUX only, never FLUX_CLEAN.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "/tmp")
import voigt_zrefine as V  # noqa: E402


def part1_geometry(flux, wave, truth, n_stack=4000):
    """Empirical stacked trough profile for real DLAs, in pixels."""
    dpx = float(np.median(np.diff(wave[0])))
    print(f"dpx = {dpx:.4f} A   C_TAU = {V.C_TAU:.4e}")

    # truth catalogue, same band + logNHI cut as the scorer
    tz, tr, tl = [], [], []
    for r in range(len(truth["N_DLA"])):
        for slot in ("1", "2"):
            if truth["N_DLA"][r] < int(slot):
                continue
            ln = float(truth[f"LOGNHI{slot}"][r])
            zt = float(truth[f"Z_DLA{slot}"][r])
            if ln < 20.3 or not (1.098 <= zt < 2.455):
                continue
            tz.append(zt)
            tr.append(r)
            tl.append(ln)
    tz = np.asarray(tz)
    tr = np.asarray(tr)
    tl = np.asarray(tl)
    print(f"truth DLAs (band+logN cut): {tz.size}")

    bins = [(20.3, 20.5), (20.5, 20.8), (20.8, 21.1), (21.1, 21.6)]
    out = []
    print()
    print(f"{'logNHI bin':>14} {'n':>6} {'z_med':>6} {'depth@0px':>10} "
          f"{'px<0.9':>7} {'px<0.5':>7} {'px<0.2':>7} "
          f"{'sig_px':>8} {'dz_vel':>8}")
    for lo, hi in bins:
        m = (tl >= lo) & (tl < hi)
        if m.sum() < 30:
            continue
        idx = np.flatnonzero(m)
        if idx.size > n_stack:
            idx = np.random.default_rng(0).choice(idx, n_stack, replace=False)
        profs = []
        for i in idx:
            r = int(tr[i])
            w = wave[r]
            f = flux[r]
            # allow fractional pixel position: use nearest pixel centre
            ip = int(np.argmin(np.abs(w - V.LAM_LYA * (1.0 + tz[i]))))
            if ip < V.HALF_WIN + 1 or ip > len(w) - V.HALF_WIN - 2:
                continue
            sl = slice(ip - V.HALF_WIN, ip + V.HALF_WIN + 1)
            fw = f[sl]
            ww = w[sl]
            cont = V._continuum(fw, ww)
            if np.median(cont) <= 0:
                continue
            profs.append(fw / cont)
        if not profs:
            continue
        P = np.asarray(profs)                       # [n, 2*HALF_WIN+1]
        med = np.median(P, axis=0)                  # robust stacked profile
        ipc = med.size // 2
        # per-pixel scatter of the *stack* -> the noise the fit contends with
        sig_px = 1.4826 * np.median(np.abs(P - med), axis=0)
        sig_stack = float(np.median(sig_px) / np.sqrt(P.shape[0]))
        n09 = int((med < 0.9).sum())
        n05 = int((med < 0.5).sum())
        n02 = int((med < 0.2).sum())
        # velocity error implied if z is pinned only to the pixel grid
        zz = float(np.median(tz[idx]))
        dz_vel = V.C_KMS * dpx / (V.LAM_LYA * (1.0 + zz))
        print(f"{lo:6.1f}-{hi:4.1f} {P.shape[0]:6d} {zz:6.2f} "
              f"{med[ipc]:10.3f} {n09:7d} {n05:7d} {n02:7d} "
              f"{float(np.median(sig_px)):8.3f} {dz_vel:8.0f}")
        out.append({
            "bin": [lo, hi], "n": int(P.shape[0]), "z_med": zz,
            "depth_center": float(med[ipc]),
            "n_px_below_0.9": n09, "n_px_below_0.5": n05,
            "n_px_below_0.2": n02,
            "per_pixel_scatter": float(np.median(sig_px)),
            "stack_noise": sig_stack,
            "km_s_per_px": dz_vel,
            "profile": [float(x) for x in med],
        })
    return {"dpx": dpx, "bins": out}


def part2_crb(flux, wave, truth, per_pixel_sigma=0.276, z_ref=1.5):
    """Cramer-Rao floor for a damping-wing template fit (continuum known)."""
    dpx = float(np.median(np.diff(wave[0])))
    out = []
    print()
    print("template-fit Cramer-Rao floor (continuum known, per-pixel sigma=%.3f)"
          % per_pixel_sigma)
    print(f"{'logN':>6} {'T@0':>7} {'dT/dz rms':>10} {'CRB_dv':>9} "
          f"{'vs 233':>8} {'vs 160':>8}")
    for logn in (19.8, 20.0, 20.3, 20.5, 20.8, 21.0, 21.3, 21.6):
        N = 10.0**logn
        lam_c = V.LAM_LYA * (1.0 + z_ref)
        # +-25 px observed window, pixel-integrated like the fit
        xx = np.arange(-V.HALF_WIN, V.HALF_WIN + 1) * dpx
        lam_pix = lam_c + xx

        def prof(z):
            lc = V.LAM_LYA * (1.0 + z)
            fine = lam_pix[:, None] + (np.arange(V.SUB) + 0.5) * (dpx / V.SUB)
            dlam_rest_cm = ((fine - lc) / (1.0 + z)) * V.A_CM
            tau = V.C_TAU * N / (dlam_rest_cm**2 + (V.EPS_REST * V.A_CM) ** 2)
            return np.exp(-tau).mean(axis=1)

        # numerical dT/dz: 1 px step in z
        dz = dpx / V.LAM_LYA
        dT = (prof(z_ref + dz) - prof(z_ref - dz)) / (2.0 * dz)
        info_z = float(np.sum(dT**2) / per_pixel_sigma**2)
        crb_z = 1.0 / np.sqrt(info_z)
        crb_dv = float(V.C_KMS * crb_z / (1.0 + z_ref))
        print(f"{logn:6.1f} {prof(z_ref)[V.HALF_WIN]:7.3f} "
              f"{np.sqrt(np.mean(dT**2))*dpx:10.4f} {crb_dv:9.0f} "
              f"{crb_dv/233:8.2f} {crb_dv/160:8.2f}")
        out.append({"logn": logn, "crb_dv": crb_dv,
                    "dTdz_rms_times_dpx": float(np.sqrt(np.mean(dT**2)) * dpx)})
    return out


def part4_forest_confusion(flux, wave, truth, n_rows=1500, seed=0):
    """Is the failure noise, or the Ly-a forest?

    In DLA-free spectra the forest alone produces deep troughs.  If a forest
    line inside the fit window is as deep as the target DLA, the SSE minimiser
    locks onto the wrong feature -> the fit is hijacked, not merely imprecise.

    Reports, per injected logN:
      depth percentiles of the window minimum BEFORE injection (the confusion
      floor), and the displacement of the window's argmin AFTER injection.
    """
    rng = np.random.default_rng(seed)
    clean = np.flatnonzero(truth["N_DLA"] == 0)
    clean = rng.choice(clean, size=min(n_rows, clean.size), replace=False)
    dpx = float(np.median(np.diff(wave[0])))
    lam_lo, lam_hi = V.LAM_LYA * 1.15, V.LAM_LYA * (1.0 + 2.40)

    F = 8            # fit half window (px)
    rows, poss = [], []
    for r in clean:
        w = wave[r]
        cand = np.flatnonzero((w >= lam_lo) & (w <= lam_hi))
        cand = cand[(cand >= V.HALF_WIN + 1) & (cand <= len(w) - V.HALF_WIN - 2)]
        if cand.size == 0:
            continue
        for p in rng.choice(cand, size=min(6, cand.size), replace=False):
            rows.append(int(r))
            poss.append(int(p))
    rows = np.asarray(rows)
    poss = np.asarray(poss)
    print(f"\nforest confusion probe: {rows.size} (row, position) samples")

    outs = {"n": int(rows.size), "depth_floor": {}, "argmin": {}}
    for tag, logn_inj in (("clean", None), ("logN=20.3", 20.3),
                          ("logN=20.8", 20.8), ("logN=21.3", 21.3)):
        ycen, minv, disp = [], [], []
        for r, ip in zip(rows, poss):
            w = wave[r]
            f = flux[r]
            # fixed, non-degenerate normalisation: per-spectrum p75, exactly
            # what the trained models see upstream
            c0 = float(np.percentile(f, 75))
            if c0 <= 0:
                continue
            sl = slice(ip - V.HALF_WIN, ip + V.HALF_WIN + 1)
            ww = w[sl]
            ff = f[sl]
            if logn_inj is None:
                y = ff / c0
            else:
                zt = ww[V.HALF_WIN] / V.LAM_LYA - 1.0
                dl = (ww - ww[V.HALF_WIN]) / (1.0 + zt) * V.A_CM
                tau = V.C_TAU * 10.0**logn_inj / (
                    dl**2 + (V.EPS_REST * V.A_CM) ** 2)
                y = (ff * np.exp(-tau)) / c0
            j = int(np.argmin(y))
            ycen.append(float(y[V.HALF_WIN]))
            minv.append(float(y[j]))
            disp.append(j - V.HALF_WIN)
        ycen = np.asarray(ycen)
        minv = np.asarray(minv)
        d = np.asarray(disp)
        print(f"  {tag:>10}: n={ycen.size}  min(y|p75) p10/p25/med "
              f"{np.percentile(minv,10):.3f}/{np.percentile(minv,25):.3f}/"
              f"{np.median(minv):.3f}   y@centre med {np.median(ycen):.3f}   "
              f"argmin within +-1px {np.mean(np.abs(d) <= 1)*100:.1f}%   "
              f"|d|>2px {np.mean(np.abs(d) > 2)*100:.1f}%")
        outs["depth_floor"][tag] = {
            "min_y_p10": float(np.percentile(minv, 10)),
            "min_y_p25": float(np.percentile(minv, 25)),
            "min_y_med": float(np.median(minv)),
            "y_centre_med": float(np.median(ycen)),
        }
        outs["argmin"][tag] = {
            "frac_within_1px": float(np.mean(np.abs(d) <= 1)),
            "frac_off_gt2px": float(np.mean(np.abs(d) > 2)),
            "mad_px": float(np.median(np.abs(d - np.median(d))) * 1.4826),
        }
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "C", "D"])
    ap.add_argument("--n-inject", type=int, default=250)
    ap.add_argument("--out", default="/tmp/voigt_limits.json")
    args = ap.parse_args()

    flux, wave = V.read_flux()
    truth = V.read_truth()
    print(f"flux {flux.shape}  truth {len(truth['N_DLA'])}", flush=True)

    payload = {}
    if args.stage in ("all", "D"):
        payload["forest_confusion"] = part4_forest_confusion(flux, wave, truth)
    if args.stage == "all":
        payload["geometry"] = part1_geometry(flux, wave, truth)
        payload["crb"] = part2_crb(flux, wave, truth)
    if args.stage in ("all", "C"):
        print("\nideal-mode sigma sweep (flat continuum + white noise)",
              flush=True)
        payload["ideal_sweep"] = V.stage_a(
            flux, wave, truth, n_per_bin=args.n_inject, ideal=True,
            sigmas=(0.10, 0.20, 0.28, 0.40, 0.50),
        )

    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
