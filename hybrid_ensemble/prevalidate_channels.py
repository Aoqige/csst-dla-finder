"""Pre-validation of the new engineered input channels (flux_sig / flux_feat).

Why this exists
---------------
The instrument that condemned ``resid``/``grad`` used three things the original
probe lacked: a scan-level (no-oracle) statistic, noise normalisation, and an
SNR-matched no-DLA control.  We apply *the same instrument* to the new
non-linear channels so the verdict on them comes from the same yardstick.

Two questions
-------------
Q1 detectability : as a scan-level detector (position unknown), does the channel
                   separate DLA from no-DLA better than the raw flux channel?
                   -> single-feature AUC, binned by SNR.

Q2 redundancy    : can a conv stack synthesise the channel from the six
                   flux-mode inputs?  If yes, adding it buys no *accessibility*.
                     B1 linear conv k=7 (least squares)  -- lower bound
                     B2 3x Conv1d(k=7,d=192) + GELU      -- same shape as the
                        transformer's conv stem; upper bound on what the
                        architecture can synthesise.

Caveat stated up front: Q2 is a *necessary* condition, not a predictor of the
training delta.  A channel the stem can already compute is certainly redundant;
a channel it cannot compute is only *potentially* useful (optimisation and
interaction effects decide the rest, and those only training can see).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import astropy.io.fits as t
from scipy.ndimage import median_filter, uniform_filter1d

sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/hybrid_ensemble"))
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import (  # noqa: E402
    build_engineered_views,
    local_noise_sigma,
    normalize_with_scale,
    smooth_flux,
    wavelength_norm,
)

LYA = 1215.67
WIN = 6
WLEN = 2 * WIN + 1
MIN_Z = 1.10
TEST = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"

# downward channels: absorption == negative excursion.  mf is pre sign-flipped.
DOWNWARD = {"flux", "smooth", "resid", "grad", "sig", "trough"}


def robust_sigma(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return float(1.4826 * mad)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort() + 1
    rp = r[: pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def snr_bin(s: float) -> str:
    if not np.isfinite(s):
        return "nan"
    if s < 1:
        return "SNR<1"
    if s < 2:
        return "SNR1-2"
    if s < 3:
        return "SNR2-3"
    return "SNR3+"


def main() -> None:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    th = t.open(TRUTH)
    truth = th["TRUTH"].data
    wave = np.asarray(th["WAVELENGTH"].data, dtype=np.float32)
    th.close()
    tf = t.open(TEST)
    raw = np.asarray(tf["FLUX"].data, dtype=np.float32)
    meta = tf["META"].data
    tf.close()

    n_spec, n_wave = raw.shape
    snr = np.asarray(truth["SNR_GU"], dtype=float)
    ln1 = np.asarray(truth["LOGNHI1"], dtype=float)
    has = np.asarray(truth["HAS_DLA"], dtype=bool)
    zq = np.asarray(meta["Z_QSO"], dtype=float)

    scale = np.percentile(raw, 75, axis=1, keepdims=True).astype(np.float32)
    wave_n = wavelength_norm(wave)

    lo = max(WIN, int(np.argmin(np.abs(wave - LYA * (1.0 + MIN_Z)))))
    hi = n_wave - WIN - 1

    chans = {k: np.full(n_spec, np.nan) for k in ("flux", "smooth", "resid", "grad", "sig", "trough", "mf")}
    # six flux-mode inputs, for the reconstructibility probes
    X = np.empty((n_spec, 6, n_wave), dtype=np.float32)
    # targets.  resid/sig/trough/mf are *local* (window <= 31 px) and were shown
    # to be largely synthesised by the stem.  The cnorm*/snrc group is the
    # *non-local* class: their windows (101-201 px) far exceed the stem's ~15 px
    # receptive field, so this is where a surviving candidate must come from.
    Y = {
        k: np.empty((n_spec, n_wave), dtype=np.float32)
        for k in ("sig", "trough", "mf", "resid", "cnorm101", "cnorm201", "snrc")
    }

    t0 = time.time()
    for i in range(n_spec):
        flux = normalize_with_scale(raw[i], scale[i])
        smooth = smooth_flux(flux, 15)
        resid = (flux - smooth).astype(np.float32)
        grad = np.gradient(flux).astype(np.float32)
        sig, trough, mf = build_engineered_views(flux, smooth)

        X[i, 0] = flux
        X[i, 1] = smooth
        X[i, 2] = zq[i]
        X[i, 3] = snr[i]
        X[i, 4] = wave_n
        X[i, 5] = (wave < LYA * (1.0 + zq[i])).astype(np.float32)
        Y["sig"][i] = sig
        Y["trough"][i] = trough
        Y["mf"][i] = mf
        Y["resid"][i] = resid
        # non-local candidates
        cont101 = median_filter(flux, 101, mode="reflect")
        cont201 = median_filter(flux, 201, mode="reflect")
        Y["cnorm101"][i] = flux / np.maximum(cont101, 1e-6)
        Y["cnorm201"][i] = flux / np.maximum(cont201, 1e-6)
        Y["snrc"][i] = uniform_filter1d(local_noise_sigma(resid), 101, mode="reflect")

        for name, arr in (("flux", flux), ("smooth", smooth), ("resid", resid),
                          ("grad", grad), ("sig", sig), ("trough", trough), ("mf", mf)):
            sgn = -1.0 if name in DOWNWARD else 1.0
            v = sgn * arr
            sig_c = robust_sigma(v)
            if not np.isfinite(sig_c) or sig_c <= 1e-9:
                continue
            # deepest absorption anywhere in the allowed search range
            chans[name][i] = float(np.nanmax(v[lo:hi])) / sig_c

        if (i + 1) % 5000 == 0:
            print(f"  built {i + 1}/{n_spec}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"  channel build done in {time.time() - t0:.0f}s\n")

    # ---------------- Q1: scan-level detectability ----------------
    dla = has & np.isfinite(ln1) & (ln1 >= 20.3)
    nodla = ~has
    bins = ["SNR<1", "SNR1-2", "SNR2-3", "SNR3+"]
    sb = np.array([snr_bin(s) for s in snr])

    print("=" * 96)
    print("Q1  scan-level detectability: AUC(DLA vs no-DLA), position NOT given (no oracle)")
    print("=" * 96)
    hdr = f"{'channel':9s} {'AUC(all)':>9s} " + " ".join(f"{b:>8s}" for b in bins)
    print(hdr)
    for name in ("flux", "smooth", "resid", "grad", "sig", "trough", "mf"):
        v = chans[name]
        row = [f"{auc(v[dla], v[nodla]):9.3f}"]
        for b in bins:
            m = sb == b
            row.append(f"{auc(v[dla & m], v[nodla & m]):8.3f}")
        print(f"{name:9s} " + " ".join(row))

    # ---------------- Q2: reconstructibility ----------------
    print()
    print("=" * 96)
    print("Q2  reconstructibility from the six flux-mode inputs")
    print("=" * 96)
    print("  B1 = linear conv k=7, least squares  (what a *linear* path can reach)")
    print("  B2 = Conv1d(6->192,k7) x3 + GELU -> 1  (same shape as the transformer stem)")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_spec)
    tr, va = perm[:16000], perm[16000:]

    def lincv(Xs, ys):
        """Accumulate normal equations for a linear k=7 conv, then report R^2."""
        XtX = np.zeros((42, 42), dtype=np.float64)
        Xty = np.zeros(42, dtype=np.float64)
        for i in Xs:
            pad = np.pad(X[i], ((0, 0), (3, 3)), mode="reflect")
            W = np.lib.stride_tricks.sliding_window_view(pad, 7, axis=1)  # [6,681,7]
            W = np.transpose(W, (1, 0, 2)).reshape(n_wave, 42)
            XtX += W.T @ W
            Xty += W.T @ ys[i]
        coef = np.linalg.solve(XtX + 1e-6 * np.eye(42), Xty)
        yv, yh = [], []
        for i in va:
            pad = np.pad(X[i], ((0, 0), (3, 3)), mode="reflect")
            W = np.lib.stride_tricks.sliding_window_view(pad, 7, axis=1)
            W = np.transpose(W, (1, 0, 2)).reshape(n_wave, 42)
            yv.append(ys[i])
            yh.append(W @ coef)
        yv, yh = np.concatenate(yv), np.concatenate(yh)
        return 1.0 - np.sum((yv - yh) ** 2) / np.sum((yv - yv.mean()) ** 2)

    TARGETS = ("resid", "sig", "mf", "trough", "cnorm101", "cnorm201", "snrc")
    r2_lin = {k: lincv(tr, Y[k]) for k in TARGETS}
    for k in TARGETS:
        print(f"  B1  {k:9s} R2 = {r2_lin[k]:6.3f}")

    # B2 - small conv net
    try:
        import torch
        import torch.nn as nn

        dev = torch.device(device)
        torch.manual_seed(0)

        class Net(nn.Module):
            def __init__(self, c_in=6, d=192, k=7):
                super().__init__()
                p = k // 2
                self.net = nn.Sequential(
                    nn.Conv1d(c_in, d, k, padding=p), nn.GELU(),
                    nn.Conv1d(d, d, k, padding=p), nn.GELU(),
                    nn.Conv1d(d, d, k, padding=p), nn.GELU(),
                    nn.Conv1d(d, 1, 1),
                )

            def forward(self, x):
                return self.net(x).squeeze(1)

        Xtr = torch.tensor(X[tr], device=dev)
        Xva = torch.tensor(X[va], device=dev)
        for k in TARGETS:
            ytr = torch.tensor(Y[k][tr], device=dev)
            yva = torch.tensor(Y[k][va], device=dev)
            net = Net().to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
            best = -9.9
            bs = 128
            for ep in range(25):
                net.train()
                idx = torch.randperm(len(tr), device=dev)
                for s in range(0, len(tr), bs):
                    b = idx[s:s + bs]
                    opt.zero_grad()
                    loss = nn.functional.mse_loss(net(Xtr[b]), ytr[b])
                    loss.backward()
                    opt.step()
                sched.step()
                net.eval()
                with torch.no_grad():
                    pr = net(Xva)
                    r2 = (1 - ((yva - pr) ** 2).sum() / ((yva - yva.mean()) ** 2).sum()).item()
                best = max(best, r2)
            print(f"    B2 {k:9s} best val R2 = {best:6.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  B2 skipped: {exc}")

    print()
    print("Reading: B1 ~ 1.0  => the channel is a linear function of the inputs (redundant).")
    print("         B2 high   => the conv stem can synthesise it   (no new accessibility).")
    print("         both low  => genuinely new accessibility; training decides usefulness.")


if __name__ == "__main__":
    main()
