from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import wofz

from .constants import LYMAN_ALPHA_A

CM_PER_A = 1e-8
C_CGS = 2.99792458e10
E_ESU = 4.803204712570263e-10
M_E_G = 9.1093837015e-28


@dataclass(frozen=True)
class VoigtFitResult:
    z_dla: float
    lognhi: float
    delta_chi2: float
    null_chi2: float
    best_chi2: float


def voigt_transmission_lya(
    wavelength: np.ndarray,
    z_dla: float,
    lognhi: float,
    b_kms: float = 30.0,
    gamma: float = 6.265e8,
    oscillator_strength: float = 0.4164,
) -> np.ndarray:
    wavelength = np.asarray(wavelength, dtype=np.float64)
    rest_wavelength_cm = wavelength * CM_PER_A / (1.0 + float(z_dla))
    lambda0_cm = LYMAN_ALPHA_A * CM_PER_A
    nu = C_CGS / rest_wavelength_cm
    nu0 = C_CGS / lambda0_cm
    b_cms = float(b_kms) * 1e5
    delta_nu_d = nu0 * b_cms / C_CGS
    a = gamma / (4.0 * np.pi * delta_nu_d)
    u = (nu - nu0) / delta_nu_d
    profile = np.real(wofz(u + 1j * a))
    sigma = np.sqrt(np.pi) * E_ESU**2 / (M_E_G * C_CGS) * oscillator_strength / delta_nu_d * profile
    tau = np.power(10.0, float(lognhi)) * sigma
    tau = np.nan_to_num(tau, nan=0.0, posinf=700.0, neginf=0.0)
    return np.exp(-np.clip(tau, 0.0, 700.0)).astype(np.float32)


def fit_voigt_local(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_clean: np.ndarray,
    z_init: float,
    lognhi_init: float,
    z_half_window: float = 0.02,
    z_steps: int = 9,
    lognhi_half_window: float = 0.5,
    lognhi_steps: int = 21,
    rest_window_a: float = 80.0,
) -> VoigtFitResult:
    wavelength = np.asarray(wavelength, dtype=np.float32)
    flux = np.asarray(flux, dtype=np.float32)
    flux_clean = np.asarray(flux_clean, dtype=np.float32)

    center_wavelength = LYMAN_ALPHA_A * (1.0 + float(z_init))
    local_width = rest_window_a * (1.0 + float(z_init))
    mask = np.abs(wavelength - center_wavelength) <= local_width
    if mask.sum() < 5:
        return VoigtFitResult(float(z_init), float(lognhi_init), 0.0, float("nan"), float("nan"))

    wl = wavelength[mask]
    y = flux[mask]
    continuum = np.where(np.abs(flux_clean[mask]) > 1e-30, flux_clean[mask], 1e-30)
    null_model = continuum
    null_chi2 = float(np.mean((y - null_model) ** 2))

    z_grid = np.linspace(float(z_init) - z_half_window, float(z_init) + z_half_window, z_steps)
    log_grid = np.linspace(float(lognhi_init) - lognhi_half_window, float(lognhi_init) + lognhi_half_window, lognhi_steps)
    log_grid = np.clip(log_grid, 19.0, 23.0)
    best = (float(z_init), float(lognhi_init), float("inf"))
    for z_dla in z_grid:
        for lognhi in log_grid:
            transmission = voigt_transmission_lya(wl, float(z_dla), float(lognhi))
            model = continuum * transmission
            chi2 = float(np.mean((y - model) ** 2))
            if chi2 < best[2]:
                best = (float(z_dla), float(lognhi), chi2)

    delta_chi2 = (null_chi2 - best[2]) / max(null_chi2, 1e-30)
    return VoigtFitResult(best[0], best[1], float(delta_chi2), null_chi2, best[2])
