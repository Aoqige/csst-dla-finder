from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import LYMAN_ALPHA_A

FEATURE_MODES = {"flux", "flux_clean", "residual", "flux_residual", "all"}
LABEL_COLUMNS = ["Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"]


def normalize_spectrum(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)

    median = np.median(values[finite]).astype(np.float32)
    values = np.where(finite, values, median).astype(np.float32, copy=False)
    std = np.std(values).astype(np.float32)
    if not np.isfinite(std) or std < eps:
        std = np.float32(1.0)
    values = (values - median) / std
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def build_spectrum_features(
    flux: np.ndarray,
    flux_clean: np.ndarray | None,
    feature_mode: str,
) -> np.ndarray:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode={feature_mode!r}. Choose from {sorted(FEATURE_MODES)}")

    channels: list[np.ndarray] = []
    if feature_mode in {"flux", "flux_residual", "all"}:
        channels.append(normalize_spectrum(flux))
    if feature_mode in {"flux_clean", "all"}:
        if flux_clean is None:
            raise ValueError(f"flux_clean is required for feature_mode={feature_mode}")
        channels.append(normalize_spectrum(flux_clean))
    if feature_mode in {"residual", "flux_residual", "all"}:
        if flux_clean is None:
            raise ValueError(f"flux_clean is required for feature_mode={feature_mode}")
        channels.append(normalize_spectrum(np.asarray(flux_clean, dtype=np.float32) - np.asarray(flux, dtype=np.float32)))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def feature_channels(feature_mode: str) -> int:
    if feature_mode == "all":
        return 3
    if feature_mode == "flux_residual":
        return 2
    if feature_mode in {"flux", "flux_clean", "residual"}:
        return 1
    raise ValueError(f"Unknown feature_mode={feature_mode!r}")


def feature_requires_flux_clean(feature_mode: str) -> bool:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unknown feature_mode={feature_mode!r}. Choose from {sorted(FEATURE_MODES)}")
    return feature_mode != "flux"


def wavelength_to_fractional_index(wavelength: np.ndarray, z_dla: float) -> float:
    target = LYMAN_ALPHA_A * (1.0 + float(z_dla))
    grid = np.arange(len(wavelength), dtype=np.float32)
    return float(np.interp(target, wavelength.astype(np.float32), grid))


def gaussian_peak(length: int, center: float, sigma_bins: float) -> np.ndarray:
    x = np.arange(length, dtype=np.float32)
    return np.exp(-0.5 * ((x - float(center)) / float(sigma_bins)) ** 2).astype(np.float32)


@dataclass(frozen=True)
class TargetConfig:
    sigma_bins: float = 2.0
    region_half_width_bins: int = 8
    region_lognhi_scale: float = 2.0


def make_detection_targets(
    wavelength: np.ndarray,
    n_dla: int,
    z_dlas: np.ndarray,
    lognhis: np.ndarray,
    config: TargetConfig,
) -> dict[str, np.ndarray]:
    length = len(wavelength)
    heatmap = np.zeros(length, dtype=np.float32)
    region = np.zeros(length, dtype=np.float32)
    lognhi_sum = np.zeros(length, dtype=np.float32)
    lognhi_weight = np.zeros(length, dtype=np.float32)
    offset = np.zeros(length, dtype=np.float32)
    offset_weight = np.zeros(length, dtype=np.float32)

    for slot in range(int(np.clip(n_dla, 0, 2))):
        z_dla = float(z_dlas[slot])
        lognhi = float(lognhis[slot])
        if not np.isfinite(z_dla) or not np.isfinite(lognhi):
            continue
        if z_dla < 0:
            continue

        frac_idx = wavelength_to_fractional_index(wavelength, z_dla)
        center_idx = int(np.clip(np.round(frac_idx), 0, length - 1))
        peak = gaussian_peak(length, frac_idx, config.sigma_bins)
        heatmap = np.maximum(heatmap, peak)
        lognhi_sum += peak * np.float32(lognhi)
        lognhi_weight += peak

        half_width = int(round(config.region_half_width_bins + config.region_lognhi_scale * max(lognhi - 20.3, 0.0)))
        lo = max(center_idx - half_width, 0)
        hi = min(center_idx + half_width + 1, length)
        region[lo:hi] = 1.0

        offset[center_idx] = np.float32(frac_idx - center_idx)
        offset_weight[center_idx] = 1.0

    lognhi_map = np.zeros(length, dtype=np.float32)
    nonzero = lognhi_weight > 0
    lognhi_map[nonzero] = lognhi_sum[nonzero] / lognhi_weight[nonzero]
    return {
        "heatmap": heatmap,
        "region": region,
        "lognhi_map": lognhi_map,
        "lognhi_weight": lognhi_weight,
        "offset": offset,
        "offset_weight": offset_weight,
    }


class DLATrainDataset(Dataset):
    def __init__(
        self,
        wavelength: np.ndarray,
        flux: np.ndarray,
        flux_clean: np.ndarray | None,
        labels: pd.DataFrame,
        feature_mode: str = "all",
        target_config: TargetConfig | None = None,
        indices: np.ndarray | None = None,
    ):
        if len(flux) != len(labels):
            raise ValueError(f"flux rows ({len(flux)}) != labels rows ({len(labels)})")
        if feature_requires_flux_clean(feature_mode) and flux_clean is None:
            raise ValueError(f"flux_clean is required for feature_mode={feature_mode}")
        if flux_clean is not None and len(flux_clean) != len(flux):
            raise ValueError(f"flux_clean rows ({len(flux_clean)}) != flux rows ({len(flux)})")

        missing = {"Z_QSO", "N_DLA", *LABEL_COLUMNS} - set(labels.columns)
        if missing:
            raise KeyError(f"Missing label columns: {sorted(missing)}")

        self.wavelength = np.asarray(wavelength, dtype=np.float32)
        self.flux = flux
        self.flux_clean = flux_clean
        self.feature_mode = feature_mode
        self.target_config = target_config or TargetConfig()

        if indices is None:
            self.indices = np.arange(len(labels), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        labels_reset = labels.reset_index(drop=True)
        self.z_qso = labels_reset["Z_QSO"].to_numpy(dtype=np.float32)
        self.n_dla = np.clip(labels_reset["N_DLA"].to_numpy(dtype=np.int64), 0, 2)
        self.z_dlas = labels_reset[["Z_DLA1", "Z_DLA2"]].to_numpy(dtype=np.float32)
        self.lognhis = labels_reset[["LOGNHI1", "LOGNHI2"]].to_numpy(dtype=np.float32)
        self.snr_gu = (
            labels_reset["SNR_GU"].to_numpy(dtype=np.float32)
            if "SNR_GU" in labels_reset
            else np.full(len(labels_reset), np.nan, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        flux_clean_row = None if self.flux_clean is None else self.flux_clean[idx]
        spectrum = build_spectrum_features(self.flux[idx], flux_clean_row, self.feature_mode)
        targets = make_detection_targets(
            self.wavelength,
            int(self.n_dla[idx]),
            self.z_dlas[idx],
            self.lognhis[idx],
            self.target_config,
        )
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "spectrum": torch.from_numpy(spectrum),
            "z_qso": torch.tensor([self.z_qso[idx]], dtype=torch.float32),
            "n_dla": torch.tensor(int(self.n_dla[idx]), dtype=torch.long),
            "snr_gu": torch.tensor([self.snr_gu[idx]], dtype=torch.float32),
            "heatmap": torch.from_numpy(targets["heatmap"]),
            "region": torch.from_numpy(targets["region"]),
            "lognhi_map": torch.from_numpy(targets["lognhi_map"]),
            "lognhi_weight": torch.from_numpy(targets["lognhi_weight"]),
            "offset": torch.from_numpy(targets["offset"]),
            "offset_weight": torch.from_numpy(targets["offset_weight"]),
        }


class DLATestDataset(Dataset):
    def __init__(
        self,
        flux: np.ndarray,
        flux_clean: np.ndarray | None,
        meta: pd.DataFrame,
        feature_mode: str = "all",
    ):
        if len(flux) != len(meta):
            raise ValueError(f"flux rows ({len(flux)}) != meta rows ({len(meta)})")
        if feature_requires_flux_clean(feature_mode) and flux_clean is None:
            raise ValueError(f"flux_clean is required for feature_mode={feature_mode}")
        if flux_clean is not None and len(flux_clean) != len(flux):
            raise ValueError(f"flux_clean rows ({len(flux_clean)}) != flux rows ({len(flux)})")
        if "Z_QSO" not in meta.columns:
            raise KeyError("Missing META column: Z_QSO")
        self.flux = flux
        self.flux_clean = flux_clean
        self.meta = meta.reset_index(drop=True)
        self.feature_mode = feature_mode
        self.z_qso = self.meta["Z_QSO"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.flux)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        flux_clean_row = None if self.flux_clean is None else self.flux_clean[idx]
        spectrum = build_spectrum_features(self.flux[idx], flux_clean_row, self.feature_mode)
        return {
            "index": torch.tensor(idx, dtype=torch.long),
            "spectrum": torch.from_numpy(spectrum),
            "z_qso": torch.tensor([self.z_qso[idx]], dtype=torch.float32),
        }
