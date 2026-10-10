from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def _require_hdu(hdul: fits.HDUList, name: str) -> Any:
    try:
        return hdul[name].data
    except (KeyError, IndexError) as exc:
        names = [hdu.name for hdu in hdul]
        raise KeyError(f"Missing FITS extension {name!r}. Available HDUs: {names}") from exc


def _native_dataframe(table_data: Any) -> pd.DataFrame:
    if table_data is None:
        raise ValueError("FITS table extension is empty.")

    df = Table(table_data).to_pandas()
    for col in df.columns:
        series = df[col]
        if series.dtype == object:
            df[col] = series.apply(
                lambda value: value.decode("utf-8").strip()
                if isinstance(value, (bytes, bytearray))
                else value
            )
        elif np.issubdtype(series.dtype, np.floating):
            df[col] = np.asarray(series, dtype=np.float32)
        elif np.issubdtype(series.dtype, np.integer):
            df[col] = np.asarray(series, dtype=np.int64)
    return df


def load_train_fits(
    path: str | Path,
    load_flux_clean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with fits.open(path, memmap=True) as hdul:
        wavelength = np.array(_require_hdu(hdul, "WAVELENGTH"), dtype=np.float32, copy=True)
        flux = np.array(_require_hdu(hdul, "FLUX"), dtype=np.float32, copy=True)
        flux_clean = (
            np.array(_require_hdu(hdul, "FLUX_CLEAN"), dtype=np.float32, copy=True)
            if load_flux_clean
            else None
        )
        labels = _native_dataframe(_require_hdu(hdul, "LABELS"))
    return wavelength, flux, flux_clean, labels


def load_test_fits(
    path: str | Path,
    load_flux_clean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with fits.open(path, memmap=True) as hdul:
        wavelength = np.array(_require_hdu(hdul, "WAVELENGTH"), dtype=np.float32, copy=True)
        flux = np.array(_require_hdu(hdul, "FLUX"), dtype=np.float32, copy=True)
        flux_clean = (
            np.array(_require_hdu(hdul, "FLUX_CLEAN"), dtype=np.float32, copy=True)
            if load_flux_clean
            else None
        )
        meta = _native_dataframe(_require_hdu(hdul, "META"))
    return wavelength, flux, flux_clean, meta


def load_truth_fits(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with fits.open(path, memmap=True) as hdul:
        wavelength = np.array(_require_hdu(hdul, "WAVELENGTH"), dtype=np.float32, copy=True)
        truth = _native_dataframe(_require_hdu(hdul, "TRUTH"))
    return wavelength, truth


def summarize_train_fits(path: str | Path) -> dict[str, Any]:
    wavelength, flux, flux_clean, labels = load_train_fits(path)
    if flux_clean is None:
        raise RuntimeError("FLUX_CLEAN was not loaded while summarizing the training FITS.")
    n_dla = labels["N_DLA"].to_numpy(dtype=np.int64)
    has_dla = labels["HAS_DLA"].to_numpy(dtype=np.int64) if "HAS_DLA" in labels else (n_dla > 0).astype(np.int64)

    return {
        "wavelength_shape": list(wavelength.shape),
        "wavelength_min": float(np.nanmin(wavelength)),
        "wavelength_max": float(np.nanmax(wavelength)),
        "wavelength_step_median": float(np.nanmedian(np.diff(wavelength))),
        "flux_shape": list(flux.shape),
        "flux_clean_shape": list(flux_clean.shape),
        "label_columns": list(labels.columns),
        "n_dla_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(n_dla, return_counts=True))},
        "has_dla_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(has_dla, return_counts=True))},
        "z_qso_min": float(np.nanmin(labels["Z_QSO"])),
        "z_qso_max": float(np.nanmax(labels["Z_QSO"])),
    }
