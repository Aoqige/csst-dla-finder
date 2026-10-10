from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import AUX_SUBMISSION_COLUMNS, SUBMISSION_COLUMNS
from .data import DLATestDataset, DLATrainDataset, TargetConfig, feature_channels
from .decode import Candidate, decode_model_batch
from .model import create_model
from .physics import fit_voigt_local
from .utils import safe_torch_load


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = create_model(
        input_channels=int(config.get("input_channels", feature_channels(config.get("feature_mode", "all")))),
        base_channels=int(config.get("base_channels", 96)),
        num_blocks=int(config.get("num_blocks", 8)),
        dropout=float(config.get("dropout", 0.1)),
        use_context_channels=bool(config.get("use_context_channels", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def _slots_from_candidates(candidates: list[Candidate]) -> dict[str, float]:
    values = sorted(candidates[:2], key=lambda item: item.confidence, reverse=True)
    while len(values) < 2:
        values.append(Candidate(-1.0, 0.0, 0.0, 0.0, -1))
    return {
        "N_DLA": int(sum(1 for item in values if item.z_dla >= 0)),
        "CONF1": float(values[0].confidence),
        "CONF2": float(values[1].confidence),
        "Z_DLA1": float(values[0].z_dla),
        "LOGNHI1": float(values[0].lognhi) if values[0].z_dla >= 0 else 0.0,
        "Z_DLA2": float(values[1].z_dla),
        "LOGNHI2": float(values[1].lognhi) if values[1].z_dla >= 0 else 0.0,
    }


def _refine_candidates(
    candidates: list[Candidate],
    wavelength: np.ndarray,
    flux_row: np.ndarray,
    flux_clean_row: np.ndarray,
    enabled: bool,
) -> list[Candidate]:
    if not enabled:
        return candidates
    refined: list[Candidate] = []
    for item in candidates:
        fit = fit_voigt_local(
            wavelength,
            flux_row,
            flux_clean_row,
            z_init=item.z_dla,
            lognhi_init=item.lognhi,
        )
        confidence = item.confidence * max(0.0, min(1.0, fit.delta_chi2))
        refined.append(
            Candidate(
                z_dla=fit.z_dla,
                lognhi=fit.lognhi,
                confidence=confidence,
                peak_score=item.peak_score,
                peak_index=item.peak_index,
            )
        )
    return refined


@torch.no_grad()
def predict_rows(
    model: torch.nn.Module,
    loader: DataLoader,
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_clean: np.ndarray,
    meta_or_labels: pd.DataFrame,
    device: torch.device,
    id_column: str | None,
    min_z: float | None,
    min_peak_distance: int,
    confidence_threshold: float | None,
    lognhi_min: float,
    lognhi_max: float,
    use_voigt_refine: bool,
) -> pd.DataFrame:
    rows = []
    meta = meta_or_labels.reset_index(drop=True)
    for batch in tqdm(loader, desc="predict"):
        indices = batch["index"].cpu().numpy().astype(np.int64)
        z_qso = batch["z_qso"].numpy().reshape(-1)
        outputs = model(batch["spectrum"].to(device), batch["z_qso"].to(device))
        decoded = decode_model_batch(
            outputs,
            wavelength,
            z_qso,
            min_z=min_z,
            min_peak_distance=min_peak_distance,
            confidence_threshold=confidence_threshold,
            lognhi_min=lognhi_min,
            lognhi_max=lognhi_max,
        )
        for local_row, original_idx in enumerate(indices):
            _, candidates = decoded[local_row]
            candidates = _refine_candidates(
                candidates,
                wavelength,
                flux[int(original_idx)],
                flux_clean[int(original_idx)],
                enabled=use_voigt_refine,
            )
            row_id = int(meta.iloc[int(original_idx)][id_column]) if id_column and id_column in meta.columns else int(original_idx)
            out = {
                "id": row_id,
                "index": int(original_idx),
                "SNR_GU": float(meta.iloc[int(original_idx)]["SNR_GU"]) if "SNR_GU" in meta.columns else np.nan,
            }
            out.update(_slots_from_candidates(candidates))
            rows.append(out)
    return pd.DataFrame(rows)


def make_train_dataset_from_config(
    wavelength: np.ndarray,
    flux: np.ndarray,
    flux_clean: np.ndarray,
    labels: pd.DataFrame,
    config: dict,
    indices: np.ndarray,
) -> DLATrainDataset:
    target_config = TargetConfig(
        sigma_bins=float(config.get("sigma_bins", 2.0)),
        region_half_width_bins=int(config.get("region_half_width_bins", 8)),
        region_lognhi_scale=float(config.get("region_lognhi_scale", 2.0)),
    )
    return DLATrainDataset(
        wavelength,
        flux,
        flux_clean,
        labels,
        feature_mode=config.get("feature_mode", "all"),
        target_config=target_config,
        indices=indices,
    )


def make_test_dataset_from_config(
    flux: np.ndarray,
    flux_clean: np.ndarray,
    meta: pd.DataFrame,
    config: dict,
) -> DLATestDataset:
    return DLATestDataset(
        flux,
        flux_clean,
        meta,
        feature_mode=config.get("feature_mode", "all"),
    )


def submission_frame(rows: pd.DataFrame, include_aux: bool) -> pd.DataFrame:
    clean = rows.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    clean.loc[clean["N_DLA"] == 0, ["Z_DLA1", "Z_DLA2"]] = -1.0
    clean.loc[clean["N_DLA"] == 0, ["LOGNHI1", "LOGNHI2"]] = 0.0
    clean.loc[clean["N_DLA"] <= 1, "Z_DLA2"] = -1.0
    clean.loc[clean["N_DLA"] <= 1, "LOGNHI2"] = 0.0
    if include_aux:
        return clean[AUX_SUBMISSION_COLUMNS]
    return clean[SUBMISSION_COLUMNS]
