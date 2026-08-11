from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .constants import C_KMS, LYMAN_ALPHA_A


@dataclass(frozen=True)
class Candidate:
    z_dla: float
    lognhi: float
    confidence: float
    peak_score: float
    peak_index: int


def wavelength_to_z(wavelength: np.ndarray) -> np.ndarray:
    return np.asarray(wavelength, dtype=np.float32) / LYMAN_ALPHA_A - 1.0


def fractional_index_to_z(wavelength: np.ndarray, index: float) -> float:
    grid = np.arange(len(wavelength), dtype=np.float32)
    wl = float(np.interp(float(index), grid, np.asarray(wavelength, dtype=np.float32)))
    return wl / LYMAN_ALPHA_A - 1.0


def velocity_delta_kms(z_pred: float, z_true: float) -> float:
    return C_KMS * abs(float(z_pred) - float(z_true)) / (1.0 + float(z_true))


def _local_peak_indices(score: np.ndarray, valid: np.ndarray, min_distance: int) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32)
    masked = np.where(valid, score, -np.inf)
    candidates = []
    for idx in np.where(np.isfinite(masked))[0]:
        lo = max(0, idx - min_distance)
        hi = min(len(score), idx + min_distance + 1)
        if masked[idx] >= np.max(masked[lo:hi]):
            candidates.append(int(idx))
    if not candidates:
        candidates = [int(idx) for idx in np.where(valid)[0]]
    candidates_arr = np.asarray(candidates, dtype=np.int64)
    order = np.argsort(masked[candidates_arr])[::-1]
    return candidates_arr[order]


def select_candidates(
    heatmap_score: np.ndarray,
    region_score: np.ndarray | None,
    offset: np.ndarray,
    lognhi: np.ndarray,
    count_prob: np.ndarray,
    wavelength: np.ndarray,
    z_qso: float,
    max_dlas: int = 2,
    min_z: float | None = None,
    min_peak_distance: int = 4,
    confidence_threshold: float | None = None,
    lognhi_min: float = 19.0,
    lognhi_max: float = 23.0,
) -> tuple[int, list[Candidate]]:
    z_grid = wavelength_to_z(wavelength)
    z_floor = float(np.nanmin(z_grid)) if min_z is None else float(min_z)
    z_ceil = max(float(z_qso), z_floor)
    valid = (z_grid >= z_floor) & (z_grid <= z_ceil)
    heatmap_score = np.asarray(heatmap_score, dtype=np.float32)
    if region_score is not None:
        region_score = np.asarray(region_score, dtype=np.float32)
        selection_score = heatmap_score * (0.5 + 0.5 * region_score)
    else:
        selection_score = heatmap_score

    count_prob = np.asarray(count_prob, dtype=np.float32)
    n_from_count = int(np.clip(np.argmax(count_prob), 0, max_dlas))
    n_take = max_dlas if confidence_threshold is not None else n_from_count
    if n_take <= 0:
        return n_from_count, []

    peak_indices = _local_peak_indices(selection_score, valid, min_peak_distance)
    selected: list[Candidate] = []
    for peak_idx in peak_indices:
        if any(abs(int(peak_idx) - item.peak_index) < min_peak_distance for item in selected):
            continue
        count_factor = float(count_prob[min(max(n_from_count, 1), len(count_prob) - 1)])
        peak_score = float(selection_score[peak_idx])
        confidence = peak_score * count_factor
        if confidence_threshold is not None and confidence < confidence_threshold:
            continue
        corrected_idx = float(peak_idx) + float(np.clip(offset[peak_idx], -0.5, 0.5))
        z_dla = float(np.clip(fractional_index_to_z(wavelength, corrected_idx), z_floor, z_ceil))
        value = float(np.clip(lognhi[peak_idx], lognhi_min, lognhi_max))
        selected.append(
            Candidate(
                z_dla=z_dla,
                lognhi=value,
                confidence=confidence,
                peak_score=peak_score,
                peak_index=int(peak_idx),
            )
        )
        if len(selected) >= n_take:
            break

    selected = sorted(selected, key=lambda item: item.confidence, reverse=True)
    if confidence_threshold is None:
        selected = selected[:n_from_count]
    else:
        selected = selected[:max_dlas]
        n_from_count = min(len(selected), max_dlas)
    return n_from_count, selected


@torch.no_grad()
def decode_model_batch(
    outputs: dict[str, torch.Tensor],
    wavelength: np.ndarray,
    z_qso: np.ndarray,
    min_z: float | None,
    min_peak_distance: int,
    confidence_threshold: float | None,
    lognhi_min: float,
    lognhi_max: float,
) -> list[tuple[int, list[Candidate]]]:
    heat = torch.sigmoid(outputs["heatmap_logits"]).detach().cpu().numpy()
    region = torch.sigmoid(outputs["region_logits"]).detach().cpu().numpy() if "region_logits" in outputs else None
    offset = outputs["offset"].detach().cpu().numpy()
    lognhi = outputs["lognhi"].detach().cpu().numpy()
    count_prob = torch.softmax(outputs["count_logits"], dim=1).detach().cpu().numpy()

    decoded = []
    for row_idx in range(heat.shape[0]):
        decoded.append(
            select_candidates(
                heat[row_idx],
                None if region is None else region[row_idx],
                offset[row_idx],
                lognhi[row_idx],
                count_prob[row_idx],
                wavelength,
                float(z_qso[row_idx]),
                max_dlas=2,
                min_z=min_z,
                min_peak_distance=min_peak_distance,
                confidence_threshold=confidence_threshold,
                lognhi_min=lognhi_min,
                lognhi_max=lognhi_max,
            )
        )
    return decoded
