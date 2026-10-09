from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import LOGNHI_BINS, LYMAN_ALPHA_A, SNR_BINS
from .decode import velocity_delta_kms


@dataclass(frozen=True)
class DLARecord:
    spectrum_index: int
    z_dla: float
    lognhi: float
    snr: float
    slot: int
    confidence: float = 1.0


def _in_wavelength_band(z_dla: float, wavelength_band: tuple[float, float] | None) -> bool:
    if wavelength_band is None:
        return True
    obs_wavelength = LYMAN_ALPHA_A * (1.0 + float(z_dla))
    return float(wavelength_band[0]) <= obs_wavelength < float(wavelength_band[1])


def truth_records_from_labels(
    labels: pd.DataFrame,
    indices: Iterable[int],
    lognhi_min: float = 20.3,
    wavelength_band: tuple[float, float] | None = None,
) -> list[DLARecord]:
    records: list[DLARecord] = []
    labels_reset = labels.reset_index(drop=True)
    for idx in indices:
        row = labels_reset.iloc[int(idx)]
        snr = float(row["SNR_GU"]) if "SNR_GU" in labels_reset.columns else float("nan")
        n_dla = int(np.clip(row["N_DLA"], 0, 2))
        for slot in range(n_dla):
            z_dla = float(row[f"Z_DLA{slot + 1}"])
            lognhi = float(row[f"LOGNHI{slot + 1}"])
            if not np.isfinite(z_dla) or not np.isfinite(lognhi):
                continue
            if lognhi < lognhi_min:
                continue
            if not _in_wavelength_band(z_dla, wavelength_band):
                continue
            records.append(DLARecord(int(idx), z_dla, lognhi, snr, slot + 1, 1.0))
    return records


def prediction_records_from_rows(
    predictions: pd.DataFrame,
    lognhi_floor: float = 19.0,
    wavelength_band: tuple[float, float] | None = None,
) -> list[DLARecord]:
    records: list[DLARecord] = []
    for _, row in predictions.iterrows():
        idx = int(row["index"])
        snr = float(row["SNR_GU"]) if "SNR_GU" in predictions.columns else float("nan")
        for slot in (1, 2):
            z_dla = float(row[f"Z_DLA{slot}"])
            lognhi = float(row[f"LOGNHI{slot}"])
            conf = float(row.get(f"CONF{slot}", 1.0))
            if not np.isfinite(z_dla) or z_dla < 0:
                continue
            if not np.isfinite(lognhi) or lognhi < lognhi_floor:
                continue
            if not _in_wavelength_band(z_dla, wavelength_band):
                continue
            records.append(DLARecord(idx, z_dla, lognhi, snr, slot, conf))
    return records


def match_records(
    truth: list[DLARecord],
    pred: list[DLARecord],
    match_kms: float = 600.0,
) -> list[tuple[int, int, float]]:
    pairs = []
    for ti, true_item in enumerate(truth):
        for pi, pred_item in enumerate(pred):
            if true_item.spectrum_index != pred_item.spectrum_index:
                continue
            delta = velocity_delta_kms(pred_item.z_dla, true_item.z_dla)
            if delta < match_kms:
                pairs.append((ti, pi, delta))

    pairs = sorted(pairs, key=lambda item: item[2])
    used_truth: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for ti, pi, delta in pairs:
        if ti in used_truth or pi in used_pred:
            continue
        used_truth.add(ti)
        used_pred.add(pi)
        matches.append((ti, pi, delta))
    return matches


def _bin_label(edges: list[float], idx: int) -> str:
    lo = edges[idx]
    hi = edges[idx + 1]
    if np.isinf(hi):
        return f"[{lo:g},inf)"
    return f"[{lo:g},{hi:g})"


def _find_bin(value: float, edges: list[float]) -> int | None:
    if not np.isfinite(value):
        return None
    for idx in range(len(edges) - 1):
        if edges[idx] <= value < edges[idx + 1]:
            return idx
    return None


def _signed_velocity_error(pred_item: DLARecord, true_item: DLARecord) -> float:
    return 299792.458 * (float(pred_item.z_dla) - float(true_item.z_dla)) / (1.0 + float(true_item.z_dla))


def evaluate_detections(
    truth: list[DLARecord],
    pred: list[DLARecord],
    match_kms: float = 600.0,
    snr_bins: list[float] | None = None,
    lognhi_bins: list[float] | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    snr_bins = snr_bins or SNR_BINS
    lognhi_bins = lognhi_bins or LOGNHI_BINS
    matches = match_records(truth, pred, match_kms=match_kms)
    matched_truth = {ti for ti, _, _ in matches}
    matched_pred = {pi for _, pi, _ in matches}
    signed_dv_by_match = {
        (ti, pi): _signed_velocity_error(pred[pi], truth[ti])
        for ti, pi, _ in matches
    }
    dlog_by_match = {
        (ti, pi): float(pred[pi].lognhi - truth[ti].lognhi)
        for ti, pi, _ in matches
    }

    rows = []
    weighted_f1_sum = 0.0
    truth_weight_sum = 0
    for snr_idx in range(len(snr_bins) - 1):
        for log_idx in range(len(lognhi_bins) - 1):
            true_in_bin = [
                i
                for i, item in enumerate(truth)
                if _find_bin(item.snr, snr_bins) == snr_idx and _find_bin(item.lognhi, lognhi_bins) == log_idx
            ]
            pred_in_bin = [
                i
                for i, item in enumerate(pred)
                if _find_bin(item.snr, snr_bins) == snr_idx and _find_bin(item.lognhi, lognhi_bins) == log_idx
            ]
            n_truth = len(true_in_bin)
            n_pred = len(pred_in_bin)
            truth_set = set(true_in_bin)
            pred_set = set(pred_in_bin)
            n_match_truth_bin = sum(1 for i in true_in_bin if i in matched_truth)
            n_match_pred_bin = sum(1 for i in pred_in_bin if i in matched_pred)
            completeness = n_match_truth_bin / n_truth if n_truth else 0.0
            purity = n_match_pred_bin / n_pred if n_pred else 0.0
            f1 = 2.0 * completeness * purity / (completeness + purity) if completeness + purity > 0 else 0.0
            if n_truth:
                weighted_f1_sum += f1 * n_truth
                truth_weight_sum += n_truth
            true_bin_matches = [(ti, pi) for ti, pi, _ in matches if ti in truth_set]
            bin_dv = np.asarray([signed_dv_by_match[(ti, pi)] for ti, pi in true_bin_matches], dtype=np.float32)
            bin_dlog = np.asarray([dlog_by_match[(ti, pi)] for ti, pi in true_bin_matches], dtype=np.float32)
            rows.append(
                {
                    "SNR bin": _bin_label(snr_bins, snr_idx),
                    "LOGNHI bin": _bin_label(lognhi_bins, log_idx),
                    "N_truth": n_truth,
                    "N_pred": n_pred,
                    "N_match_T": n_match_truth_bin,
                    "N_match_P": n_match_pred_bin,
                    "purity": purity,
                    "complete": completeness,
                    "f1": f1,
                    "mean_dv": float(np.mean(bin_dv)) if len(bin_dv) else 0.0,
                    "std_dv": float(np.std(bin_dv)) if len(bin_dv) else 0.0,
                    "mean_dlog": float(np.mean(bin_dlog)) if len(bin_dlog) else 0.0,
                    "std_dlog": float(np.std(bin_dlog)) if len(bin_dlog) else 0.0,
                }
            )

    abs_velocity_errors = np.asarray([delta for _, _, delta in matches], dtype=np.float32)
    signed_velocity_errors = np.asarray(
        [signed_dv_by_match[(ti, pi)] for ti, pi, _ in matches],
        dtype=np.float32,
    )
    lognhi_errors = np.asarray([dlog_by_match[(ti, pi)] for ti, pi, _ in matches], dtype=np.float32)
    z_bias = float(np.mean(signed_velocity_errors)) if len(signed_velocity_errors) else float("nan")
    z_scatter = float(np.std(signed_velocity_errors)) if len(signed_velocity_errors) else float("nan")
    lognhi_bias = float(np.mean(lognhi_errors)) if len(lognhi_errors) else float("nan")
    lognhi_scatter = float(np.std(lognhi_errors)) if len(lognhi_errors) else float("nan")
    score_z = float(np.exp(-z_scatter / 300.0) * np.exp(-abs(z_bias) / 150.0)) if len(matches) else 0.0
    score_nhi = (
        float(np.exp(-lognhi_scatter / 0.25) * np.exp(-abs(lognhi_bias) / 0.1))
        if len(matches)
        else 0.0
    )
    parameter_score = 0.5 * (score_z + score_nhi)
    detection_score = float(weighted_f1_sum / truth_weight_sum) if truth_weight_sum else 0.0
    final_score = 0.6 * detection_score + 0.4 * parameter_score
    summary = {
        "n_truth": int(len(truth)),
        "n_pred": int(len(pred)),
        "n_match": int(len(matches)),
        "overall_completeness": float(len(matches) / len(truth)) if truth else 0.0,
        "overall_purity": float(len(matches) / len(pred)) if pred else 0.0,
        "detection_score": detection_score,
        "detection_score_weighted_bin_f1": detection_score,
        "mean_abs_velocity_error_kms": float(np.mean(abs_velocity_errors)) if len(abs_velocity_errors) else float("nan"),
        "mean_dv": z_bias,
        "std_dv": z_scatter,
        "z_velocity_bias_kms": z_bias,
        "z_velocity_scatter_kms": z_scatter,
        "mean_dlognhi": lognhi_bias,
        "std_dlognhi": lognhi_scatter,
        "lognhi_bias": lognhi_bias,
        "lognhi_scatter": lognhi_scatter,
        "score_z": score_z,
        "score_nhi": score_nhi,
        "parameter_score": parameter_score,
        "final_score": final_score,
        "local_score_approx": final_score,
    }
    return summary, pd.DataFrame(rows)


def format_csst_report(
    summary: dict[str, float],
    bin_table: pd.DataFrame,
    truth_fits: str,
    pred_csv: str,
    snr_field: str,
    wavelength_band: tuple[float, float] | None,
    min_lognhi: float,
    model_name: str | None = None,
    pred_lognhi_floor: float | None = None,
) -> str:
    band_text = (
        f"[{wavelength_band[0]:g},{wavelength_band[1]:g})"
        if wavelength_band is not None
        else "all"
    )
    lines = [
        f"truth_fits: {truth_fits}",
        f"snr_field: {snr_field}",
        f"wavelength_band_A: {band_text}",
        f"min_lognhi: {min_lognhi:g}",
    ]
    if pred_lognhi_floor is not None:
        lines.append(f"pred_lognhi_floor_for_diagnostics: {pred_lognhi_floor:g}")
    lines.extend(
        [
            f"total counted DLAs: {int(summary['n_truth'])}",
            f"pred_csv: {pred_csv}",
            f"raw predicted DLAs after filters: {int(summary['n_pred'])}",
            f"binned predicted DLAs: {int(summary['n_pred'])}",
            f"raw matched DLAs: {int(summary['n_match'])}",
            f"overall purity: {summary['overall_purity']:.6f}",
            f"overall completeness: {summary['overall_completeness']:.6f}",
            "N_match_T is binned by true LOGNHI/SNR and is used for completeness.",
            "N_match_P is binned by predicted LOGNHI/SNR and is used for purity.",
        ]
    )
    score_header = "score using local CSST-style scorer"
    if model_name:
        score_header += f" (model={model_name})"
    lines.extend(
        [
            f"{score_header}:",
            f"  final_score: {summary['final_score']:.6f}",
            f"  detection_score: {summary['detection_score']:.6f}",
            f"  parameter_score: {summary['parameter_score']:.6f}",
            f"  mean_dv/std_dv: {summary['mean_dv']:.3f} / {summary['std_dv']:.3f}",
            f"  mean_dlognhi/std_dlognhi: {summary['mean_dlognhi']:.4f} / {summary['std_dlognhi']:.4f}",
        ]
    )
    if summary["n_pred"] < 0.1 * max(summary["n_truth"], 1):
        lines.append("WARNING: raw predicted DLAs are less than 10% of truth; completeness is likely the main issue.")

    table_columns = [
        "SNR bin",
        "LOGNHI bin",
        "N_truth",
        "N_pred",
        "N_match_T",
        "N_match_P",
        "purity",
        "complete",
        "mean_dv",
        "std_dv",
        "mean_dlog",
        "std_dlog",
    ]
    table = bin_table[table_columns].copy()
    lines.append(table.to_string(index=False, float_format=lambda value: f"{value:8.4f}"))
    return "\n".join(lines) + "\n"
