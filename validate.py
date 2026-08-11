from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .fits_io import load_test_fits, load_train_fits, load_truth_fits
from .inference import (
    load_model_from_checkpoint,
    make_test_dataset_from_config,
    make_train_dataset_from_config,
    predict_rows,
)
from .metrics import (
    evaluate_detections,
    format_csst_report,
    prediction_records_from_rows,
    truth_records_from_labels,
)
from .train import make_split
from .utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a DLA detector on test.fits + test_truth.fits.")
    parser.add_argument("--test_fits", default="/Users/cham/project/Astro/test.fits")
    parser.add_argument("--truth_fits", default="/Users/cham/project/Astro/test_truth.fits")
    parser.add_argument("--train_fits", default="/Users/cham/project/Astro/train.fits")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_z", type=float, default=None)
    parser.add_argument("--min_peak_distance", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--lognhi_min", type=float, default=19.0)
    parser.add_argument("--lognhi_max", type=float, default=23.0)
    parser.add_argument("--truth_lognhi_min", type=float, default=20.3)
    parser.add_argument("--pred_lognhi_floor", type=float, default=20.3)
    parser.add_argument("--diagnostic_pred_lognhi_floor", type=float, default=19.0)
    parser.add_argument("--match_kms", type=float, default=600.0)
    parser.add_argument("--wavelength_min", type=float, default=2550.0)
    parser.add_argument("--wavelength_max", type=float, default=4200.0)
    parser.add_argument("--use_voigt_refine", action="store_true")
    parser.add_argument(
        "--use_train_split",
        action="store_true",
        help="Legacy mode: evaluate on the saved validation split from train.fits.",
    )
    return parser.parse_args()


def load_validation_indices(checkpoint_path: Path, config: dict, labels_n_dla: np.ndarray) -> np.ndarray:
    split_path = checkpoint_path.parent / "splits.npz"
    if split_path.exists():
        return np.load(split_path)["val_idx"].astype(np.int64)
    _train_idx, val_idx = make_split(
        labels_n_dla,
        float(config.get("val_size", 0.15)),
        int(config.get("split_seed", 42)),
        config.get("max_samples"),
    )
    return val_idx


def align_truth_to_meta(meta: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    meta = meta.reset_index(drop=True)
    truth = truth.reset_index(drop=True)
    if "TARGETID" in meta.columns and "TARGETID" in truth.columns:
        if truth["TARGETID"].duplicated().any():
            raise ValueError("truth_fits contains duplicated TARGETID values.")
        truth_by_id = truth.set_index("TARGETID", drop=False)
        target_ids = meta["TARGETID"].to_numpy()
        missing = [int(value) for value in target_ids if value not in truth_by_id.index]
        if missing:
            preview = ", ".join(str(value) for value in missing[:5])
            raise KeyError(f"truth_fits is missing {len(missing)} TARGETID values from test_fits, e.g. {preview}")
        return truth_by_id.loc[target_ids].reset_index(drop=True)
    if len(meta) != len(truth):
        raise ValueError(f"test_fits rows ({len(meta)}) != truth_fits rows ({len(truth)}) and no TARGETID join is available.")
    return truth


def attach_truth_snr_to_meta(meta: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    merged = meta.reset_index(drop=True).copy()
    truth = truth.reset_index(drop=True)
    for col in ("SNR_GU", "SNR_GV", "SNR_GI"):
        if col in truth.columns:
            merged[col] = truth[col].to_numpy()
    return merged


def write_metric_bundle(
    output_dir: Path,
    rows,
    labels: pd.DataFrame,
    indices: np.ndarray,
    truth_fits: str,
    pred_csv: str,
    checkpoint_name: str,
    wavelength_band: tuple[float, float],
    truth_lognhi_min: float,
    pred_lognhi_floor: float,
    match_kms: float,
    stem: str,
) -> str:
    truth_records = truth_records_from_labels(
        labels,
        indices,
        lognhi_min=truth_lognhi_min,
        wavelength_band=wavelength_band,
    )
    pred_records = prediction_records_from_rows(
        rows,
        lognhi_floor=pred_lognhi_floor,
        wavelength_band=wavelength_band,
    )
    summary, bin_table = evaluate_detections(truth_records, pred_records, match_kms=match_kms)
    summary_path = output_dir / f"{stem}_summary.json"
    bin_path = output_dir / f"{stem}_bin_metrics.csv"
    report_path = output_dir / f"{stem}_report_csst.txt"
    bin_table.to_csv(bin_path, index=False)
    save_json(summary, summary_path)
    report = format_csst_report(
        summary,
        bin_table,
        truth_fits=truth_fits,
        pred_csv=pred_csv,
        snr_field="SNR_GU",
        wavelength_band=wavelength_band,
        min_lognhi=truth_lognhi_min,
        model_name=checkpoint_name,
        pred_lognhi_floor=pred_lognhi_floor,
    )
    report_path.write_text(report, encoding="utf-8")
    if stem == "test":
        (output_dir / "summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        bin_table.to_csv(output_dir / "bin_metrics.csv", index=False)
        (output_dir / "report_test_csst.txt").write_text(report, encoding="utf-8")
    elif stem == "val":
        (output_dir / "summary.json").write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
        bin_table.to_csv(output_dir / "bin_metrics.csv", index=False)
        (output_dir / "report_val_csst.txt").write_text(report, encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    default_subdir = "validation_train_split" if args.use_train_split else "test_evaluation"
    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / default_subdir)
    device = get_device()
    model, config = load_model_from_checkpoint(checkpoint_path, device)

    if args.use_train_split:
        print(f"Loading train FITS validation split: {args.train_fits}")
        wavelength, flux, flux_clean, labels = load_train_fits(args.train_fits)
        n_dla = np.clip(labels["N_DLA"].to_numpy(dtype=np.int64), 0, 2)
        eval_idx = load_validation_indices(checkpoint_path, config, n_dla)
        dataset = make_train_dataset_from_config(wavelength, flux, flux_clean, labels, config, eval_idx)
        meta_for_prediction = labels
        id_column = None
        truth_source = args.train_fits
        report_stem = "val"
    else:
        print(f"Loading test FITS: {args.test_fits}")
        wavelength, flux, flux_clean, meta = load_test_fits(args.test_fits)
        print(f"Loading test truth FITS: {args.truth_fits}")
        truth_wavelength, truth = load_truth_fits(args.truth_fits)
        if len(wavelength) != len(truth_wavelength) or not np.allclose(wavelength, truth_wavelength, equal_nan=True):
            raise ValueError("test_fits and truth_fits WAVELENGTH grids do not match.")
        labels = align_truth_to_meta(meta, truth)
        meta_for_prediction = attach_truth_snr_to_meta(meta, labels)
        eval_idx = np.arange(len(labels), dtype=np.int64)
        dataset = make_test_dataset_from_config(flux, flux_clean, meta_for_prediction, config)
        id_column = "TARGETID" if "TARGETID" in meta_for_prediction.columns else None
        truth_source = args.truth_fits
        report_stem = "test"

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows = predict_rows(
        model,
        loader,
        wavelength,
        flux,
        flux_clean,
        meta_for_prediction,
        device,
        id_column=id_column,
        min_z=args.min_z,
        min_peak_distance=args.min_peak_distance,
        confidence_threshold=args.confidence_threshold,
        lognhi_min=args.lognhi_min,
        lognhi_max=args.lognhi_max,
        use_voigt_refine=args.use_voigt_refine,
    )
    pred_csv = output_dir / "predictions.csv"
    rows.to_csv(pred_csv, index=False)

    wavelength_band = (args.wavelength_min, args.wavelength_max)
    report = write_metric_bundle(
        output_dir=output_dir,
        rows=rows,
        labels=labels,
        indices=eval_idx,
        truth_fits=truth_source,
        pred_csv=str(pred_csv),
        checkpoint_name=checkpoint_path.parent.name,
        wavelength_band=wavelength_band,
        truth_lognhi_min=args.truth_lognhi_min,
        pred_lognhi_floor=args.pred_lognhi_floor,
        match_kms=args.match_kms,
        stem=report_stem,
    )
    print(report)

    if args.diagnostic_pred_lognhi_floor is not None and args.diagnostic_pred_lognhi_floor < args.pred_lognhi_floor:
        diagnostic_stem = f"{report_stem}_diagnostic_lognhi{args.diagnostic_pred_lognhi_floor:g}".replace(".", "p")
        diagnostic_report = write_metric_bundle(
            output_dir=output_dir,
            rows=rows,
            labels=labels,
            indices=eval_idx,
            truth_fits=truth_source,
            pred_csv=str(pred_csv),
            checkpoint_name=checkpoint_path.parent.name,
            wavelength_band=wavelength_band,
            truth_lognhi_min=args.truth_lognhi_min,
            pred_lognhi_floor=args.diagnostic_pred_lognhi_floor,
            match_kms=args.match_kms,
            stem=diagnostic_stem,
        )
        print("Diagnostic low-LOGNHI report:")
        print(diagnostic_report)

    print(f"Saved validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
