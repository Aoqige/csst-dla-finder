from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from .fits_io import load_test_fits
from .inference import load_model_from_checkpoint, make_test_dataset_from_config, predict_rows, submission_frame
from .utils import ensure_dir, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DLA prediction CSV.")
    parser.add_argument("--test_fits", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", default="outputs/submission.csv")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--min_z", type=float, default=None)
    parser.add_argument("--min_peak_distance", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=None)
    parser.add_argument("--lognhi_min", type=float, default=19.0)
    parser.add_argument("--lognhi_max", type=float, default=23.0)
    parser.add_argument("--use_voigt_refine", action="store_true")
    parser.add_argument("--include_aux", action="store_true", help="Include N_DLA and confidence columns for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    ensure_dir(output_csv.parent)
    device = get_device()
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    print(f"Loading test FITS: {args.test_fits}")
    wavelength, flux, flux_clean, meta = load_test_fits(args.test_fits)
    dataset = make_test_dataset_from_config(flux, flux_clean, meta, config)
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
        meta,
        device,
        id_column="TARGETID",
        min_z=args.min_z,
        min_peak_distance=args.min_peak_distance,
        confidence_threshold=args.confidence_threshold,
        lognhi_min=args.lognhi_min,
        lognhi_max=args.lognhi_max,
        use_voigt_refine=args.use_voigt_refine,
    )
    submission = submission_frame(rows, include_aux=args.include_aux)
    submission.to_csv(output_csv, index=False)
    print(f"submission.shape: {submission.shape}")
    print(f"submission.columns: {list(submission.columns)}")
    if "N_DLA" in submission.columns:
        print(submission["N_DLA"].value_counts().sort_index())
    print(f"Saved submission to {output_csv}")


if __name__ == "__main__":
    main()
