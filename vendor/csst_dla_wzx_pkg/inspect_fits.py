from __future__ import annotations

import argparse
import json

from .fits_io import summarize_train_fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a DLA training FITS file.")
    parser.add_argument("--fits", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(summarize_train_fits(args.fits), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
