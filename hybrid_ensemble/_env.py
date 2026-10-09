"""Environment-overridable paths shared by the analysis scripts.

This repository carries no machine-specific absolute paths. Every external path
the tooling needs is resolved here and can be overridden through the
environment:

===========================  ==========================  ==============================
Variable                     Attribute                   Default
===========================  ==========================  ==============================
``CSST_PYTHON``              ``PYTHON``                  ``sys.executable``
``CSST_DLA_RUNS``            ``RUNS``                    ``~/csst_dla_runs``
``CSST_TEST_FITS``           ``TEST_FITS``               ``<repo>/data/test.fits``
``CSST_TEST_TRUTH``          ``TEST_TRUTH``              ``<repo>/data/test_truth.fits``
``CSST_TRAIN_FITS``          ``TRAIN_FITS``              ``<repo>/data/train.fits``
===========================  ==========================  ==============================

Scripts in this directory import it directly (``import _env``), which works
because Python puts the running script's directory first on ``sys.path``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
HYBRID = ROOT / "hybrid_ensemble"

PYTHON = os.environ.get("CSST_PYTHON", sys.executable)
RUNS = Path(os.environ.get("CSST_DLA_RUNS", str(Path.home() / "csst_dla_runs")))

_data = ROOT / "data"
TEST_FITS = os.environ.get("CSST_TEST_FITS", str(_data / "test.fits"))
TEST_TRUTH = os.environ.get("CSST_TEST_TRUTH", str(_data / "test_truth.fits"))
TRAIN_FITS = os.environ.get("CSST_TRAIN_FITS", str(_data / "train.fits"))
