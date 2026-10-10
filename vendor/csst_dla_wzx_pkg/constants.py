from __future__ import annotations

LYMAN_ALPHA_A = 1215.67
C_KMS = 299792.458

SUBMISSION_COLUMNS = ["id", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"]
AUX_SUBMISSION_COLUMNS = ["id", "N_DLA", "CONF1", "CONF2", "Z_DLA1", "LOGNHI1", "Z_DLA2", "LOGNHI2"]

SNR_BINS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, float("inf")]

# Official truth DLA bins start at 20.3, but diagnostic reports keep lower
# predicted-NHI bins to reveal "detected but underestimated LOGNHI" failures.
OFFICIAL_LOGNHI_BINS = [20.3, 20.5, 21.0, 21.5, 22.0, float("inf")]
DIAGNOSTIC_LOGNHI_BINS = [19.0, 19.5, 20.0, 20.3, 20.5, 21.0, 21.5, 22.0, float("inf")]
LOGNHI_BINS = DIAGNOSTIC_LOGNHI_BINS
