# `sota/` — current-line drivers

These are the drivers that produced the headline runs of the **`GU_qlf` line**.
They are kept here because they are the actual reproduction path: the training loop
used for the current SOTA is *not* `hybrid_ensemble/train_feature_fusion.py`, it is
`run_fusion_ema_lrstep.py` (EMA training) followed by a matched-WLS head readout.

## `drivers/`

| File | Role |
|---|---|
| `ema.py` | EMA wrapper (beta 0.999, updated after every `optimizer.step()`). |
| `run_fusion_ema_lrstep.py` | Stage-A training driver: frozen GrowNet + FlatNet towers, `residual_dilated` fusion head, EMA checkpointing. |
| `r38_matched_wls.py` | Stage-B readout: equal-weight OLS refit of the 129 head parameters on the frozen model's TRAIN-split matched objects (seeds 42–44). |
| `r39_matched_wls.py` | Same as above for seed 45. |
| `aggregate_r38.py`, `aggregate_r39.py` | Per-seed aggregation of the Stage-B outputs. |
| `r48sh_stage_a.py`, `r48sh_stage_b.py`, `r48sh_rank.py` | R48-SH: 8 seeds × 8 EMA endpoints = 64-candidate sweep; `r48sh_rank.py` builds the leaderboard. |
| `make_unified_split.py` | Builds the unified TRAIN/VAL split (400 000 / 100 000, intersection 0). |
| `run_wzx_unified.py` | Trains the WZX (FlatNet) tower on the unified split. |
| `final_eval.py` | The single TEST pass for the frozen method. |

## Recipes

The authoritative machine-readable recipe is
`results/reports/final_method_manifest.json` (with its `.sha256`). In brief:

- **Towers** — GrowNet conservative + FlatNet conservative, both frozen, eval mode.
- **Fusion head** — `merge_mode=residual_dilated`, `fusion_width=128`, `fusion_depth=3`.
- **Loss weights** — region 0.2 / logNHI 0.05 / offset 0.1 / count 0.25, unweighted
  count cross-entropy.
- **Optimiser** — AdamW, lr 5e-4, weight decay 1e-4, batch 512, EMA beta 0.999.
- **Stage B** — `scipy.linalg.lstsq` equal-weight OLS on the frozen model's matched
  TRAIN objects, 129 parameters, minimum-norm delta around `theta_E`.
- **Decoder** — threshold 0.45, min_distance 10, min_z_dla 1.1, count_bias `[0,0,0]`,
  logNHI clip `[20.3, 22.5]`.

## Notes

- The scorer bins by `pred["SNR"]`; the official `SNR_GU` field must be wired in, and
  the frozen float32 products must be used as-is — a float64 re-cast moves the
  logNHI binning and shifts the score.
- `matched` pairs must be keyed on the truth-object index, not `TARGETID`.
- `model.pt` is the last epoch, `best_model.pt` is the best epoch.
