# `hybrid_ensemble/` — core code

Shared five-head output contract: every model exports
`{heatmap, lognhi, count_logits, offset}` per pixel, so backbones are swappable into
`decode.py`, `evaluate_hybrid.py` and the offline tensor cache without downstream change.
For first-time operation, start with [the handoff guide](../HANDOFF.md); this page is the code and entry-point map.

## Shared infrastructure

| File | Role |
|---|---|
| `data.py` | Dataset and input-channel construction. |
| `decode.py` | Peak decoding, NMS, count handling, `average_predictions()`. |
| `evaluate_hybrid.py` | Ensemble evaluation harness. |
| `model.py` | Shared model scaffolding / head contract. |
| `score_test.py` | Official scoring entry point. |
| `tune_decode.py` | Decoder parameter sweep. |

## CNN dual-tower (SOTA system 1)

| File | Role |
|---|---|
| `models/dilated_resnet_5head.py` | GrowNet: 1D dilated residual backbone with layer norm, five heads. |
| `feature_fusion.py` | `DualTowerFusionNet`: fuses the dilated tower and the WZX tower (`concat(d_features, w_features)` = 192 ch → `fuse` → 128 ch shared representation → five heads). |
| `fuse_wzx_dilated.py` | WZX + dilated fusion construction. |
| `train_feature_fusion.py` | Feature-level fusion trainer. |
| `predict_feature_fusion.py` | Fusion inference. |
| `train_hybrid.py`, `predict_hybrid.py` | Single-tower / weighted-ensemble train and predict path. |

## Transformer single tower (SOTA system 2)

| File | Role |
|---|---|
| `models/transformer_5head.py` | v0: per-pixel linear projection + learned absolute positions. |
| `models/transformer_conv_stem_5head.py` | v3c: 3-layer Conv1d stem + learned absolute positions (the current TF SOTA). |
| `models/transformer_conv_stem_rope_5head.py` | v5: conv stem + RoPE. |
| `models/transformer_conv_stem_alibi_5head.py` | v6: conv stem + ALiBi. |
| `train_transformer.py` | Transformer trainer (`--arch`, `--input-mode`, cosine LR, count loss type). |
| `train_transformer_baseline.py` | Baseline trainer variant. |

## Evaluation / analysis tooling

`analyze_compl_purity.py`, `analyze_headroom.py`, `analyze_param.py`,
`bench_tf_vs_cnn.py`, `describe_misses.py`, `diagnose_dv.py`,
`diagnose_selection.py`, `eval_long60.py`, `frontier_compl_purity.py`,
`frontier_votes.py`, `line_gate.py`, `merge_predictions.py`, `plot_bin_heatmaps.py`,
`prevalidate_channels.py`, `rerun_v3c_countbias.py`, `smoke_channels.py`,
`sweep_count.py`, `sweep_count2.py`, `sweep_count3.py`, `sweep_decode.py`,
`sweep_threshold.py`, `val_union_baseline.py`, `voigt_limits.py`, `voigt_zrefine.py`.

## Not carried on this branch

The CNN+Transformer (`ct_*`) fusion line and the standalone verifier line were closed
by the experiment ledger and are **not** included here; they remain reachable through
the archived branches recorded in `docs/BRANCH_LEDGER.md`.

## Caveats

- `fuse` contains a GroupNorm over the whole sequence, so any `shared[:, k]` slice must
  come from a full-length (194 px) convolution.
- `decode` uses `n_pick = argmax(count_prob)` as a hard switch: a collapsed count head
  drives `n_pred` to zero regardless of the heatmap.
- `train_feature_fusion.py` hard-codes `num_workers=0`.
- `train_transformer.py` has a cosine LR schedule; `train_feature_fusion.py` does not.
