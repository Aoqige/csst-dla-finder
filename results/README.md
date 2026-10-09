# `results/` — reference checkpoints, recipes and score reports

Everything in this directory is a **frozen reference artifact**. `.gitignore` keeps
`*.pt` / `*.json` / `*.csv` out of the tree globally; this directory is whitelisted
explicitly (`!results/**`).

## `cnn-dual-tower/`

| Path | Content |
|---|---|
| `fusion_sig15_l40/best_model.pt` | CNN dual-tower SOTA checkpoint (`residual_dilated`, w128 d3, 40 epochs, seed 42). **VAL Final 0.6584** (Det 0.6255 / Param 0.7078, compl 0.5047, pur 0.8287). |
| `fusion_sig15_l40/training_args.json` | Exact training arguments for the above. |
| `fusion_sig15_l40/history.json` | Per-epoch metric trace. |
| `r48sh_seed51_ema_ep8.pt` | Current best single model on unified VAL. **Final 0.687569327685616** (Det 0.6712027088 / Param 0.7121192560), seed 51, EMA epoch 8, matched-WLS readout. |
| `r48sh_seed51_ema_history.json` | EMA trace for seed 51. |

## `transformer/`

| Path | Content |
|---|---|
| `tf_sig15_l40_s43/best_model.pt` | Transformer single-tower SOTA (`transformer_conv_stem`, 2,502,919 params, 40 epochs, seed 43). **VAL Final 0.6503** (Det 0.6042 / Param 0.7195, compl 0.5084, pur 0.7497). |
| `tf_sig15_l40_s43/training_args.json` | Exact training arguments for the above. |
| `tf_sig15_l40_s43/history.json` | Per-epoch metric trace. |

## `reports/`

| Path | Content |
|---|---|
| `final_method_manifest.json` (+ `.sha256`) | The frozen method specification: towers, fusion head, loss weights, optimiser, EMA, Stage-B readout, decoder, data md5s, source hashes, and the list of actions forbidden after the TEST pass. |
| `final_test_summary.csv` | Per-seed TEST results for the frozen method (seeds 42–45). |
| `final_test_results.json` | Full per-seed TEST detail, including `score_z` / `score_nhi` and bias/MAE/RMSE. |
| `r48sh_leaderboard.csv` | R48-SH: all 64 candidates (8 seeds × 8 EMA endpoints) ranked by unified VAL Final. |
| `r48sh_summary.json` | R48-SH aggregate summary. |

### Headline TEST result (frozen method)

| | Final | Detection | Parameter |
|---|---|---|---|
| mean over seeds 42–45 | **0.6696572892** | 0.6334 | 0.7241 |

Per-seed TEST Final: 42 → 0.6711227583, 43 → 0.6663102684, 44 → 0.6709186605,
45 → 0.6702774694. The `unified_val_Final` column in `final_test_summary.csv` is the
VAL counterpart and is *not* the TEST number.

## Read the checkpoints correctly

Each `.pt` holds `{model_state, config, threshold, score, training_config}`. Load with
`ck["model_state"]`; do **not** pass the whole dict with `strict=False`, which silently
leaves random initialisation behind. Assert `missing_keys` is empty after loading.
