# `results/` — reference checkpoints, recipes and score reports

Everything in this directory is a **frozen reference artifact**. `.gitignore` keeps
`*.pt` / `*.json` / `*.csv` / `*.npz` out of the tree globally; this directory is
whitelisted explicitly (`!results/**`).

## `cnn-dual-tower/r48sh_seed51_ema_ep8/` — the CNN dual-tower SOTA (0.6876)

| Path | Content |
|---|---|
| `r38_deployable_seed51.pt` | **Deployable checkpoint** — Stage-A EMA model with the Stage-B refit head applied. This is the 0.6876 model. sha256 `ce2147f6…`. |
| `r38_matched_wls_head_seed51.pt` | The 129-parameter matched-WLS head (`lognhi_delta.weight` / `.bias`). sha256 `5e2fa5d0…`. |
| `r38_matched_wls_seed51.json` | Full Stage-B record: input checkpoint md5s, decoder config, head-only-trainable audit, library versions, split md5s, WLS conditioning. |
| `r38_catalogs_seed51.npz` | VAL prediction catalogue for this candidate. |
| `ema_ep8.pt` | Raw Stage-A checkpoint (seed 51, EMA epoch 8), md5 `889e42c20ee1be032fad4a084ae3f195`. The input to Stage B. |
| `ema_history.json`, `ema_run_info.json` | EMA trace and run metadata. |

Headline: **VAL Final 0.687569327685616** (Detection 0.6712027088378448 /
Parameter 0.7121192559572727). Selected as the maximum over 64 candidates
(8 seeds × 8 EMA endpoints) on the unified VAL split — a *selected maximum*, not an
unbiased generalisation estimate. Full leaderboard: `../reports/r48sh_leaderboard.csv`.

## `cnn-dual-tower/reference_plain_fusion/` — same architecture, no EMA + WLS

| Path | Content |
|---|---|
| `best_model.pt` | `fusion_sig15_l40`: GrowNet + FlatNet frozen towers with a residual fusion head, plain 40-epoch training, seed 42. **VAL Final 0.6584** (Det 0.6255 / Param 0.7078). |
| `training_args.json`, `history.json` | Exact training arguments and per-epoch trace. |

Kept as the direct predecessor: same fusion architecture, before the EMA training and
the matched-WLS readout that take it to 0.6876. It is the subject of
`../reports/cnn_sota_report_guqlf_val.txt`.

## `transformer/tf_sig15_l40_s43/` — the Transformer SOTA (0.6503)

| Path | Content |
|---|---|
| `best_model.pt` | `transformer_conv_stem`, 2,502,919 params, 40 epochs, seed 43. **VAL Final 0.6503** (Det 0.6042 / Param 0.7195, compl 0.5084, pur 0.7497). |
| `training_args.json`, `history.json` | Exact training arguments and per-epoch trace. |

## `reports/`

| Path | Content |
|---|---|
| `final_method_manifest.json` (+ `.sha256`) | The frozen method specification: towers, fusion head, loss weights, optimiser, EMA, Stage-B readout, decoder, data md5s, source hashes, and the actions forbidden after the TEST pass. |
| `final_test_summary.csv`, `final_test_results.json` | Per-seed TEST results for the frozen method (seeds 42–45). |
| `r48sh_leaderboard.csv`, `r48sh_summary.json`, `r48sh_report.md` | R48-SH: all 64 candidates ranked by unified VAL Final, plus the aggregate summary and the round report. |
| `cnn_sota_report_guqlf_val.txt` | Official score report for `fusion_sig15_l40` (VAL). |
| `tf_sota_report_guqlf_val.txt` | Official score report for `tf_sig15_l40_s43` (VAL). |

### Headline TEST result (frozen method — the only method opened on TEST)

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
