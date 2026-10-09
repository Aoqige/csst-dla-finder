# CSST DLA Model Zoo — SOTA branch (`sota/hrh_final`)

This branch consolidates the **two current state-of-the-art systems** of the CSST DLA
finder into one self-contained tree. It is cut from `main` and carries only the core
code, the reference checkpoints, and the drivers that produced the headline numbers.
The historical exploration branches are archived and closed; the ledger is in
[`docs/BRANCH_LEDGER.md`](docs/BRANCH_LEDGER.md).

## The two SOTA systems

| System | Module | Structure | Split | Final |
|---|---|---|---|---|
| **CNN dual-tower** | `hybrid_ensemble/feature_fusion.py` (`DualTowerFusionNet`) | GrowNet dilated CNN (1.99 M) + FlatNet WZX CNN (1.02 M), both frozen, joined by a trained residual fusion head (w128 × d3) | VAL | **0.687569327685616** |
| **Transformer single tower** | `hybrid_ensemble/models/transformer_conv_stem_5head.py` | 3-layer Conv1d stem → 4 × TransformerEncoderLayer (d_model 192, 8 heads, dim_ff 768), five heads, 2.50 M | VAL | **0.6503** |

Both share the same five-head output contract `{heatmap, lognhi, count_logits, offset}`
and therefore the same `decode.py`, `evaluate_hybrid.py` and scoring path.

### About the CNN dual-tower number

`0.687569327685616` is the **full current-line recipe**, not a bare checkpoint:
Stage-A EMA training (R48-SH, seed 51, EMA epoch 8) followed by the Stage-B
matched-WLS head readout. The ready-to-infer artifacts live in
`results/cnn-dual-tower/r48sh_seed51_ema_ep8/`:

- `r38_deployable_seed51.pt` — the deployable checkpoint (E + refit head).
- `r38_matched_wls_head_seed51.pt` — the 129-parameter WLS head.
- `r38_matched_wls_seed51.json` — the full recipe and audit record.
- `ema_ep8.pt` — the raw Stage-A checkpoint (md5 `889e42c2…`), the input to Stage B.

Two caveats that must travel with this number:

1. It is a **repeated-development selection over 64 candidates on the unified VAL
   split** (8 seeds × 8 EMA endpoints, max selected), so it is a selected maximum,
   **not** an unbiased generalisation estimate.
2. The gain over the previous best comes **97.4 % from the Parameter term and
   entirely from `score_nhi`**; `n_pred` / `n_match` / precision / recall are
   bit-identical to the pre-readout model.

### The only TEST number

The single method that has been opened on TEST is the frozen R38/R39 matched-WLS
readout: **TEST Final 0.6696572892 ± 0.0019574108** (Detection 0.6334 /
Parameter 0.7241, seeds 42–45). Its specification and per-seed results are in
`results/reports/`.

### Data lines

The current standard is the **`GU_qlf` line** (`train_500k_GU_qlf.fits`, 194 px,
2554–4098 Å). Scores from different data lines are **not comparable** — the truth
catalogue density differs, which changes the `n_truth`-weighted denominator of the
Detection term. The older `20260903` line (681 px) survives only inside the archived
`network/*` branches; its numbers are recorded in `docs/BRANCH_LEDGER.md`.

## Layout

```
hybrid_ensemble/          core code (see hybrid_ensemble/README.md)
  data.py decode.py evaluate_hybrid.py model.py score_test.py tune_decode.py
  _env.py                 environment-overridable paths for the tooling
  feature_fusion.py fuse_wzx_dilated.py
  train_feature_fusion.py predict_feature_fusion.py train_hybrid.py predict_hybrid.py
  train_transformer.py train_transformer_baseline.py
  models/  dilated_resnet_5head.py + 4 transformer variants
  analyze_* bench_tf_vs_cnn diagnose_* frontier_* sweep_* plot_bin_heatmaps ...
src/csst_dla/             scoring / SNR / targets / FITS IO
scripts/                  split and dense-target builders
sota/verify.py            standalone self-check (see below)
sota/drivers/             the drivers that produced the headline runs
results/                  reference checkpoints, recipes and score reports
vendor/csst_dla_wzx_pkg/  WZX tower package, bundled so the fusion model rebuilds
docs/BRANCH_LEDGER.md     archived-branch ledger
```

## Verify this branch

```bash
python3 sota/verify.py
```

Runs against the repository alone — no external paths, no challenge data. It checks
that every module imports, that the committed library carries no machine-specific
absolute paths, that all sources compile, and — for each of the four reference
checkpoints — that it loads, matches its recorded tensor and parameter counts,
**rebuilds from its own `config` block and loads with an exact key match**, and
completes a forward pass emitting the five-head contract.

Adding the challenge data enables the numeric check:

```bash
python3 sota/verify.py \
  --targets   <cnn_targets_unified_seed42_sig15.npz> \
  --train-fits <train_500k_GU_qlf.fits>
```

which rebuilds the VAL truth catalogue, re-scores the bundled prediction catalogue
with the official scorer, and compares the result to the recorded 0.687569327685616.
Both modes exit non-zero if anything fails.

Two directories are exempt from the absolute-path check on purpose:
`sota/drivers/` (verbatim historical run records) and `vendor/` (third-party sources
kept byte-identical to upstream).

## Dependencies

`requirements.txt` pins the versions the reference runs were produced with
(Python 3.10). The only third-party package that used to live outside the repository
— `csst_dla_wzx_pkg` — is now bundled under `vendor/`.

## Reference checkpoints

| File | System | Params | Final | Split |
|---|---|---|---|---|
| `results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_deployable_seed51.pt` | CNN dual-tower (deployable) | 3,555,527 | 0.6875693277 | VAL |
| `results/cnn-dual-tower/reference_plain_fusion/best_model.pt` | CNN dual-tower, plain training (no EMA + WLS) | 3,555,527 | 0.6584 | VAL |
| `results/transformer/tf_sig15_l40_s43/best_model.pt` | Transformer single tower | 2,502,919 | 0.6503 | VAL |

Each checkpoint holds `{model_state, config, threshold, score, training_config}`.
Read weights from `ck["model_state"]`; passing the whole dict with `strict=False`
silently leaves random initialisation — assert that `missing_keys` is empty.

## Reproduce

```bash
# dense targets + split
PYTHONPATH=src python3 scripts/make_cnn_targets.py --help

# Transformer single tower
PYTHONPATH=src python3 hybrid_ensemble/train_transformer.py \
  --arch transformer_conv_stem --input-mode flux \
  --epochs 40 --lr 1e-3 --lr-schedule cosine --batch-size 512 --seed 43 \
  --lognhi-loss-weight 0.15 --targets <targets.npz> --train-fits <train.fits> \
  --out-dir <run dir>

# CNN dual-tower fusion (both towers frozen)
PYTHONPATH=src python3 hybrid_ensemble/train_feature_fusion.py --help
```

The current-line recipe (EMA training + matched-WLS head readout) is the one in
`sota/drivers/`; the frozen method is specified in
`results/reports/final_method_manifest.json`.

---

## Upstream: CSST DLA Model Zoo

This repository is a lightweight model-zoo workspace for CSST DLA detection
experiments. Each network can live on its own branch, while shared data format
and evaluation conventions stay documented on `main`.

### Data Policy

Do not commit challenge data, model checkpoints, generated predictions, or run
outputs. The `.gitignore` excludes common large artifacts such as FITS files,
PyTorch checkpoints, and output folders. This branch carries a narrow, explicit
whitelist for the reference checkpoints under `results/` only.
