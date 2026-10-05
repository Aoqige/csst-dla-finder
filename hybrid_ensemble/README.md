# Dilated ResNet Five-Head Ensemble

This branch keeps the hybrid pipeline's data preparation, decoding, evaluation,
and prediction scripts, while replacing its CNN with the local dilated
five-head residual architecture:

- multi-channel 1D spectrum features
- 1D dilated residual backbone with layer normalization
- heatmap, broad-region, `LOGNHI`, count, and optional offset heads
- weighted ensemble evaluation and prediction
- configurable DLA redshift lower bound via `--min-z-dla`

No data files or trained checkpoints are included.

## Expected FITS Layout

Training FITS:

- `WAVELENGTH`
- `FLUX`
- `FLUX_CLEAN`
- `LABELS`

Test FITS:

- `WAVELENGTH`
- `FLUX`
- `FLUX_CLEAN`
- `META`

The lightweight FITS reader in `src/csst_dla/fits_utils.py` expects the same
fixed challenge table layout used by this project.

## Prepare Targets

Create a split:

```bash
PYTHONPATH=src python3 scripts/make_split.py \
  --train-fits /path/to/train.fits \
  --out splits/split_seed42.npz \
  --seed 42
```

Create dense targets. For lower-resolution spectra, reduce pixel radii so the
physical wavelength width stays comparable.

```bash
PYTHONPATH=src python3 scripts/make_cnn_targets.py \
  --train-fits /path/to/train.fits \
  --split splits/split_seed42.npz \
  --out outputs/cnn_targets_seed42.npz \
  --sigma-pixels 1.0 \
  --low-lognhi-radius-pixels 3 \
  --mid-lognhi-radius-pixels 6 \
  --high-lognhi-radius-pixels 13 \
  --very-high-lognhi-radius-pixels 20
```

## Train Members

For a low-redshift dataset such as `QSO=1.10-2.35`, use `--min-z-dla 1.10`.
For the original higher-redshift range, the default `--min-z-dla 1.55` can be
left unchanged.

Run all commands below from the repository root. Set the data and output paths
once before training:

```bash
export TRAIN_FITS=/absolute/path/to/train.fits
export TARGETS=outputs/cnn_targets_seed42.npz
export RUN_ROOT=hybrid_ensemble/runs
export MIN_Z_DLA=1.10
```

The commands save `best_model.pt` using the best validation score and also
write `model.pt`, `history.json`, and `training_args.json` for each member.

### Member 1: all channels

```bash
PYTHONPATH=src python3 hybrid_ensemble/train_hybrid.py \
  --targets "$TARGETS" \
  --train-fits "$TRAIN_FITS" \
  --out-dir "$RUN_ROOT/member_all_seed42" \
  --input-mode all \
  --hidden 96 \
  --num-blocks 4 \
  --norm-type layer \
  --head-layers 1 \
  --epochs 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 42 \
  --threshold 0.40 \
  --min-z-dla "$MIN_Z_DLA" \
  --truth-min-lognhi 20.3 \
  --count-loss-weight 0.35 \
  --region-loss-weight 0.2 \
  --lognhi-loss-weight 0.05 \
  --offset-loss-weight 0.05 \
  --num-workers 8 \
  --device auto
```

### Member 2: residual channels

```bash
PYTHONPATH=src python3 hybrid_ensemble/train_hybrid.py \
  --targets "$TARGETS" \
  --train-fits "$TRAIN_FITS" \
  --out-dir "$RUN_ROOT/member_residual_seed43" \
  --input-mode residual \
  --hidden 96 \
  --num-blocks 4 \
  --norm-type layer \
  --head-layers 1 \
  --epochs 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 43 \
  --threshold 0.40 \
  --min-z-dla "$MIN_Z_DLA" \
  --truth-min-lognhi 20.3 \
  --count-loss-weight 0.35 \
  --region-loss-weight 0.2 \
  --lognhi-loss-weight 0.05 \
  --offset-loss-weight 0.05 \
  --num-workers 8 \
  --device auto
```

### Member 3: flux channels

```bash
PYTHONPATH=src python3 hybrid_ensemble/train_hybrid.py \
  --targets "$TARGETS" \
  --train-fits "$TRAIN_FITS" \
  --out-dir "$RUN_ROOT/member_flux_seed44" \
  --input-mode flux \
  --hidden 96 \
  --num-blocks 4 \
  --norm-type layer \
  --head-layers 1 \
  --epochs 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 44 \
  --threshold 0.40 \
  --min-z-dla "$MIN_Z_DLA" \
  --truth-min-lognhi 20.3 \
  --count-loss-weight 0.35 \
  --region-loss-weight 0.2 \
  --lognhi-loss-weight 0.05 \
  --offset-loss-weight 0.05 \
  --num-workers 8 \
  --device auto
```

## Evaluate Ensemble

The evaluator reads the `val` split stored in `$TARGETS` and writes both the
validation score report and the configuration consumed by prediction.

```bash
PYTHONPATH=src python3 hybrid_ensemble/evaluate_hybrid.py \
  --models \
    "$RUN_ROOT/member_all_seed42/best_model.pt" \
    "$RUN_ROOT/member_residual_seed43/best_model.pt" \
    "$RUN_ROOT/member_flux_seed44/best_model.pt" \
  --weights 1.0 1.0 0.7 \
  --targets "$TARGETS" \
  --train-fits "$TRAIN_FITS" \
  --batch-size 64 \
  --threshold 0.25 \
  --min-distance 3 \
  --min-z-dla "$MIN_Z_DLA" \
  --truth-min-lognhi 20.3 \
  --lognhi-min 20.3 \
  --lognhi-max 22.5 \
  --count-bias -0.8 1.0 0.4 \
  --count-min-prob 0.0 \
  --soft-radius 1 \
  --soft-power 3.0 \
  --fit-lognhi-calibration \
  --out "$RUN_ROOT/ensemble_eval.json" \
  --device auto
```

The evaluator writes both:

- `$RUN_ROOT/ensemble_eval.json` with the validation score and `bin_details`
- `$RUN_ROOT/ensemble_eval_config.json` for `predict_hybrid.py`

The config file is consumed by prediction.

## Predict

```bash
PYTHONPATH=src python3 hybrid_ensemble/predict_hybrid.py \
  --ensemble-config hybrid_ensemble/runs/ensemble_eval_config.json \
  --test-fits /path/to/test.fits \
  --out hybrid_ensemble/runs/submission_hybrid.csv \
  --device auto
```
