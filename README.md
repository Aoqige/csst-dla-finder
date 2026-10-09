# CSST DLA Model Zoo

This repository is a lightweight model-zoo workspace for CSST DLA detection
experiments. Each network can live on its own branch, while shared data format
and evaluation conventions stay documented on `main`.

## Branch Layout

- `main`: repository overview, contribution rules, and shared conventions.
- `network/hybrid`: the hybrid ensemble implementation.
- `network/<name>`: proposed branch naming pattern for other submitted models.

For a new model, branch from `main`:

```bash
git checkout main
git checkout -b network/your-model-name
```

Keep model-specific code self-contained and document the expected train,
evaluate, and predict commands in that branch's README.

## Train and evaluate `network/hrh_update`

Run these commands from the repository root after installing `requirements.txt`.
Define the data and output locations once; set `MIN_Z_DLA=1.55` for the
original higher-redshift range.

```bash
export TRAIN_FITS=/absolute/path/to/train.fits
export TARGETS=outputs/cnn_targets_seed42.npz
export RUN_ROOT=hybrid_ensemble/runs
export MIN_Z_DLA=1.10
```

Train one `all`-channel member:

```bash
PYTHONPATH=src python3 hybrid_ensemble/train_hybrid.py \
  --targets "$TARGETS" \
  --train-fits "$TRAIN_FITS" \
  --out-dir "$RUN_ROOT/member_all_seed42" \
  --input-mode all \
  --hidden 96 \
  --num-blocks 4 \
  --epochs 25 \
  --batch-size 128 \
  --lr 1e-3 \
  --seed 42 \
  --threshold 0.40 \
  --min-z-dla "$MIN_Z_DLA" \
  --count-loss-weight 0.35 \
  --region-loss-weight 0.2 \
  --lognhi-loss-weight 0.05 \
  --offset-loss-weight 0.05 \
  --device auto
```

Evaluate the standard three-member ensemble on the validation split recorded in
`$TARGETS`:

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

The complete three-member training commands and output details are in
[`hybrid_ensemble/README.md`](hybrid_ensemble/README.md).

## Data Policy

Do not commit challenge data, model checkpoints, generated predictions, or run
outputs. The `.gitignore` excludes common large artifacts such as FITS files,
PyTorch checkpoints, and output folders.

