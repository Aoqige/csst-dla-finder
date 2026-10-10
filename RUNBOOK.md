# RUNBOOK — training, evaluation and test commands

Every command needed to reproduce the two SOTA systems on this branch. Commands are
written against a checkout of this repository; nothing below assumes a particular
machine.

> **Read this first.** The commands run as written once §0 is exported. The only
> things you may need to change are listed in [§0.2 What you may need to change](#02-what-you-may-need-to-change).
> Every flag below was checked against the target script's own argument parser.

---

## 0. Environment

```bash
export REPO=/path/to/csst-dla-finder          # this checkout
export PY=/path/to/python                     # Python 3.10, see requirements.txt
export PYTHONPATH=$REPO/src:$REPO/hybrid_ensemble:$REPO/vendor
```

Install dependencies (the reference runs used the pinned versions in
`requirements.txt`, Python 3.10.16):

```bash
$PY -m pip install -r $REPO/requirements.txt
```

Sanity check the branch before anything else — no data required:

```bash
python3 $REPO/sota/verify.py
```

To prove the branch actually **trains** end to end, run the smoke test. It exercises
every training entry point at 1 epoch / a few hundred samples, then reproduces the
SOTA number through Stage B and runs inference + scoring. Steps whose inputs are
missing are skipped, not failed.

```bash
bash $REPO/sota/smoke_test.sh
```

Last measured: **9 passed, 0 failed, 0 skipped** — including
`R38_Final = 0.687569327685616` — from a clean `git archive` of this branch, with the
external `csst_dla_wzx_pkg` removed from the machine.

### 0.1 Data and paths

The challenge data is **not** committed. Export the paths once:

```bash
export TRAIN_FITS=/data/heruihua/newer/train_500k_GU_qlf.fits      # 500k spectra + LABELS
export TEST_FITS=/data/heruihua/newer/test_100k_GU.fits           # TEST spectra + TRUTH
export TEST_TRUTH=/data/heruihua/newer/test_truth_100k_GU.fits
export RUNS=$HOME/csst_dla_runs                                    # run root
export R12=$RUNS/20261003_r12                                     # unified split + towers
export R14=$RUNS/20261003_r14                                     # WZX tower
export R48=$RUNS/20261008_r48sh                                   # R48-SH run root
```

### 0.2 What you may need to change

| Where | What | Why |
|---|---|---|
| `export REPO`, `export PY` | your paths | obvious |
| `export TRAIN_FITS` / `TEST_FITS` / `TEST_TRUTH` | your data paths | the data is not committed |
| `export RUNS`, `R12`, `R14`, `R48` | your run root | any directory works; the scripts create subdirectories |
| `--device cuda` / `cuda:0` | `cpu` if no GPU | CUDA is assumed throughout |
| `CUDA_VISIBLE_DEVICES=0` | your free GPU | check `nvidia-smi` first |
| `<predictions.csv>`, `<score.json>`, `<member1.pt>` in §4.1/§4.2 | your files | these are the only real placeholders |

Everything else — every hyper-parameter, path suffix and seed — is the value the
reference run used. Nothing needs editing inside the repository.

Those values are not transcribed by hand: each command was diffed against the
arguments the runs themselves recorded — `training_args.json` for the Transformer
and the GrowNet tower, `config.json` for the WZX tower, and the `training_config`
embedded in `ema_ep8.pt` for Stage A. **81 of 81 flags match**, with no omissions.

Three settings are **environment variables, not flags** — easy to miss:

| Setting | Where |
|---|---|
| `FUS_SUBSET=both`, `EMA_BASE=sg_off` | prefix of the Stage-A command (§3.2) |
| `UNIFIED_SPLITS=<splits_unified.npz>` | prefix of the WZX tower command (§3.1) |
| `CUDA_VISIBLE_DEVICES=<gpu>` | Stage A (§3.2) |

Environment overrides read by the drivers and tooling (all optional):

| Variable | Used by | Default |
|---|---|---|
| `CSST_PYTHON` | analysis tooling | `sys.executable` |
| `CSST_DLA_RUNS` | `final_eval.py`, `aggregate_*.py` | `~/csst_dla_runs` |
| `CSST_TRAIN_FITS`, `CSST_TEST_FITS`, `CSST_TEST_TRUTH` | `final_eval.py`, `make_unified_split.py` | `~/data/...` |
| `CSST_UNIFIED_SRC`, `CSST_UNIFIED_OUTDIR` | `make_unified_split.py` | see §1.3 |

### 0.3 ⚠ Two different VAL splits are in play

This matters more than anything else on this page. The reported numbers were **not
all scored on the same validation set**:

| Number | VAL split | `val_idx` md5 | val size |
|---|---|---|---|
| 0.6584 (CNN dual-tower, plain) | "old" split | `dc8f3921f845` | 99,999 |
| 0.6503 (Transformer single tower) | "old" split | `dc8f3921f845` | 99,999 |
| **0.687569327685616** (CNN dual-tower + EMA + WLS) | **unified split** | `69424cc24375` | 100,000 |
| 0.6696572892 (frozen method, **TEST**) | — (TEST) | — | — |

The two val sets overlap in only 19,993 indices (**Jaccard 0.111**) — they are
different random splits of the same 500 000 spectra, so their Final scores are **not
directly comparable**. §2 below evaluates on the old split; §3 on the unified split.

---

## 1. Data preparation

### 1.1 Train/val split

```bash
$PY $REPO/scripts/make_split.py \
  --train-fits $TRAIN_FITS \
  --out $RUNS/splits/split_seed42.npz \
  --val-fraction 0.2 --seed 42
```

### 1.2 Dense targets

Two target recipes exist; they differ **only** in the Gaussian target width, and the
file name records it (`sig15` = σ 1.5 px, `sig10` = σ 1.0 px, no suffix = the σ 3.0
default). The reference runs use **σ 1.5**.

```bash
$PY $REPO/scripts/make_cnn_targets.py \
  --train-fits $TRAIN_FITS \
  --split $RUNS/splits/split_seed42.npz \
  --out $RUNS/common/cnn_targets_seed42_sig15.npz \
  --sigma-pixels 1.5 --min-lognhi 20.3 \
  --low-lognhi-radius-pixels 10 --mid-lognhi-radius-pixels 20 \
  --high-lognhi-radius-pixels 50 --very-high-lognhi-radius-pixels 80
```

The logNHI thresholds (`20.3 / 21.0 / 21.5 / 22.0`) and the four radii
(`10 / 20 / 50 / 80`) are the values recorded inside the reference npz files; the
radii are **not** the script defaults, so pass them explicitly. The script writes
`target_*` keys recording exactly what it used — check them after a run.

### 1.3 Unified split — required for the CNN dual-tower

The two towers and the fusion head must share **one** index split, otherwise the
FlatNet tower has seen part of the evaluation set.

```bash
$PY $REPO/sota/drivers/make_unified_split.py \
  --src $RUNS/common/cnn_targets_seed42_sig15.npz \
  --outdir $R12
```

It reconstructs the full 500 000-row target tensors from the input npz, re-splits
them with a count-stratified `train_test_split` (seed 42, val 0.2), and writes:

```
$R12/cnn_targets_unified_seed42_sig15.npz     # GrowNet + fusion use this directly
$R12/splits_unified.npz                       # train_idx / val_idx, for the WZX tower
```

Defaults for `--src` / `--outdir` come from `CSST_UNIFIED_SRC` / `CSST_UNIFIED_OUTDIR`.

---

## 2. Transformer single tower (SOTA system 2 — VAL Final 0.6503)

Trained and evaluated on the **old split** (§0.3).

### 2.1 Train

```bash
$PY $REPO/hybrid_ensemble/train_transformer.py \
  --targets $RUNS/common/cnn_targets_seed42_sig15.npz \
  --train-fits $TRAIN_FITS \
  --out-dir $RUNS/tf_sig15_l40_s43 \
  --arch transformer_conv_stem --input-mode flux \
  --num-conv-layers 3 --conv-kernel 7 \
  --d-model 192 --nhead 8 --num-layers 4 --dim-ff 768 --dropout 0.1 --max-len 1024 \
  --epochs 40 --batch-size 512 --lr 1e-3 --weight-decay 1e-4 --grad-clip 1.0 \
  --lr-schedule cosine --lr-warmup-epochs 1 --seed 43 \
  --threshold 0.45 --min-z-dla 1.1 \
  --region-loss-weight 0.2 --lognhi-loss-weight 0.15 --offset-loss-weight 0.1 \
  --count-loss-weight 0.25 --count-class-weight-power 0.0 --count-loss-type ce \
  --truth-min-lognhi 20.3 --high-lognhi-threshold 22.0 \
  --high-lognhi-center-weight 4.0 --high-lognhi-log-weight 4.0 \
  --num-workers 8 --device cuda
```

The other three architectures are `--arch transformer` (v0, 0.3779),
`transformer_conv_stem_rope` (v5, 0.4919) and `transformer_conv_stem_alibi`
(v6, 0.4698) — the older 681-px line.

### 2.2 Evaluate on VAL

```bash
$PY $REPO/hybrid_ensemble/evaluate_hybrid.py \
  --models $RUNS/tf_sig15_l40_s43/best_model.pt \
  --targets $RUNS/common/cnn_targets_seed42_sig15.npz \
  --train-fits $TRAIN_FITS \
  --threshold 0.45 --min-distance 10 --min-z-dla 1.1 \
  --count-bias 0 0 0 --truth-min-lognhi 20.3 \
  --out $RUNS/tf_sig15_l40_s43/eval_val.json --device cuda
```

`--targets` supplies the **val indices**; the labels themselves come from
`--train-fits`. Passing the unified npz here would evaluate on the other split.

### 2.3 Predict on TEST

```bash
$PY $REPO/hybrid_ensemble/evaluate_hybrid.py \
  --models $RUNS/tf_sig15_l40_s43/best_model.pt \
  --targets $RUNS/common/cnn_targets_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --threshold 0.45 --min-distance 10 --min-z-dla 1.1 --count-bias 0 0 0 \
  --out $RUNS/tf_sig15_l40_s43/ensemble_eval_config.json --device cuda

$PY $REPO/hybrid_ensemble/predict_hybrid.py \
  --ensemble-config $RUNS/tf_sig15_l40_s43/ensemble_eval_config.json \
  --test-fits $TEST_FITS \
  --out $RUNS/tf_sig15_l40_s43/submission.csv --device cuda
```

---

## 3. CNN dual-tower (SOTA system 1 — VAL Final 0.6876)

Three steps: train the two towers, train the fusion head with EMA (Stage A), then
fit the 129-parameter matched-WLS readout (Stage B). `0.687569327685616` is the
result of all three together — the checkpoint alone is not the 0.6876 model.
Trained and evaluated on the **unified split**.

### 3.1 Train the two towers

GrowNet (dilated CNN tower):

```bash
cd $REPO/hybrid_ensemble && $PY train_hybrid.py \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --out-dir $R12/tower_grow_ctrl_u \
  --input-mode flux --hidden 96 --num-blocks 4 --norm-type layer --head-layers 1 \
  --epochs 10 --batch-size 512 --lr 1e-3 --seed 42 --min-z-dla 1.1 \
  --region-loss-weight 0.2 --lognhi-loss-weight 0.05 --offset-loss-weight 0.1 \
  --count-loss-weight 0.25 --count-class-weight-power 0.35 \
  --truth-min-lognhi 20.3 --high-lognhi-threshold 22.0 \
  --high-lognhi-center-weight 4.0 --high-lognhi-log-weight 4.0 \
  --num-workers 8 --device cuda
```

FlatNet (WZX tower) — trained through the vendored package, with `make_split`
overridden so it uses the unified split:

```bash
cd $REPO && UNIFIED_SPLITS=$R12/splits_unified.npz \
  $PY $REPO/sota/drivers/run_wzx_unified.py \
  --train_fits $TRAIN_FITS --output_dir $R14/tower_flat_cons_u_r14 \
  --feature_mode flux --epochs 10 --batch_size 512 --lr 8e-4 --weight_decay 1e-4 \
  --val_size 0.2 --seed 42 --split_seed 42 --num_workers 8 \
  --base_channels 96 --num_blocks 8 --dropout 0.1 --disable_tqdm
```

`UNIFIED_SPLITS` must be set — the driver exits immediately without it.

```bash
export DCK=$R12/tower_grow_ctrl_u/model.pt
export WCK=$R14/tower_flat_cons_u_r14/best_model.pt
```

### 3.2 Stage A — EMA training of the fusion head

Both towers stay frozen; only the `residual_dilated` fusion head (w128 × d3) trains.
`EMA_BASE=sg_off` selects the exact base class used by the reference runs.

Single seed (the seed-45 reference invocation, 10 epochs):

```bash
cd $REPO/hybrid_ensemble && \
FUS_SUBSET=both EMA_BASE=sg_off CUDA_VISIBLE_DEVICES=0 \
$PY $REPO/sota/drivers/run_fusion_ema_lrstep.py \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --dilated-checkpoint $DCK --wzx-checkpoint $WCK \
  --merge-mode residual_dilated --fusion-width 128 --fusion-depth 3 \
  --batch-size 512 --lr 5e-4 --min-z-dla 1.1 \
  --region-loss-weight 0.20 --lognhi-loss-weight 0.05 --offset-loss-weight 0.1 \
  --count-loss-weight 0.25 --num-workers 8 --device cuda \
  --epochs 10 --seed 45 --out-dir $RUNS/20261005_r39/seed45 \
  --lr-drop-epoch 11 --lr-after-drop 5e-5
```

The R48-SH sweep (8 seeds × 8 EMA endpoints = 64 candidates) is the same recipe with
`--epochs 20 --lr-drop-epoch 21` and a driver whose only change is the EMA save
schedule (`EMA_SAVE_EPOCHS = (6,8,10,12,14,16,18,20)` instead of `epoch >= 36`):

```bash
cd $REPO/hybrid_ensemble && \
FUS_SUBSET=both EMA_BASE=sg_off CUDA_VISIBLE_DEVICES=0 \
$PY $REPO/sota/drivers/r48sh_stage_a.py \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --dilated-checkpoint $DCK --wzx-checkpoint $WCK \
  --merge-mode residual_dilated --fusion-width 128 --fusion-depth 3 \
  --batch-size 512 --lr 5e-4 --min-z-dla 1.1 \
  --region-loss-weight 0.20 --lognhi-loss-weight 0.05 --offset-loss-weight 0.1 \
  --count-loss-weight 0.25 --num-workers 8 --device cuda \
  --epochs 20 --seed 51 --out-dir $R48/seed51 \
  --lr-drop-epoch 21 --lr-after-drop 5e-5
```

Output: `$R48/seed51/ema_ep{6,8,10,12,14,16,18,20}.pt`. The winner is **seed 51,
EMA epoch 8**.

### 3.3 Stage B — matched-WLS head readout

Fits the 129 parameters of the logNHI head by equal-weight OLS on the frozen model's
TRAIN-split matched objects, then writes back in float32.

> **Use the right driver.** `r38_matched_wls.py` and `r39_matched_wls.py` carry a
> hard-coded reference dict covering **seeds 42/43/44 and 45 only**; any other seed
> raises `KeyError` *after* the expensive decode. For seed 51 use
> `r48sh_stage_b.py`, which is the same policy with that lookup made optional.

```bash
$PY $REPO/sota/drivers/r48sh_stage_b.py \
  --seed 51 \
  --e-ckpt $R48/seed51/ema_ep8.pt \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz \
  --train-fits $TRAIN_FITS \
  --dilated-ckpt $DCK --wzx-ckpt $WCK \
  --device cuda:0 --out-dir $R48/cand/seed51_ep8
```

Output: `r38_deployable_seed51.pt` (the 0.6876 model), `r38_matched_wls_head_seed51.pt`,
`r38_matched_wls_seed51.json`, `r38_catalogs_seed51.npz`.

The same command with `r38_matched_wls.py` and `--seed 42` (or 43, 44) reproduces the
R38 seeds; `r39_matched_wls.py --seed 45` reproduces the R39 seed.

### 3.4 Rank the candidates

```bash
$PY $REPO/sota/drivers/r48sh_rank.py --root $R48
```

Writes `r48sh_leaderboard.csv`, `r48sh_summary.json`, `r48sh_report.md`.

### 3.5 Alternative — plain fusion trainer (no EMA, no WLS)

This is the 0.6584 reference model. Note it was scored on the **old split**, so it
uses the σ1.5 npz from §1.2, not the unified one:

```bash
cd $REPO/hybrid_ensemble && $PY train_feature_fusion.py \
  --targets $RUNS/common/cnn_targets_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --dilated-checkpoint $DCK --wzx-checkpoint $WCK \
  --merge-mode residual_dilated --fusion-width 128 --fusion-depth 3 \
  --epochs 40 --batch-size 512 --lr 5e-4 --seed 42 --min-z-dla 1.1 \
  --region-loss-weight 0.2 --lognhi-loss-weight 0.05 --offset-loss-weight 0.1 \
  --count-loss-weight 0.25 --num-workers 8 --device cuda \
  --out-dir $RUNS/fusion_sig15_l40
```

---

## 4. Evaluate

### 4.1 Official score of a prediction catalogue

```bash
$PY $REPO/hybrid_ensemble/score_test.py \
  --predictions <predictions.csv> \
  --truth $TEST_TRUTH \
  --out <score.json> \
  --min-lognhi 20.3 --snr-field SNR_GU
```

### 4.2 Ensemble evaluation over several members

`evaluate_hybrid.py` loads **single towers** — `arch` = `dilated` or `transformer*`.
It cannot load a dual-tower fusion checkpoint (whose `config` has `merge_mode` /
`dilated_config` / `wzx_config` and no `input_mode`). Evaluate the fusion model with
`predict_feature_fusion.py` + `score_test.py` (§5.2), or through the Stage-B driver.

```bash
$PY $REPO/hybrid_ensemble/evaluate_hybrid.py \
  --models <member1.pt> <member2.pt> \
  --weights 1.0 1.0 \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz --train-fits $TRAIN_FITS \
  --threshold 0.45 --min-distance 10 --min-z-dla 1.1 \
  --count-bias 0 0 0 --truth-min-lognhi 20.3 \
  --out $RUNS/ensemble_eval.json --device cuda
```

### 4.3 Re-score the bundled catalogue (no model run needed)

```bash
python3 $REPO/sota/verify.py \
  --targets $R12/cnn_targets_unified_seed42_sig15.npz \
  --train-fits $TRAIN_FITS
```

Reproduces `0.687569327685616` from `results/cnn-dual-tower/r48sh_seed51_ema_ep8/`.

---

## 5. Test

### 5.1 The frozen method — one-shot TEST

`final_eval.py` runs four stages, each a separate process invocation. The manifest
is written **before** any TEST access; `valcheck` aborts if the frozen VAL endpoints
do not reproduce.

```bash
$PY $REPO/sota/drivers/final_eval.py --stage manifest --device cuda:0
$PY $REPO/sota/drivers/final_eval.py --stage valcheck --device cuda:0
$PY $REPO/sota/drivers/final_eval.py --stage predict  --device cuda:0
$PY $REPO/sota/drivers/final_eval.py --stage score    --device cuda:0
```

Recorded result (seeds 42–45): **TEST Final 0.6696572892 ± 0.0019574108**,
Detection 0.6334 / Parameter 0.7241. Outputs land in `CSST_DLA_RUNS/20261005_final/`.
It reads `CSST_DLA_RUNS`, `CSST_TRAIN_FITS`, `CSST_TEST_FITS`, `CSST_TEST_TRUTH`.

### 5.2 Predict with the CNN dual-tower SOTA checkpoint

```bash
$PY $REPO/hybrid_ensemble/predict_feature_fusion.py \
  --checkpoint $REPO/results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_deployable_seed51.pt \
  --test-fits $TEST_FITS \
  --out $RUNS/r48sh_seed51/submission.csv \
  --threshold 0.45 --min-distance 10 --min-z-dla 1.1 --device cuda
```

Both towers are rebuilt from the `dilated_config` / `wzx_config` blocks embedded in
the checkpoint, so **no external tower checkpoint is needed** — the training-time
paths recorded inside it are not read. (`--dilated-checkpoint` / `--wzx-checkpoint`
override that, for legacy checkpoints that lack the embedded configs.)

Then score it with `score_test.py` (§4.1).

---

## 6. Recorded numbers

| System | Split | Final | Det | Param |
|---|---|---|---|---|
| CNN dual-tower (R48-SH seed51 / EMA ep8 + WLS) | **unified** VAL | **0.687569327685616** | 0.6712 | 0.7121 |
| Transformer single tower (`tf_sig15_l40_s43`) | old VAL | **0.6503** | 0.6042 | 0.7195 |
| CNN dual-tower, plain training (reference) | old VAL | 0.6584 | 0.6255 | 0.7078 |
| Frozen method (R38/R39 matched-WLS, seeds 42–45) | **TEST** | **0.6696572892** | 0.6334 | 0.7241 |

`0.687569327685616` is a maximum over 64 candidates on unified VAL, so it is a
selected maximum, not an unbiased generalisation estimate. The only method opened on
TEST is the frozen R38/R39 readout. See §0.3 for why the two VAL columns are not
comparable.

Full recipe: `results/reports/final_method_manifest.json` and
`results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_matched_wls_seed51.json`.
