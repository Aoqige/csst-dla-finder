# RUNBOOK — training, evaluation and test commands

Every command needed to reproduce the two SOTA systems on this branch. Commands are
written against a checkout of this repository; nothing below assumes a particular
machine.

> **New maintainer or AI?** Read [HANDOFF.md](HANDOFF.md) first for the data gate, proof levels, framework routing, and TEST boundary.
>
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

#### Epochs and learning rate are not optional

Every training step has a default for `--epochs` and `--lr`, so a command will run
without them — but **none of the defaults equals the value the reference run used**:

| Step | `--epochs` default | `--lr` default | **reference run** | LR schedule |
|---|---|---|---|---|
| GrowNet tower (§3.1) | 16 | 1e-3 | **10 / 1e-3** | none — constant |
| FlatNet tower (§3.1) | 30 | 8e-4 | **10 / 8e-4** | `CosineAnnealingLR(T_max=epochs)` |
| Stage A fusion head (§3.2) | 8 | 8e-4 | **20 / 5e-4** | step drop at `--lr-drop-epoch` |

Two couplings make this worse than a plain wrong number:

* **The FlatNet cosine schedule is parameterised by `epochs`**
  (`CosineAnnealingLR(optimizer, T_max=args.epochs)`). Changing `--epochs` does not
  just change how long it trains — it changes the entire learning-rate curve.
* **Stage A drops the LR at `--lr-drop-epoch`.** The reference invocation is
  `--epochs 20 --lr-drop-epoch 21`: the drop is deliberately placed one epoch *past*
  the end so it never fires and the LR stays flat at 5e-4. Raising `--epochs` above
  21 silently makes the LR collapse to 5e-5 mid-run.

`EMA_SAVE_EPOCHS = (6,8,10,12,14,16,18,20)` is also hard-coded in
`r48sh_stage_a.py`: with fewer than 20 epochs you cannot produce all eight EMA
endpoints — and the 0.6876 candidate (EMA epoch 8) is one of them.

Stage A's `--lr` applies to the **fusion head only**: both towers stay frozen
(`--train-backbones` is off by default), matching `train_backbones=False` in the
recorded config. `--backbone-lr` is only consulted when `--train-backbones` is set.

#### Every argument that affects training is written out

A handful of flags equal the script's current default. They are written explicitly
anyway, so a command never depends on a default staying put:

* the vendored FlatNet package's loss weights and target widths — `--lambda_count`,
  `--lambda_heatmap`, `--lambda_region`, `--lambda_lognhi`, `--lambda_offset`,
  `--sigma_bins`, `--region_half_width_bins`, `--region_lognhi_scale`,
  `--heatmap_positive_weight`, `--region_positive_weight` (§3.1);
* Stage A's `--high-lognhi-threshold / -center-weight / -log-weight`, matching §2.1
  and §3.1 where the same three are already explicit (§3.2).

Boolean switches that stayed **off** are deliberately absent — omitting a
`store_true` flag is how "off" is expressed: `--train-backbones`,
`--no_class_weights`, `--no_context_channels`, `--allow_legacy_input_modes`,
`--offset_target_clip`.

This was checked mechanically, not by eye. `$REPO/sota/audit_runbook.py` reads every
recorded setting back out of `training_args.json` / `config.json` / the checkpoint's
embedded `training_config`, and sorts each one into: written-and-equal,
written-but-different, **equal-to-default**, or **missing** (recorded, not written,
and the script default differs — the only category that silently changes the
experiment). Current counts of the two bad categories: **0 and 0**.

Run it yourself against a run root that still holds the reference runs:

```bash
RUNS=$HOME/csst_dla_runs python3 $REPO/sota/audit_runbook.py $REPO
```

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
  --base_channels 96 --num_blocks 8 --dropout 0.1 --disable_tqdm \
  --sigma_bins 2.0 --region_half_width_bins 8 --region_lognhi_scale 2.0 \
  --lambda_count 1.0 --lambda_heatmap 1.0 --lambda_region 0.25 \
  --lambda_lognhi 0.25 --lambda_offset 0.2 \
  --heatmap_positive_weight 10.0 --region_positive_weight 2.0
```

The last three lines are the values the reference run used **and** the current
defaults of the vendored package. They are written out anyway: this package is
third-party code, and if its defaults ever change the command would otherwise
silently train a different loss. `--no_class_weights` / `--no_context_channels`
are `store_true` and stayed off, so they are deliberately absent.

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
  --high-lognhi-threshold 22.0 --high-lognhi-center-weight 4.0 --high-lognhi-log-weight 4.0 \
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
  --high-lognhi-threshold 22.0 --high-lognhi-center-weight 4.0 --high-lognhi-log-weight 4.0 \
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

---

## 7. Reproducibility

### 7.1 CNN dual-tower — bit-reproducible

Verified on 2026-10-09 by retraining from scratch with this branch, in a clean
`git archive` of it, on different GPUs than the original run.

**Two seeds were retrained end to end** (Stage A, 20 epochs each) and compared with
the original runs:

| seed | retrain `E_Final` | recorded | retrain post-WLS Final | recorded |
|---|---|---|---|---|
| 51 | 0.6759726478960849 | 0.675972648 | **0.687569327685616** | 0.687569328 |
| 52 | 0.6711619670364222 | 0.671161967 | **0.6829976568035152** | 0.682997657 |

All four agree to full precision, and the EMA VAL trajectory is identical at every
one of the 21 epochs (`max|Δ| = 0.0000 pp`, `n_pred` included).

**Do not compare checkpoint md5s.** The retrained `ema_ep8.pt` files have *different*
md5 from the originals, which looks like non-reproducibility and is not: every one of
the **246 weight tensors is bit-identical** (`max|d| = 0.000e+00`). The only
difference in the file is `training_config["out_dir"]` — the output-directory string
you passed on the command line. Compare `model_state` tensors, or the recorded
metrics, never the file hash.

**Seed spread.** Across the eight R48-SH seeds the best unified-VAL Final ranges
0.6809–0.6876 (0.67 pp); the two retrained seeds land 0.46 pp apart. Treat differences
below ~0.7 pp between single seeds as noise, not signal.

### 7.2 Transformer single tower — NOT bit-reproducible (GPU non-determinism)

Retraining seed 43 with the recipe of §2.1, in a clean `git archive`, does **not**
reproduce the recorded run:

| | best epoch | VAL Final |
|---|---|---|
| recorded | 33 | 0.650330197 |
| retrained | 29 | **0.658630** |

Δbest **+0.83 pp**. The two trajectories diverge from epoch 2 and reach
`max|Δ| = 10.56 pp`; `n_pred` differs from epoch 2 onward. `training_args.json`
agrees on 39/40 fields — the only difference is `out_dir`. The loss curves stay
close (epoch 1 differs by 2.4e-3, later epochs ~1e-4), so this is not a
configuration mistake.

**The cause is GPU operator non-determinism, not data loading.** Three 1-epoch runs,
same GPU, same seed, full training set:

| run | `num_workers` | epoch-1 loss |
|---|---|---|
| a | 0 | 0.142193644740955 |
| b | 0 | 0.146460078262036 |
| c | 8 | 0.143304908537655 |

(a) and (b) use identical settings and still differ, so the DataLoader is not the
cause. `train_transformer.py` sets `torch.manual_seed(seed)` but supplies no
`worker_init_fn`, no `use_deterministic_algorithms` and no `generator=`; the attention
path uses the non-deterministic flash / memory-efficient SDP kernels.

**Forcing determinism does fix it.** With `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True)`, and flash / memory-efficient SDP disabled
(math SDP only), two 1-epoch runs become bit-identical:

| run | epoch-1 loss |
|---|---|
| a | 0.141228320549423 |
| b | 0.141228320549423 |

```python
# sitecustomize.py on PYTHONPATH, or the same lines at the top of a training script
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
```

**Consequence for reading the numbers.** The recorded 0.6503 is one draw from a
distribution with a run-to-run spread of order **0.8 pp**. A single-seed comparison
between the Transformer and the CNN tower (0.6503 vs 0.6584, 0.0081 apart) is inside
that noise. The CNN dual-tower numbers do not have this problem — §7.1 reproduces
them to the last digit.
