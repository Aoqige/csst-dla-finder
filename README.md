# CSST DLA Model Zoo — SOTA branch (`network/hrh_final`)

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
RUNBOOK.md                every training / evaluation / test command
```

**Looking for commands?** [`RUNBOOK.md`](RUNBOOK.md) has the full sequence —
environment, data preparation, tower training, Stage A/B, evaluation and the
one-shot TEST pass.

## Verify this branch

```bash
python3 sota/verify.py
```

Runs against the repository alone — no external paths, no challenge data. It checks
that every module imports, that the committed library carries no machine-specific
absolute paths, that all sources compile, and — for each of the four reference
checkpoints — that it loads, matches its recorded tensor and `state_dict` element counts,
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

| File | System | `state_dict` elements | `nn.Parameter` elements | Final | Split |
|---|---|---:|---:|---|---|
| `results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_deployable_seed51.pt` | CNN dual-tower (deployable) | 3,555,527 | 3,552,053 | 0.6875693277 | VAL |
| `results/cnn-dual-tower/reference_plain_fusion/best_model.pt` | CNN dual-tower, plain training (no EMA + WLS) | 3,555,527 | 3,552,053 | 0.6584 | VAL |
| `results/transformer/tf_sig15_l40_s43/best_model.pt` | Transformer single tower | 2,502,919 | 2,502,919 | 0.6503 | VAL |

`sota/verify.py` checks the serialized `state_dict` element count. The CNN has
3,474 persistent WZX buffer elements in addition to its model `nn.Parameter`
elements, which is why those two columns differ. The Transformer totals are
identical. Stage B refits the existing 129-element `lognhi_delta` head; it adds no
parameters.

Each checkpoint holds `{model_state, config, threshold, score, training_config}`.
Read weights from `ck["model_state"]`; passing the whole dict with `strict=False`
silently leaves random initialisation — assert that `missing_keys` is empty.

## Reproduce

[`RUNBOOK.md`](RUNBOOK.md) is the canonical end-to-end recipe. Run its environment
and data-preparation sections first: they define `REPO`, `PY`, `TRAIN_FITS`, `RUNS`,
`R12`, `R14`, and `R48`, and produce the target/split inputs below. A bare `--help`
invocation only displays options; it is not a training recipe.

### Transformer single tower (exact recorded command)

This is the command for `tf_sig15_l40_s43` (0.6503 on the old VAL split). The
target npz comes from RUNBOOK section 1.2; do not substitute the CNN unified split.

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

### CNN dual-tower final checkpoint

`r38_deployable_seed51.pt` and 0.687569327685616 are not produced by
`train_feature_fusion.py` alone. Follow RUNBOOK sections 1.3 and 3.1 through 3.3, in order:

1. Make the unified split, then train the GrowNet and FlatNet towers and export
   `DCK` and `WCK` (sections 1.3 and 3.1).
2. Run the recorded 20-epoch `r48sh_stage_a.py` command with seed 51 and use
   `ema_ep8.pt` (section 3.2).
3. Run `r48sh_stage_b.py` with that Stage-A checkpoint; it writes
   `r38_deployable_seed51.pt` (section 3.3).

To reproduce the selected-maximum procedure rather than just its seed-51 winner,
run the full 8-seed x 8-EMA-endpoint sweep in section 3.2 and rank it with section
3.4. `train_feature_fusion.py` is the separate plain, old-split 0.6584 reference,
not the final CNN result.

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
