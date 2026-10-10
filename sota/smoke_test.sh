#!/usr/bin/env bash
# smoke_test.sh -- prove this branch trains and runs end to end, on its own.
#
#   bash sota/smoke_test.sh
#
# Step 1 needs nothing but the repository.  Steps 2-4 need the challenge data
# (not committed); steps 3 and 4 additionally need the two tower checkpoints the
# reference runs used.  Missing inputs make a step SKIP, not FAIL.
#
# Everything runs in a scratch directory; nothing is written into the repository.
set -uo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python3}"

TRAIN_FITS="${TRAIN_FITS:-/data/heruihua/newer/train_500k_GU_qlf.fits}"
TEST_FITS="${TEST_FITS:-/data/heruihua/newer/test_100k_GU.fits}"
TEST_TRUTH="${TEST_TRUTH:-/data/heruihua/newer/test_truth_100k_GU.fits}"
TARGETS="${TARGETS:-$HOME/csst_dla_runs/20261003_r12/cnn_targets_unified_seed42_sig15.npz}"
SPLITS="${SPLITS:-$HOME/csst_dla_runs/20261003_r12/splits_unified.npz}"
DCK="${DCK:-$HOME/csst_dla_runs/20261003_r12/tower_grow_ctrl_u/model.pt}"
WCK="${WCK:-$HOME/csst_dla_runs/20261003_r14/tower_flat_cons_u_r14/best_model.pt}"

WORK="${WORK:-/tmp/sota_smoke_$$}"
export PYTHONPATH="$REPO/src:$REPO/hybrid_ensemble:$REPO/vendor"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$WORK"

PASS=0; FAIL=0; SKIP=0
note() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   [PASS] %s\n' "$*"; PASS=$((PASS+1)); }
bad()  { printf '   [FAIL] %s\n' "$*"; FAIL=$((FAIL+1)); }
skip() { printf '   [SKIP] %s\n' "$*"; SKIP=$((SKIP+1)); }

run() {  # run <label> <logfile> <cmd...>
  local label="$1" log="$2"; shift 2
  if "$@" >"$log" 2>&1; then ok "$label"; else
    bad "$label"; tail -5 "$log" | sed 's/^/        /'
  fi
}

# ---------------------------------------------------------------- step 1
note "step 1  self-check (repository only)"
if "$PY" "$REPO/sota/verify.py" >"$WORK/verify.log" 2>&1; then
  ok "sota/verify.py -> $(grep -c '^\[PASS\]' "$WORK/verify.log") checks"
else
  bad "sota/verify.py"; tail -20 "$WORK/verify.log" | sed 's/^/        /'
fi

# ---------------------------------------------------------------- step 2
note "step 2  training entry points (1 epoch, few hundred samples)"
if [[ ! -f "$TARGETS" || ! -f "$TRAIN_FITS" ]]; then
  skip "no targets/FITS; set TARGETS and TRAIN_FITS"
else
  run "train_transformer.py" "$WORK/tf.log" \
    "$PY" "$REPO/hybrid_ensemble/train_transformer.py" \
      --targets "$TARGETS" --train-fits "$TRAIN_FITS" --out-dir "$WORK/tf" \
      --arch transformer_conv_stem --input-mode flux --num-conv-layers 3 \
      --epochs 1 --batch-size 128 --max-train-samples 512 --max-val-samples 256 \
      --num-workers 0 --device cuda

  run "train_hybrid.py (GrowNet tower)" "$WORK/grow.log" \
    env -C "$REPO/hybrid_ensemble" \
    "$PY" train_hybrid.py \
      --targets "$TARGETS" --train-fits "$TRAIN_FITS" --out-dir "$WORK/tower_grow" \
      --input-mode flux --hidden 96 --num-blocks 4 --norm-type layer --head-layers 1 \
      --epochs 1 --batch-size 256 --max-train-samples 20000 --max-val-samples 2000 \
      --count-class-weight-power 0.35 --num-workers 0 --device cuda

  if [[ ! -f "$SPLITS" ]]; then
    skip "run_wzx_unified.py: no splits file"
  else
    run "run_wzx_unified.py (FlatNet tower)" "$WORK/flat.log" \
      env UNIFIED_SPLITS="$SPLITS" "$PY" "$REPO/sota/drivers/run_wzx_unified.py" \
        --train_fits "$TRAIN_FITS" --output_dir "$WORK/tower_flat" \
        --feature_mode flux --epochs 1 --batch_size 128 --max_samples 512 \
        --num_workers 0 --base_channels 96 --num_blocks 8 --disable_tqdm
  fi

  if [[ -f "$WORK/tower_grow/model.pt" && -f "$WORK/tower_flat/best_model.pt" ]]; then
    run "run_fusion_ema_lrstep.py (Stage A)" "$WORK/stageA.log" \
      env -C "$REPO/hybrid_ensemble" FUS_SUBSET=both EMA_BASE=sg_off \
      "$PY" "$REPO/sota/drivers/run_fusion_ema_lrstep.py" \
        --targets "$TARGETS" --train-fits "$TRAIN_FITS" \
        --dilated-checkpoint "$WORK/tower_grow/model.pt" \
        --wzx-checkpoint "$WORK/tower_flat/best_model.pt" \
        --merge-mode residual_dilated --fusion-width 128 --fusion-depth 3 \
        --batch-size 128 --max-train-samples 2000 --max-val-samples 500 \
        --epochs 1 --seed 51 --out-dir "$WORK/stageA" \
        --lr-drop-epoch 2 --lr-after-drop 5e-5 --num-workers 0 --device cuda
  else
    skip "Stage A: the two smoke towers were not produced"
  fi
fi

# ---------------------------------------------------------------- step 3
note "step 3  Stage B on the bundled checkpoint (must reproduce 0.687569327685616)"
BUNDLE="$REPO/results/cnn-dual-tower/r48sh_seed51_ema_ep8"
if [[ ! -f "$BUNDLE/ema_ep8.pt" || ! -f "$TARGETS" || ! -f "$DCK" || ! -f "$WCK" ]]; then
  skip "needs the bundled checkpoint plus TARGETS/DCK/WCK"
else
  run "r48sh_stage_b.py --seed 51" "$WORK/stageB.log" \
    "$PY" "$REPO/sota/drivers/r48sh_stage_b.py" \
      --seed 51 --e-ckpt "$BUNDLE/ema_ep8.pt" \
      --targets "$TARGETS" --train-fits "$TRAIN_FITS" \
      --dilated-ckpt "$DCK" --wzx-ckpt "$WCK" \
      --device cuda:0 --out-dir "$WORK/stageB"
  got=$(grep -o '"R38_Final": [0-9.]*' "$WORK/stageB.log" | head -1 | awk '{print $2}')
  if [[ "$got" == "0.687569327685616" ]]; then
    ok "R38_Final = $got"
  elif [[ -n "$got" ]]; then
    bad "R38_Final = $got (expected 0.687569327685616)"
  fi
fi

# ---------------------------------------------------------------- step 4
note "step 4  inference + official scoring"
if [[ ! -f "$BUNDLE/r38_deployable_seed51.pt" || ! -f "$TEST_FITS" || ! -f "$TEST_TRUTH" ]]; then
  skip "needs the deployable checkpoint plus TEST_FITS/TEST_TRUTH"
else
  run "predict_feature_fusion.py" "$WORK/predict.log" \
    "$PY" "$REPO/hybrid_ensemble/predict_feature_fusion.py" \
      --checkpoint "$BUNDLE/r38_deployable_seed51.pt" --test-fits "$TEST_FITS" \
      --out "$WORK/submission.csv" --threshold 0.45 --device cuda
  run "score_test.py" "$WORK/score.log" \
    "$PY" "$REPO/hybrid_ensemble/score_test.py" \
      --predictions "$WORK/submission.csv" --truth "$TEST_TRUTH" \
      --out "$WORK/score.json" --min-lognhi 20.3 --snr-field SNR_GU
  if [[ -f "$WORK/score.json" ]]; then
    printf '        TEST Final = %s\n' \
      "$("$PY" -c "import json;print(json.load(open('$WORK/score.json'))['final_score'])")"
  fi
fi

note "summary"
printf '   PASS %d   FAIL %d   SKIP %d\n' "$PASS" "$FAIL" "$SKIP"
printf '   scratch: %s\n' "$WORK"
[[ "$FAIL" -eq 0 ]]
