# Handoff guide: reproduce, operate, and extend `network/hrh_final`

This page is the starting point for a new human maintainer or an AI agent. It
separates four things that are easy to conflate: proving the checked-in snapshot,
re-scoring a bundled prediction catalogue, reproducing a recorded training result,
and starting a new experiment.

The command source of truth is [RUNBOOK.md](RUNBOOK.md). Do not copy a shortened
command from an issue, chat, or old branch: every command there was checked against
the recorded run arguments. This guide instead tells you which command path applies,
which evidence it supplies, and where handoff must stop for missing external assets.

## First five minutes

1. Record the exact source identity and make sure the worktree is clean:

   ```bash
   git status --short --branch
   git rev-parse HEAD
   git log -1 --oneline
   ```

   Work from an isolated checkout or worktree. Do not treat a dirty server checkout
   as a reproducible source snapshot.

2. Follow RUNBOOK section 0 to set `REPO`, `PY`, `PYTHONPATH`, and your run roots.
   The reference environment is Python 3.10.16, PyTorch 2.5.1 with CUDA 12.4, and
   the pinned non-PyTorch packages in `requirements.txt`.

3. Run the repository-only integrity check before training or loading a checkpoint:

   ```bash
   $PY $REPO/sota/verify.py
   ```

   A successful check proves imports, source portability, checkpoint rebuild and
   strict state loading. In this handoff snapshot it reports 22 passing checks. It
   does **not** prove that an external FITS file, split, or training run is correct.

4. Decide which proof you need from the matrix below. Do not call a re-score a
   retraining result, and do not call a skipped smoke-test step a pass.

## Reproduction matrix

| Goal | Required inputs | Entry point | Acceptance criterion | What it does not prove |
|---|---|---|---|---|
| Check code and bundled artifacts | repository only | `sota/verify.py` | all checks pass | external data identity or training |
| Re-score the released CNN catalogue | unified targets + TRAIN FITS | `sota/verify.py --targets ... --train-fits ...` | `0.687569327685616` on unified VAL | a fresh model was trained |
| Re-run Stage B for the released seed-51 candidate | above + GrowNet/FlatNet tower checkpoints + bundled `ema_ep8.pt` | RUNBOOK section 3.3 | regenerated deployable result is `0.687569327685616` | the two towers or Stage A were retrained |
| Retrain the released CNN candidate end to end | all CNN inputs | RUNBOOK sections 1.3 and 3.1 through 3.3 | follow RUNBOOK section 7.1; seed 51 and 52 were verified bit-for-bit at model-state level | an unbiased TEST estimate |
| Retrain the Transformer | old-split targets + TRAIN FITS | RUNBOOK section 2.1 | configuration is identical; assess against RUNBOOK section 7.2 | bit-identical weights or the exact recorded `0.6503` score |
| Run or score TEST | explicit TEST authorization and TEST FITS/truth | RUNBOOK section 5 | follow the frozen-method protocol | permission to tune, select, or refit on TEST |

The two CNN and Transformer VAL scores use different split identities. The CNN
`0.687569327685616` is a selected maximum over 64 unified-VAL candidates, not a
TEST result and not an unbiased generalisation estimate.

## External assets: handoff gate

Challenge data and historical run directories are intentionally not in Git. A new
maintainer cannot reproduce a numerical result until they receive authorized copies
of the following assets. The historical absolute paths in JSON manifests are
provenance, not paths a new machine should use.

| Asset | Needed for | Reference MD5 |
|---|---|---|
| `train_500k_GU_qlf.fits` | all training and VAL re-score | `95c35c12f4d86f9c9146220f28494254` |
| `cnn_targets_unified_seed42_sig15.npz` | CNN Stage A/B and bundle re-score | `bc2ecef3c251c04599a3a0589d2c9554` |
| GrowNet tower checkpoint (`DCK`) | CNN Stage A/B | `9aeee291c016b0ea21743ec4bd39c199` |
| FlatNet tower checkpoint (`WCK`) | CNN Stage A/B | `f56dc6505cdf1d516622c866a24fd93c` |
| TEST FITS and truth | frozen TEST protocol only | see `results/reports/final_method_manifest.json` |

After setting the RUNBOOK variables, verify the first four assets before doing any
numeric comparison:

```bash
md5sum "$TRAIN_FITS" \
  "$R12/cnn_targets_unified_seed42_sig15.npz" \
  "$DCK" \
  "$WCK"
```

The expected split identities are `train_idx` MD5
`b237712677c7a17e24cb840b051a5720` and `val_idx` MD5
`69424cc2437526c5447a09fe13f9efc0`. The full immutable evidence trail is in
`results/reports/final_method_manifest.json` and
`results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_matched_wls_seed51.json`.

If any asset is absent or has a different hash, stop the reproduction claim there.
Do not compensate by changing a seed, split, loss, decoder, threshold, or checkpoint.

## Which framework entry point to use

| Task | Use | Important boundary |
|---|---|---|
| Train a Transformer single tower | `hybrid_ensemble/train_transformer.py`; RUNBOOK section 2.1 | old split; GPU attention is nondeterministic in the recorded configuration |
| Train the GrowNet tower | `hybrid_ensemble/train_hybrid.py`; RUNBOOK section 3.1 | CNN training requires the unified split |
| Train the FlatNet/WZX tower | `sota/drivers/run_wzx_unified.py`; RUNBOOK section 3.1 | set `UNIFIED_SPLITS`; use the same split as GrowNet |
| Train the final CNN fusion head | `sota/drivers/r48sh_stage_a.py`; RUNBOOK section 3.2 | both towers remain frozen; seed 51, 20 epochs, then `ema_ep8.pt` for the released candidate |
| Produce the released CNN deployable checkpoint | `sota/drivers/r48sh_stage_b.py`; RUNBOOK section 3.3 | refits the existing 129-element logNHI head; it does not train a new backbone |
| Rank the 64 CNN candidates | `sota/drivers/r48sh_rank.py`; RUNBOOK section 3.4 | selection is on unified VAL only |
| Infer with the released CNN checkpoint | `hybrid_ensemble/predict_feature_fusion.py`; RUNBOOK section 5.2 | embedded tower configs rebuild the model; do not load the whole checkpoint with `strict=False` |
| Evaluate a single-tower/ensemble model | `hybrid_ensemble/evaluate_hybrid.py`; RUNBOOK section 4 | it cannot load a dual-tower fusion checkpoint |

All models obey the same five-head contract. Read
[hybrid_ensemble/README.md](hybrid_ensemble/README.md) before changing a backbone,
decoder, input mode, or evaluation path.

## Safe workflow for a new experiment

1. Start a new branch and a new run directory outside the repository. Preserve the
   frozen artifacts under `results/`; do not overwrite them.
2. Record `git rev-parse HEAD`, data/split hashes, the full command line, package
   versions, seed, and output directory with the run. Keep TRAIN/VAL and TEST
   identities separate in every report.
3. For a CNN dual tower, construct one unified split first. Both towers and the
   fusion driver must consume that exact split.
4. Evaluate on VAL while developing. Do not use TEST to choose a seed, epoch,
   threshold, count bias, decoder setting, head refit, or ensemble weight.
5. Report the proof level precisely: integrity check, bundle re-score, Stage-B
   replay, full retraining, or TEST protocol. Include `n_pred` and split identity;
   a Final score alone is not enough.

## Smoke-test and TEST-data safety

`bash sota/smoke_test.sh` is a useful entry-point smoke test, but missing inputs
produce `[SKIP]`, not a failure. Inspect its final PASS/FAIL/SKIP counts; a run with
skips is not evidence that skipped training paths work.

The script's fourth step will predict and score TEST if TEST files are available.
Unless you have explicit authorization to run the frozen TEST protocol, block that
step deliberately:

```bash
TEST_FITS=/__no_test_access__ \
TEST_TRUTH=/__no_test_access__ \
bash $REPO/sota/smoke_test.sh
```

This protects TEST access while still allowing repository checks and any authorized
TRAIN/VAL smoke steps. It does not replace the full RUNBOOK reproduction procedures.

## Instructions for an AI handoff

An AI taking over this repository should follow this order:

1. Read this page, the root README, and the relevant RUNBOOK section before
   proposing a command.
2. State the branch, commit, worktree status, desired evidence level, data line,
   and split before interpreting a metric.
3. Verify supplied external asset hashes before calling a result reproducible.
4. Use the historical `sota/drivers/` recipes as frozen provenance. Put new ideas in
   a separate driver or branch and keep their outputs outside Git.
5. Never silently downgrade a failure to a CPU fallback, a different split, a
   different seed, or an incomplete smoke test. Report what was verified and what
   remains unavailable.
6. Treat `final_method_manifest.json` as the TEST boundary. Its
   `forbidden_after_test` list remains in force for the frozen method.

## Handoff completion checklist

A recipient is ready to claim a reproduction only when all applicable items are true:

- [ ] source commit and clean worktree were recorded;
- [ ] `sota/verify.py` passed;
- [ ] every required external asset has the reference hash;
- [ ] the correct RUNBOOK path was run without substituted defaults;
- [ ] the reported result matches the appropriate acceptance criterion above;
- [ ] VAL/TEST and old/unified splits are labelled correctly; and
- [ ] no TEST action was taken without explicit authorization.

If one item is false, the correct status is "not yet reproduced", with the missing
input or failed gate stated explicitly.
