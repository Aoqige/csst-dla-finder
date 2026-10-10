# `sota/drivers/` — current-line run drivers

The drivers that produced the headline runs of the `GU_qlf` line. Their **recipe,
arguments and algorithmic content are verbatim** — what was run is what is here.
Only the path bootstrap has been made portable (see below), so they run from any
checkout.

| File | Run |
|---|---|
| `ema.py` | EMA wrapper (beta 0.999, updated after every `optimizer.step()`). |
| `run_fusion_ema_lrstep.py` | Stage A — EMA training of the dual-tower fusion head (R30/R32 lineage). |
| `run_fusion_subset.py` | Base class `SubsetFusionNet` (G-only / F-only / both feature subsets). Imported by `model_sg.py`. |
| `model_sg.py` | `TrainNet`: the `EMA_BASE=sg_off` variant used by the reference runs. |
| `r38_matched_wls.py`, `aggregate_r38.py` | Stage B — matched-WLS head readout, seeds 42–44 (R38). |
| `r39_matched_wls.py`, `aggregate_r39.py` | Stage B — matched-WLS head readout, seed 45 (R39). |
| `r48sh_stage_a.py`, `r48sh_stage_b.py`, `r48sh_rank.py` | R48-SH — 8 seeds × 8 EMA endpoints, 64-candidate sweep. |
| `make_unified_split.py`, `run_wzx_unified.py` | Unified TRAIN/VAL split and WZX tower training. |
| `final_eval.py` | The single TEST pass of the frozen method. |

## Portability changes

The originals resolved their imports and defaults through absolute paths
(`/home/heruihua/...`). Each driver's path bootstrap now resolves the repository
root from `__file__` and puts `src/`, `hybrid_ensemble/`, `vendor/` and the driver's
own directory on `sys.path`. Run-directory and data defaults became
environment-overridable:

| Variable | Used by |
|---|---|
| `CSST_DLA_RUNS` | `final_eval.py`, `aggregate_r38.py`, `aggregate_r39.py` |
| `CSST_TRAIN_FITS`, `CSST_TEST_FITS`, `CSST_TEST_TRUTH` | `final_eval.py`, `make_unified_split.py` |
| `CSST_R38_DIR`, `CSST_R39_DIR` | `aggregate_r38.py`, `aggregate_r39.py` |

Two files were added to this directory because the Stage-A driver imports them:
`run_fusion_subset.py` (was `~/r11/`) and `model_sg.py` (was `~/r27/`). They are
otherwise unmodified.

The `source_hashes` block inside `final_eval.py` still records the **original**
paths — that is provenance, not a runtime dependency.

## Running them

See [`../../RUNBOOK.md`](../../RUNBOOK.md) for the full command lines. For a
self-contained check of this branch that needs no challenge data:

```bash
python3 sota/verify.py
```
