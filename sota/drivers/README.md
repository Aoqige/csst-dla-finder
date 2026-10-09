# `sota/drivers/` — historical run drivers

These are the **verbatim** drivers that produced the headline runs of the `GU_qlf`
line, kept as a record of exactly what was executed:

| File | Run |
|---|---|
| `ema.py`, `run_fusion_ema_lrstep.py` | Stage A — EMA training of the dual-tower fusion head (R30/R32 lineage). |
| `r38_matched_wls.py`, `aggregate_r38.py` | Stage B — matched-WLS head readout, seeds 42–44 (R38). |
| `r39_matched_wls.py`, `aggregate_r39.py` | Stage B — matched-WLS head readout, seed 45 (R39). |
| `r48sh_stage_a.py`, `r48sh_stage_b.py`, `r48sh_rank.py` | R48-SH — 8 seeds × 8 EMA endpoints, 64-candidate sweep. |
| `make_unified_split.py`, `run_wzx_unified.py` | Unified TRAIN/VAL split and WZX tower training. |
| `final_eval.py` | The single TEST pass of the frozen method. |

## Read this before running them

They are **records, not a portable pipeline**. They still contain the absolute
paths of the machine they ran on (`/home/heruihua/...`, `/data/heruihua/...`),
they expect a specific run-directory layout, and several of them hard-code the
split and checkpoint paths of that particular round. `verify.py` deliberately
exempts this directory from its absolute-path check.

For a self-contained check of this branch, use:

```bash
python3 sota/verify.py
```

For the recipe itself, read
`results/reports/final_method_manifest.json` and
`results/cnn-dual-tower/r48sh_seed51_ema_ep8/r38_matched_wls_seed51.json`.
