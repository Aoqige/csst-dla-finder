"""Bidirectional audit of RUNBOOK.md against the arguments the reference runs recorded.

The earlier check only asked "does every flag the RUNBOOK writes match the recorded
value?".  That misses the dangerous direction: a setting the run recorded, that the
RUNBOOK does NOT write, whose script default differs from the recorded value -- run
the RUNBOOK verbatim and you silently get a different experiment.

For every training step this script:
  1. reads the recorded settings (training_args.json / config.json / training_config),
  2. reads the flags the RUNBOOK block actually passes,
  3. reads the script's argparse defaults (by AST, without executing the script),
  4. classifies each recorded key as:
       OK-WRITTEN   -- in RUNBOOK and equal
       BAD-VALUE    -- in RUNBOOK but different
       OK-DEFAULT   -- not in RUNBOOK, but the script default equals the record
       MISSING      -- not in RUNBOOK and the default differs  <-- the real bug
       NO-FLAG      -- not a CLI flag (derived value), nothing to pass

NOTE ON NAMING: this repo uses both styles -- the in-repo scripts use dashes
(`--batch-size`) and the vendored WZX package uses underscores (`--batch_size`).
Every lookup therefore tries both.

Usage:  python3 audit_runbook.py <repo>      # with RUNS=<run root> if not ~/csst_dla_runs
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
RUNBOOK = (REPO / "RUNBOOK.md").read_text(encoding="utf-8")


def flag_candidates(key: str) -> list[str]:
    """`batch_size` -> ['--batch_size', '--batch-size'] (the repo uses both)."""
    return list(dict.fromkeys(["--" + key, "--" + key.replace("_", "-")]))


# --------------------------------------------------------------------------- #
# 1. argparse defaults, by AST (no import, no execution)
# --------------------------------------------------------------------------- #
def script_flags(path: Path) -> dict[str, tuple]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "add_argument"):
            continue
        flags = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        flags = [s for s in flags if s.startswith("--")]
        if not flags:
            continue
        default, has_default, action = None, False, None
        for kw in node.keywords:
            if kw.arg == "default":
                has_default = True
                try:
                    default = ast.literal_eval(kw.value)
                except Exception:  # noqa: BLE001
                    default = ast.unparse(kw.value)
            elif kw.arg == "action":
                try:
                    action = ast.literal_eval(kw.value)
                except Exception:  # noqa: BLE001
                    pass
        if not has_default and action == "store_true":
            default = False
        for s in flags:
            out[s] = (default, action)
    return out


# --------------------------------------------------------------------------- #
# 2. RUNBOOK block -> {flag: value}
# --------------------------------------------------------------------------- #
def runbook_flags(anchor: str, block_index: int = 0) -> dict[str, str]:
    i = RUNBOOK.find(anchor)
    assert i >= 0, f"anchor not found: {anchor!r}"
    blocks = RUNBOOK[i:].split("```bash")[1:]
    block = blocks[block_index].split("```")[0].replace("\\\n", " ")
    out: dict[str, str] = {}
    for tok in re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*(?:[= ]\S+)?", block):
        parts = tok.split(None, 1)
        out.setdefault(parts[0], parts[1] if len(parts) > 1 else "<flag>")
    return out


# --------------------------------------------------------------------------- #
# 3. comparison
# --------------------------------------------------------------------------- #
def same(a, b) -> bool:
    """Recorded value vs RUNBOOK literal (which may be `$VAR`, `1e-3`, `0.20`)."""
    if isinstance(b, str) and b.startswith("$"):
        return True  # shell variable -- the path table covers these
    if b == "<flag>":
        # a bare store_true flag written in the RUNBOOK, e.g. `--disable_tqdm`
        return a is True or str(a).lower() == "true"
    if a is None and b is None:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        pass
    if isinstance(a, bool):
        return str(a).lower() == str(b).lower()
    return str(a) == str(b)


STEPS = [
    dict(name="Transformer   (2.1)", script="hybrid_ensemble/train_transformer.py",
         anchor="### 2.1 Train", block=0,
         record=("json", "results/transformer/tf_sig15_l40_s43/training_args.json")),
    dict(name="GrowNet tower (3.1)", script="hybrid_ensemble/train_hybrid.py",
         anchor="### 3.1 Train the two towers", block=0,
         record=("json", "$RUNS/20261003_r12/tower_grow_ctrl_u/training_args.json")),
    dict(name="FlatNet tower (3.1)", script="vendor/csst_dla_wzx_pkg/train.py",
         anchor="### 3.1 Train the two towers", block=1,
         record=("json", "$RUNS/20261003_r14/tower_flat_cons_u_r14/config.json")),
    dict(name="Stage A head  (3.2)", script="sota/drivers/r48sh_stage_a.py",
         extra_script="hybrid_ensemble/train_feature_fusion.py",
         anchor="### 3.2 Stage A", block=1,
         record=("ckpt", "results/cnn-dual-tower/r48sh_seed51_ema_ep8/ema_ep8.pt")),
]


def load_record(kind: str, rel: str):
    rel = rel.replace("$RUNS", os.environ.get("RUNS", str(Path.home() / "csst_dla_runs")))
    p = Path(rel)
    if not p.is_absolute():
        p = REPO / rel
    if kind == "json":
        return json.loads(p.read_text())
    import torch
    return torch.load(p, map_location="cpu", weights_only=False).get("training_config") or {}


def main() -> int:
    problems = 0
    for step in STEPS:
        print(f"\n{'='*78}\n{step['name']}\n{'='*78}")
        rb = runbook_flags(step["anchor"], step["block"])
        flags = script_flags(REPO / step["script"])
        if step.get("extra_script"):
            flags.update(script_flags(REPO / step["extra_script"]))
        try:
            rec = load_record(*step["record"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] record unreadable: {exc}")
            continue

        print(f"  RUNBOOK passes {len(rb)} flags; script defines {len(flags)}; record has {len(rec)} settings")
        buckets: dict[str, list[str]] = {k: [] for k in
                                         ("OK-WRITTEN", "BAD-VALUE", "OK-DEFAULT", "MISSING", "NO-FLAG")}
        for k, v in sorted(rec.items()):
            cands = flag_candidates(k)
            rb_flag = next((c for c in cands if c in rb), None)
            sp_flag = next((c for c in cands if c in flags), None)
            if rb_flag is not None:
                (buckets["OK-WRITTEN"] if same(v, rb[rb_flag]) else buckets["BAD-VALUE"]).append(
                    f"{k} = {v!r}   (RUNBOOK writes {rb[rb_flag]!r})")
            elif sp_flag is not None:
                dflt = flags[sp_flag][0]
                (buckets["OK-DEFAULT"] if same(v, dflt) else buckets["MISSING"]).append(
                    f"{k} = {v!r}   (script default {dflt!r})")
            else:
                buckets["NO-FLAG"].append(f"{k} = {v!r}")

        for label in ("MISSING", "BAD-VALUE", "OK-DEFAULT", "OK-WRITTEN", "NO-FLAG"):
            items = buckets[label]
            mark = "  [!!] " if label in ("MISSING", "BAD-VALUE") else "  [ok] "
            print(f"{mark}{label:11s} {len(items)}")
            for it in items[:45]:
                print(f"        {it}")
        problems += len(buckets["MISSING"]) + len(buckets["BAD-VALUE"])

        unknown = [f for f in rb if f not in flags]
        if unknown:
            print(f"  [??] RUNBOOK flags absent from the parser: {unknown}")

    print(f"\n{'='*78}\nTOTAL hard problems (MISSING + BAD-VALUE): {problems}\n{'='*78}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
