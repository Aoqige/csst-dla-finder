#!/usr/bin/env python3
"""Standalone self-check for the ``sota/hrh_final`` branch.

Runs against this repository only -- no external paths, no challenge data.

    python3 sota/verify.py

Checks, in order:

  1. every first-party module imports;
  2. the committed library carries no absolute paths;
  3. each reference checkpoint loads and matches its recorded tensor count and
     parameter count;
  4. each checkpoint is rebuilt from its own ``config`` block and loaded with
     ``strict=True`` (an exact key match), which is what proves the code on this
     branch is the code that produced the checkpoint;
  5. each rebuilt model completes a forward pass on synthetic input and emits
     the five-head contract;
  6. the bundled prediction catalogue has the recorded schema.

Optional numeric check (needs the challenge data, which is not committed):

    python3 sota/verify.py --targets <cnn_targets_unified_seed42_sig15.npz> \
                           --train-fits <train_500k_GU_qlf.fits>

adds step 7: rebuild the VAL truth catalogue, re-score the bundled prediction
catalogue with the official scorer, and compare against the recorded
0.687569327685616.

Exit code 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "hybrid_ensemble", ROOT / "vendor"):
    sys.path.insert(0, str(_p))

RESULTS = ROOT / "results"
CNN = RESULTS / "cnn-dual-tower"
TF = RESULTS / "transformer"

# ---------------------------------------------------------------- references

RECORDED = {
    "cnn_best": {
        "path": CNN / "r48sh_seed51_ema_ep8" / "r38_deployable_seed51.pt",
        "n_params": 3_555_527,
        "n_tensors": 246,
        "note": "CNN dual-tower SOTA (deployable: Stage-A EMA ep8 + Stage-B WLS head)",
    },
    "cnn_stage_a": {
        "path": CNN / "r48sh_seed51_ema_ep8" / "ema_ep8.pt",
        "n_params": 3_555_527,
        "n_tensors": 246,
        "note": "CNN dual-tower Stage-A checkpoint (seed 51, EMA epoch 8)",
    },
    "cnn_reference": {
        "path": CNN / "reference_plain_fusion" / "best_model.pt",
        "n_params": 3_555_527,
        "n_tensors": 246,
        "note": "CNN dual-tower, plain training (no EMA + WLS readout)",
    },
    "transformer": {
        "path": TF / "tf_sig15_l40_s43" / "best_model.pt",
        "n_params": 2_502_919,
        "n_tensors": 69,
        "note": "Transformer single-tower SOTA",
    },
}

SCORES = {
    "cnn_best_val_final": 0.687569327685616,
    "cnn_reference_val_final": 0.6584,
    "transformer_val_final": 0.6503,
    "frozen_method_test_final": 0.6696572892,
}

# Forward signatures, computed on CPU from a fixed synthetic batch with the
# canonical construction.  A checkpoint's weights are not enough to pin the
# function -- constructor arguments such as the dilated tower's dilation schedule
# never appear in the state dict -- so the output is hashed instead.
FORWARD_HASH = {
    "cnn_best": "3fe23ecde1e0a29d",
    "cnn_stage_a": "825b5a12d67a9c72",
    "cnn_reference": "faa8b552cfd48910",
    "transformer": "1d25d16bb272bece",
}

CATALOGUE = CNN / "r48sh_seed51_ema_ep8" / "r38_catalogs_seed51.npz"
CATALOGUE_FIELDS = {
    "E_E_TARGETID", "E_E_Z_QSO", "E_E_Z_DLA", "E_E_CONFIDENCE", "E_E_SNR", "E_E_LOG_NHI",
    "R38_TARGETID", "R38_Z_QSO", "R38_Z_DLA", "R38_CONFIDENCE", "R38_SNR", "R38_LOG_NHI",
}

MODULES = [
    "csst_dla.scoring", "csst_dla.snr", "csst_dla.targets",
    "csst_dla.fits_utils", "csst_dla.data",
    "data", "decode", "evaluate_hybrid", "model", "score_test", "tune_decode",
    "feature_fusion", "fuse_wzx_dilated",
    "predict_hybrid", "predict_feature_fusion",
    "models.dilated_resnet_5head",
    "models.transformer_5head",
    "models.transformer_conv_stem_5head",
    "models.transformer_conv_stem_rope_5head",
    "models.transformer_conv_stem_alibi_5head",
    "csst_dla_wzx_pkg.model",
    "csst_dla_wzx_pkg.inference",
]

# ---------------------------------------------------------------- machinery

_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            detail = fn()
            _RESULTS.append((name, True, "" if detail is None else str(detail)))
        except Exception as exc:  # noqa: BLE001
            _RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
            if VERBOSE:
                traceback.print_exc()
        return fn
    return deco


VERBOSE = False


def load_ckpt(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def state_of(ckpt: dict) -> dict:
    for key in ("model_state", "model_state_dict"):
        if key in ckpt:
            return ckpt[key]
    raise KeyError(f"no model_state in checkpoint (keys={sorted(ckpt)})")


def build_from_config(cfg: dict):
    """Rebuild the model described by a checkpoint's own ``config`` block."""
    arch = str(cfg.get("arch", "")).lower()
    if arch:
        from evaluate_hybrid import load_checkpoint  # noqa: F401  (kept for parity)
        from model import input_channels
        in_channels = int(cfg.get("in_channels", input_channels(cfg["input_mode"])))
        if arch == "transformer_conv_stem":
            from models.transformer_conv_stem_5head import _build_transformer_conv_stem_5head as builder
        elif arch == "transformer_conv_stem_rope":
            from models.transformer_conv_stem_rope_5head import _build_transformer_conv_stem_rope_5head as builder
        elif arch == "transformer_conv_stem_alibi":
            from models.transformer_conv_stem_alibi_5head import _build_transformer_conv_stem_alibi_5head as builder
        elif arch == "transformer":
            from models.transformer_5head import _build_transformer_5head as builder
        else:
            raise ValueError(f"unknown arch {arch!r}")
        return builder(
            in_channels=in_channels,
            d_model=int(cfg.get("d_model", 192)),
            nhead=int(cfg.get("nhead", 8)),
            num_layers=int(cfg.get("num_layers", 4)),
            dim_ff=int(cfg.get("dim_ff", 768)),
            dropout=float(cfg.get("dropout", 0.1)),
            conv_kernel=int(cfg.get("conv_kernel", 7)),
            num_conv_layers=int(cfg.get("num_conv_layers", 1)),
            use_offset=bool(cfg.get("use_offset", True)),
            max_len=int(cfg.get("max_len", 1024)),
        )

    # dual-tower fusion
    from model import HybridDlaNet, input_channels
    from csst_dla_wzx_pkg.model import create_model
    from csst_dla_wzx_pkg.data import feature_channels
    from feature_fusion import DualTowerFusionNet

    dc = cfg["dilated_config"]
    wc = cfg["wzx_config"]
    # Build the dilated tower through HybridDlaNet, exactly as the training code
    # does.  Its dilation schedule (stages=_stages_for(num_blocks)) and n_bins are
    # NOT part of the state dict, so building the raw module with the
    # _build_dilated_resnet_5head defaults yields identical weights but a
    # different network.
    dilated = HybridDlaNet(
        in_channels=input_channels(dc.get("input_mode", "flux")),
        hidden=int(dc.get("hidden", 96)),
        num_blocks=int(dc.get("num_blocks", 4)),
        with_offset=bool(dc.get("with_offset", True)),
        norm_type=str(dc.get("norm_type", "layer")),
        head_layers=int(dc.get("head_layers", 1)),
    ).model
    wzx = create_model(
        input_channels=int(wc.get("input_channels", feature_channels(wc.get("feature_mode", "all")))),
        base_channels=int(wc.get("base_channels", 96)),
        num_blocks=int(wc.get("num_blocks", 8)),
        dropout=float(wc.get("dropout", 0.1)),
        use_context_channels=bool(wc.get("use_context_channels", True)),
    )
    return DualTowerFusionNet(
        dilated,
        wzx,
        merge_mode=str(cfg["merge_mode"]),
        width=int(cfg.get("fusion_width", 128)),
        depth=int(cfg.get("fusion_depth", 3)),
        freeze_backbones=bool(cfg.get("freeze_backbones", True)),
    )


def forward_hash(net, cfg: dict) -> str:
    """Deterministic forward signature on a fixed synthetic batch (CPU, rounded).

    A strict state-dict match is necessary but NOT sufficient: constructor
    arguments that never reach the state dict -- the dilated tower's dilation
    schedule, for instance -- change the function while leaving the weights
    identical.  Hashing the output catches exactly that.
    """
    import hashlib
    import numpy as np
    import torch

    net = net.to("cpu").eval()
    g = torch.Generator().manual_seed(0)
    n_bins = 194
    with torch.no_grad():
        if hasattr(net, "wzx_backbone"):
            from model import input_channels
            from csst_dla_wzx_pkg.data import feature_channels
            dc = cfg["dilated_config"]
            wc = cfg["wzx_config"]
            h = torch.randn(2, input_channels(dc.get("input_mode", "flux")), n_bins, generator=g)
            w = torch.randn(
                2,
                int(wc.get("input_channels", feature_channels(wc.get("feature_mode", "all")))),
                n_bins,
                generator=g,
            )
            z = torch.tensor([1.5, 2.0])
            out = net(h, w, z)
        else:
            x = torch.randn(2, int(cfg.get("in_channels", 6)), n_bins, generator=g)
            out = net(x)
    blob = b"".join(np.round(out[k].numpy(), 5).tobytes() for k in sorted(out))
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------- checks

@check("imports")
def _imports():
    failed = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
    if failed:
        raise RuntimeError("failed imports: " + "; ".join(failed))
    return f"{len(MODULES)} modules"


@check("no_absolute_paths")
def _no_abs():
    import re
    # a machine path is `<slash><home|data|Users><slash><name>`; `<repo>/data/x`
    # in a docstring is a placeholder, not a machine path, hence the lookbehind.
    pattern = re.compile(r"(?<![\w>])/(?:home|data|Users)/[\w.\-]+")
    # Two directories are exempt on purpose:
    #   sota/drivers/  verbatim historical run records (they *should* keep the
    #                  paths of the machine they ran on)
    #   vendor/        third-party sources kept byte-identical to upstream
    exempt = ("sota/drivers/", "vendor/")
    hits = []
    for sub in ("hybrid_ensemble", "scripts", "src", "sota", "vendor"):
        for p in (ROOT / sub).rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            if rel.startswith(exempt):
                continue
            for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{rel}:{i}")
    if hits:
        raise RuntimeError("absolute paths found: " + ", ".join(hits[:10]))
    return "clean (sota/drivers/ and vendor/ exempt)"


@check("compile_all")
def _compile_all():
    import py_compile
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("hybrid_ensemble", "scripts", "src", "sota", "vendor"):
            for p in sorted((ROOT / sub).rglob("*.py")):
                out = Path(tmp) / f"{abs(hash(str(p)))}.pyc"
                try:
                    py_compile.compile(str(p), cfile=str(out), doraise=True)
                except Exception as exc:  # noqa: BLE001
                    bad.append(f"{p.relative_to(ROOT)} ({exc})")
    if bad:
        raise RuntimeError("compile failures: " + "; ".join(bad[:5]))
    return "all .py compile"


def _bound_names(tree):
    """Names bound anywhere in the module (imports, assignments, args, ...)."""
    import ast
    import builtins
    names = set(dir(builtins))
    names.update({"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "self", "cls"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.asname or a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                    names.add(arg.arg)
                if a.vararg:
                    names.add(a.vararg.arg)
                if a.kwarg:
                    names.add(a.kwarg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                names.add(arg.arg)
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            for t in ast.walk(node.optional_vars):
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


@check("no_unbound_names")
def _no_unbound():
    """Catch names used but never imported/assigned -- the failure mode of a
    mechanical path rewrite (e.g. `os.environ` in a file without `import os`)."""
    import ast
    bad = []
    for sub in ("hybrid_ensemble", "scripts", "src", "sota"):
        for p in sorted((ROOT / sub).rglob("*.py")):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            except SyntaxError as exc:
                bad.append(f"{p.relative_to(ROOT)}: SYNTAX {exc}")
                continue
            names = _bound_names(tree)
            for node in ast.walk(tree):
                hit = None
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id not in names:
                        hit = f"{node.lineno}: `{node.value.id}.`"
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in names:
                        hit = f"{node.lineno}: `{node.id}`"
                if hit:
                    bad.append(f"{p.relative_to(ROOT)} {hit}")
    if bad:
        raise RuntimeError("unbound names: " + "; ".join(bad[:8]))
    return "no unbound names"


def _ckpt_check(tag: str):
    ref = RECORDED[tag]

    @check(f"checkpoint:{tag}")
    def _inner():
        ck = load_ckpt(ref["path"])
        ms = state_of(ck)
        n_tensors = len(ms)
        n_params = sum(int(v.numel()) for v in ms.values() if hasattr(v, "numel"))
        assert n_tensors == ref["n_tensors"], f"n_tensors {n_tensors} != {ref['n_tensors']}"
        assert n_params == ref["n_params"], f"n_params {n_params} != {ref['n_params']}"
        for key in ("config", "threshold"):
            assert key in ck, f"missing top-level key {key!r}"
        return f"{n_tensors} tensors / {n_params} params / thr={ck['threshold']}"

    @check(f"rebuild+strictload:{tag}")
    def _inner2():
        import torch
        ck = load_ckpt(ref["path"])
        net = build_from_config(ck["config"])
        missing, unexpected = net.load_state_dict(state_of(ck), strict=False)
        assert not missing, f"missing keys: {list(missing)[:5]}"
        assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
        n_state = sum(int(v.numel()) for v in net.state_dict().values())
        return f"exact key match, {n_state} state entries (params+buffers)"

    @check(f"forward:{tag}")
    def _inner3():
        import torch
        ck = load_ckpt(ref["path"])
        net = build_from_config(ck["config"])
        net.load_state_dict(state_of(ck), strict=False)
        net.eval()
        g = torch.Generator().manual_seed(1)
        with torch.no_grad():
            if hasattr(net, "wzx_backbone"):
                out = net(torch.zeros(2, 6, 194), torch.zeros(2, 1, 194), torch.tensor([1.5, 2.0]))
            else:
                out = net(torch.zeros(2, int(ck["config"].get("in_channels", 6)), 194))
        return f"{sorted(out)} shapes={ {k: tuple(v.shape) for k, v in out.items()} }"

    @check(f"forward_hash:{tag}")
    def _inner4():
        ck = load_ckpt(ref["path"])
        net = build_from_config(ck["config"])
        net.load_state_dict(state_of(ck), strict=False)
        got = forward_hash(net, ck["config"])
        want = FORWARD_HASH[tag]
        assert got == want, (
            f"forward signature {got} != recorded {want} -- the rebuilt network is "
            f"not the checkpointed one (check constructor-only arguments)")
        return f"{got}"


for _tag in RECORDED:
    _ckpt_check(_tag)


@check("catalogue")
def _catalogue():
    import numpy as np
    z = np.load(CATALOGUE, allow_pickle=True)
    got = set(z.files)
    assert got == CATALOGUE_FIELDS, f"field mismatch: extra={got - CATALOGUE_FIELDS} missing={CATALOGUE_FIELDS - got}"
    n = int(z["R38_TARGETID"].size)
    assert n == 1653, f"R38_TARGETID size {n} != 1653"
    return f"{n} predictions, {len(got)} fields"


@check("runbook_paths")
def _runbook_paths():
    """Every `$REPO/...` file referenced by RUNBOOK.md must exist."""
    import re
    rb = ROOT / "RUNBOOK.md"
    if not rb.exists():
        return "no RUNBOOK.md"
    text = rb.read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"\$REPO/([A-Za-z0-9_./-]+\.(?:py|md|json|txt|npz))", text)))
    missing = [r for r in refs if not (ROOT / r).exists()]
    if missing:
        raise RuntimeError("RUNBOOK references missing files: " + ", ".join(missing))
    return f"{len(refs)} referenced paths exist"


# ---------------------------------------------------------------- optional rescore

def rescore(targets: str, train_fits: str, split: str, snr_field: str) -> str:
    import numpy as np
    import torch  # noqa: F401
    from csst_dla.scoring import labels_to_truth, score_catalog
    from data import HybridTrainDataset

    ds = HybridTrainDataset(targets, train_fits, split=split, input_mode="flux")
    truth = labels_to_truth(ds.labels, ds.indices, min_lognhi=20.3, snr_field=snr_field)
    z = np.load(CATALOGUE, allow_pickle=True)
    pred = {
        "TARGETID": z["R38_TARGETID"].astype(np.int64),
        "Z_QSO": z["R38_Z_QSO"].astype(np.float32),
        "Z_DLA": z["R38_Z_DLA"].astype(np.float32),
        "LOG_NHI": z["R38_LOG_NHI"].astype(np.float32),
        "SNR": z["R38_SNR"].astype(np.float32),
    }
    res = score_catalog(truth, pred)
    got = float(res.final_score)
    want = SCORES["cnn_best_val_final"]
    ok = abs(got - want) < 1e-6
    detail = f"re-scored VAL Final = {got!r} (recorded {want!r})"
    assert ok, detail
    return detail


# ---------------------------------------------------------------- main

def main() -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", help="targets .npz for the numeric re-score")
    ap.add_argument("--train-fits", help="training FITS for the numeric re-score")
    ap.add_argument("--split", default="val")
    ap.add_argument("--snr-field", default="SNR_GU")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    VERBOSE = args.verbose

    if args.targets and args.train_fits:
        @check("rescore:val")
        def _rescore():
            return rescore(args.targets, args.train_fits, args.split, args.snr_field)
        _rescore()
    elif args.targets or args.train_fits:
        print("!! --targets and --train-fits must be given together; skipping the numeric check")

    width = max(len(n) for n, _, _ in _RESULTS)
    n_fail = 0
    for name, ok, detail in _RESULTS:
        flag = "PASS" if ok else "FAIL"
        n_fail += 0 if ok else 1
        print(f"[{flag}] {name:<{width}}  {detail}")

    print()
    print(f"recorded scores: {json.dumps(SCORES, indent=None)}")
    print(f"{len(_RESULTS) - n_fail}/{len(_RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
