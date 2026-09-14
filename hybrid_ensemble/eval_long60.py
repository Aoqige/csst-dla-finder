"""Does over-training (60 epochs) help?  Evaluate transformer_v3c_long60 on test.

Compares, under an identical harness:
  * v3c  (10 epochs, the current best single tower, test Final 0.4980)
  * long60 (60 epochs, otherwise identical config)
first as single members (count_bias 0 and 0.75), then by swapping long60 in for
v3c inside the 4-way union ensemble (current SOTA 0.5455).

Run:  CUDA_VISIBLE_DEVICES=2 /home/dingjch/anaconda3/envs/ML_env/bin/python eval_long60.py
"""
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HB = os.path.expanduser("~/csst-dla-finder/hybrid_ensemble")
sys.path.insert(0, HB)
sys.path.insert(0, os.path.expanduser("~/csst-dla-finder/src"))

from data import HybridTestDataset                          # noqa: E402
from decode import predict_member, softmax                  # noqa: E402
from evaluate_hybrid import load_checkpoint, resolve_device  # noqa: E402
from predict_hybrid import write_submission                 # noqa: E402
import score_test                                           # noqa: E402
from csst_dla.scoring import score_catalog                  # noqa: E402

PY = "/home/dingjch/anaconda3/envs/ML_env/bin/python"
TEST_FITS = "/data/aoqige/test.fits"
TRUTH = "/data/aoqige/test_truth.fits"
MERGE = os.path.join(HB, "merge_predictions.py")
RUNS = os.path.expanduser("~/csst_dla_runs")

SOTA_CSV = os.path.join(RUNS, "20260904/feature_no_clean_flux_context_v1/predictions.csv")

MEMBERS = [
    ("v3c",    os.path.join(RUNS, "20260911/transformer_v3c_lognhi/best_model.pt"), "short10"),
    ("long60", os.path.join(RUNS, "20260912/transformer_v3c_long60/best_model.pt"), "long60"),
    ("v5",     os.path.join(RUNS, "20260911/transformer_conv_stem_v5_rope/best_model.pt"), "short10"),
    ("v6",     os.path.join(RUNS, "20260912/transformer_conv_stem_v6_alibi/best_model.pt"), "short10"),
]
BIAS = {"v3c": [0.0, 0.75, 0.75], "long60": [0.0, 0.75, 0.75],
        "v5": [0.0, 0.75, 0.75], "v6": [0.0, 1.5, 1.5]}
DECODE = {"threshold": 0.45, "min_distance": 10, "min_z_dla": 1.10,
          "use_offset": True, "lognhi_clip_min": 19.5, "lognhi_clip_max": 22.5}
DV_MERGE = 1500.0
WORK = "/tmp/eval_long60"


def merge(a, b, out):
    subprocess.run([PY, MERGE, a, b, out, "union", str(DV_MERGE)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def score_csv(path, truth, snr_by_target):
    catalog = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if catalog.ndim == 0:
        catalog = np.asarray([catalog])
    out = {"TARGETID": [], "Z_DLA": [], "LOG_NHI": [], "SNR": []}
    for row in catalog:
        tid = int(row["id"])
        for slot in (1, 2):
            z = float(row["Z_DLA%d" % slot]); ln = float(row["LOGNHI%d" % slot])
            if z <= 0 or ln < 20.3:
                continue
            if not (2550.0 <= 1215.67 * (1 + z) < 4200.0):
                continue
            out["TARGETID"].append(tid); out["Z_DLA"].append(z)
            out["LOG_NHI"].append(ln); out["SNR"].append(snr_by_target.get(tid, np.nan))
    out["TARGETID"] = np.asarray(out["TARGETID"], np.int64)
    for k in ("Z_DLA", "LOG_NHI", "SNR"):
        out[k] = np.asarray(out[k], np.float32)
    return score_catalog(truth, out)


def report(label, r, ref=None):
    tail = "" if ref is None else "   (ref %.4f, delta %+.4f)" % (ref, r.final_score - ref)
    print("%-42s Final=%.4f Det=%.4f Param=%.4f compl=%.4f pur=%.4f n_pred=%5d n_match=%4d%s"
          % (label, r.final_score, r.detection_score, r.parameter_score,
             r.completeness, r.purity, r.n_pred, r.n_match, tail), flush=True)


def main():
    os.makedirs(WORK, exist_ok=True)
    device = resolve_device("cuda")
    cache = {}
    for name, ckpt, tag in MEMBERS:
        model, cfg = load_checkpoint(ckpt, device)
        ds = HybridTestDataset(TEST_FITS, cfg["input_mode"])
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        pred = predict_member(model, loader, device)
        cache[name] = (pred, ds, cfg, tag)
        ck = torch.load(ckpt, map_location="cpu")
        print("[loaded] %-6s %-8s %s  (saved epoch %s, val Final %.4f, params %.2fM)"
              % (name, tag, os.path.basename(os.path.dirname(ckpt)),
                 ck.get("score", {}).get("epoch", "?"),
                 ck.get("score", {}).get("final_score", float("nan")),
                 sum(p.numel() for p in model.state_dict().values()) / 1e6), flush=True)
        cp = softmax(pred["count_logits"])
        pick = np.argmax(cp, axis=1)
        mx = pred["heatmap"].max(axis=1)
        print("          gate: argmax(count)=0 %.1f%% | max-heatmap>=0.45 %.1f%% | "
              "in[0.20,0.45) %.1f%% | mean max-hm %.4f"
              % (100 * (pick == 0).mean(), 100 * (mx >= 0.45).mean(),
                 100 * ((mx >= 0.20) & (mx < 0.45)).mean(), mx.mean()), flush=True)
        del model
        torch.cuda.empty_cache()

    truth = score_test.load_truth(TRUTH)
    snr_by_target = score_test.load_snr_by_target(TRUTH, "SNR_GU")

    def emit(name, bias, tag=""):
        pred, ds, cfg, _ = cache[name]
        c = dict(cfg); c.update(DECODE); c["count_bias"] = bias
        p = os.path.join(WORK, "%s_b%s%s.csv" % (name, bias[1], tag))
        write_submission(p, ds, pred, c)
        return p

    print("\n=== single members (test set, 20k) ===")
    singles = {}
    for name in ("v3c", "long60"):
        for bias, bl in (([0.0, 0.0, 0.0], "bias0"), (BIAS[name], "bias.75")):
            p = emit(name, bias)
            r = score_csv(p, truth, snr_by_target)
            singles[(name, bl)] = p
            ref = 0.4980 if (name == "v3c" and bl == "bias0") else None
            report("%s single  %s" % (name, bl), r, ref)

    print("\n=== 4-way union: long60 swapped in for v3c ===")
    v5 = emit("v5", BIAS["v5"], "_ens")
    v6 = emit("v6", BIAS["v6"], "_ens")
    for label, member in (("union(v3c,v5,v6,SOTA)  baseline", "v3c"),
                          ("union(long60,v5,v6,SOTA) swap", "long60")):
        a = singles[(member, "bias.75")]
        ch1 = os.path.join(WORK, "u1_%s.csv" % member); merge(a, v5, ch1)
        ch2 = os.path.join(WORK, "u2_%s.csv" % member); merge(ch1, v6, ch2)
        ch3 = os.path.join(WORK, "u3_%s.csv" % member); merge(ch2, SOTA_CSV, ch3)
        r = score_csv(ch3, truth, snr_by_target)
        report(label, r, 0.5455)
        # also: union of BOTH v3c and long60 (5-way)
        ch4 = os.path.join(WORK, "u4_%s.csv" % member); merge(ch3, singles[("v3c" if member == "long60" else "long60", "bias.75")], ch4)
        r5 = score_csv(ch4, truth, snr_by_target)
        report("   + also adding the other tower (5-way)", r5, 0.5455)

    print("\ndone. csvs in %s" % WORK)


if __name__ == "__main__":
    main()
