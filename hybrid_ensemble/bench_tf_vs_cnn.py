"""Wall-clock and VRAM benchmark: conv-stem Transformer vs dilated-resnet CNN.

Both were trained under identical conditions:
  batch_size=512, FP32, no AMP, num_workers=0, seq len 681, 6 input channels.
This script reproduces that step (forward + backward + AdamW) and reports
seconds/step and peak allocated VRAM, then probes how far the batch can scale.
"""
import gc
import json
import time

import torch

torch.manual_seed(0)
DEV = "cuda"
L = 681
IN_CH = 6


def build_cnn():
    from model import HybridDlaNet
    return HybridDlaNet(IN_CH, hidden=96, num_blocks=4, with_offset=True,
                        norm_type="layer", head_layers=1)


def build_tf():
    from models.transformer_conv_stem_5head import _build_transformer_conv_stem_5head
    return _build_transformer_conv_stem_5head(
        in_channels=IN_CH, d_model=192, nhead=8, num_layers=4,
        dim_ff=768, dropout=0.1, conv_kernel=7, num_conv_layers=3)


def synthetic_loss(out):
    total = None
    for _k, v in out.items():
        if not torch.is_tensor(v):
            continue
        term = v.float().mean()
        total = term if total is None else total + term
    return total


def measure(builder, batch, warmup=3, iters=8):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = builder().to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.randn(batch, IN_CH, L, device=DEV)

        for _ in range(warmup):
            opt.zero_grad(set_to_none=True)
            synthetic_loss(model(x)).backward()
            opt.step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            opt.zero_grad(set_to_none=True)
            synthetic_loss(model(x)).backward()
            opt.step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

        train_peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        n_par = sum(p.numel() for p in model.parameters())

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        infer_peak = torch.cuda.max_memory_allocated() / 1024 ** 3

        del model, opt, x
        torch.cuda.empty_cache()
        return {"batch": batch, "params": n_par,
                "sec_per_step": round(dt, 4),
                "train_peak_gb": round(train_peak, 2),
                "infer_peak_gb": round(infer_peak, 2),
                "samples_per_sec": round(batch / dt, 1)}
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        return {"batch": batch, "oom": True}


def main():
    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    batches = [512, 1024, 2048, 4096, 8192, 16384]
    results = {}
    for name, builder in (("CNN_dilated", build_cnn), ("TF_conv_stem", build_tf)):
        print("=" * 24, name)
        rows = []
        for b in batches:
            row = measure(builder, b)
            rows.append(row)
            print("   ", json.dumps(row), flush=True)
            if row.get("oom"):
                break
        results[name] = rows
    print("DONE")


if __name__ == "__main__":
    main()
