# Transformer five-head DLA models — reference results

Per-pixel Transformer encoders that share the **same five-head output contract** as the
dilated-ResNet baseline, so they can be swapped into `decode.py`,
`average_predictions()` and the offline tensor cache without any change downstream.

Every architecture in this repo exports the same four tensors:
`{heatmap, lognhi, count_logits, offset}`.

## Architectures

Shared backbone: `Conv1d` stem (k=7, 3 layers) → positions →
4 × `TransformerEncoderLayer` (d_model 192, 8 heads, dim_ff 768, dropout 0.1),
`max_len` 1024. Variants differ only in the positional scheme and stem wiring.

| `--arch` | Module | Positional scheme | Params | Test Final |
|---|---|---|---|---|
| `transformer` | `models/transformer_5head.py` | learned absolute (v0) | 2.50M | 0.3779 |
| `transformer_conv_stem` | `models/transformer_conv_stem_5head.py` | learned absolute (v3c) | 2,502,919 | 0.4980 |
| `transformer_conv_stem_rope` | `models/transformer_conv_stem_rope_5head.py` | RoPE (v5) | 2,306,311 | 0.4919 |
| `transformer_conv_stem_alibi` | `models/transformer_conv_stem_alibi_5head.py` | ALiBi (v6) | 2,306,311 | 0.4698 |

Learned absolute positions come out ahead of both RoPE and ALiBi at this scale. A
larger ~12.1M-parameter variant was tried as well and scored worse.

## Reference checkpoints

`checkpoints/*.pt` each hold `{model_state, config, threshold, score, training_config}`.

Scores are `score_20p3` on the 20 000-spectrum test set (1644 scoreable truth lines,
`logNHI >= 20.3`); `Final = 0.6·Det + 0.4·Param`.

| File | Backbone | Params | Size | Final | Det | Param | Compl | Pur | n_pred | n_match |
|---|---|---|---|---|---|---|---|---|---|---|
| `v3c_lognhi.pt` | `transformer_conv_stem` | 2,502,919 | 9.6M | **0.49798** | 0.48350 | 0.51970 | 0.43917 | 0.85545 | 844 | 722 |
| `v5_rope.pt` | `transformer_conv_stem_rope` | 2,306,311 | 8.9M | 0.49190 | 0.46873 | 0.52666 | 0.40633 | 0.89785 | 744 | 668 |
| `v6_alibi.pt` | `transformer_conv_stem_alibi` | 2,306,311 | 8.9M | 0.46984 | 0.44478 | 0.50741 | 0.37956 | 0.88136 | 708 | 624 |
| `cnn_sota.pt` | dilated-ResNet fusion head | 3,555,527 | 14M | 0.49555 | 0.48154 | 0.51657 | 0.42579 | 0.88384 | 792 | 700 |

`metrics/<name>/` carries the per-variant `training_args.json`, `history.json`,
`ensemble_config.json` and the full `score_20p3.json` (bin-level detail included).

## Cross-architecture fusion

The strongest configuration found so far averages the exported tensors of three
*towers* — `v3c`, `v5_rope` and the CNN `cnn_sota` — and adds a count bias of 1.25.
No training is involved; the members are simply averaged tensor-wise.

| | Final | Det | Param | Compl | Pur | n_pred | n_match |
|---|---|---|---|---|---|---|---|
| `avg(v3c, v5_rope, cnn_sota) + bias 1.25` | **0.55238** | 0.58890 | 0.49759 | 0.57299 | 0.72685 | 1296 | 942 |

The gain comes from the **architecture axis only**. Averaging two seeds of the same
backbone, two loss variants, or two input-channel variants of one backbone all land at
or below the single-model score — so the diversity that pays off here is architectural,
not merely an ensembling effect.

## Reproduce

```bash
python hybrid_ensemble/train_transformer.py \
  --arch transformer_conv_stem \
  --input-mode flux \
  --epochs 10 --lr 1e-3 --lr-schedule cosine \
  --batch-size 512 --seed 42 \
  --lognhi-loss-weight 0.15 \
  --targets  <cnn_targets_seed42.npz> \
  --train-fits <train_5e5.fits> \
  --out-dir  <run dir>
```

`--count-loss-type {ce,focal}` (default `ce`) and `--count-focal-gamma` are available as
an extra knob; the reference numbers above were all produced with the default `ce`.

Re-score a stored prediction file:

```bash
python hybrid_ensemble/score_test.py \
  --predictions <pred.csv> --truth <test_truth.fits> --out <score.json>
```

## Caveat on the v3c ↔ v5/v6 comparison

`v3c` was trained with `--lognhi-loss-weight 0.15`, whereas `v5_rope` and `v6_alibi`
used `0.05`. The architecture ranking above therefore also carries a loss-weight
change and should not be read as a clean positional-encoding comparison.
