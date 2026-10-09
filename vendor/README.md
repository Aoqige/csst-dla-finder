# `vendor/` — third-party sources bundled for self-containment

## `csst_dla_wzx_pkg/`

The WZX ("FlatNet") tower package. `hybrid_ensemble/feature_fusion.py` builds
`DualTowerFusionNet` from two backbones — a dilated CNN backbone from this
repository plus this WZX backbone — so the fusion checkpoints under `results/`
cannot be rebuilt without it.

- **Origin**: upstream branch `network/wzx` of `Aoqige/csst-dla-finder`, commit
  `87be853` ("Add files via upload"). The reference runs imported it from
  `~/csst_dla_wzx_pkg`, a symlink to a checkout of that branch.
- **Licence**: MIT (see `csst_dla_wzx_pkg/LICENSE`), retained verbatim.
- **Changes**: none. The files are byte-identical to the origin; only
  `__pycache__/` was dropped.

Importing it now only requires `<repo>/vendor` on `sys.path`, which
`hybrid_ensemble/{train,predict}_feature_fusion.py` and `fuse_wzx_dilated.py`
already arrange. The historical `sys.path.insert(0, "/home/heruihua")` hack that
resolved the symlink is gone.
