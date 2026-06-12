"""Cheap, standalone round-trip test for slat_studio.io (no pipeline weights needed).

Builds a small synthetic SLAT SparseTensor, saves it, reloads it, and asserts the coords
and feats come back bit-for-bit identical. Run via the trellis env (needs spconv + a GPU):

    PYTHONPATH=.../third_party/TRELLIS python scripts/test_slat_io.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trellis.modules import sparse as sp  # noqa: E402
from slat_studio.io import save_slat, load_slat, read_extra  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, "io_roundtrip.npz")

torch.manual_seed(0)
N, C = 137, 8
xyz = torch.randint(0, 64, (N, 3), dtype=torch.int32)
batch = torch.zeros(N, 1, dtype=torch.int32)
coords = torch.cat([batch, xyz], dim=1).cuda()        # [N, 4]
feats = torch.randn(N, C, dtype=torch.float32).cuda()  # [N, C]
slat = sp.SparseTensor(feats=feats, coords=coords)

save_slat(slat, path, extra={"prompt": "unit-test", "seed": 0})
print(f"[io] wrote {path} ({os.path.getsize(path)} bytes)")

reloaded = load_slat(path, device="cuda")
ok_coords = torch.equal(reloaded.coords, slat.coords)
ok_feats = torch.equal(reloaded.feats, slat.feats)
extra = read_extra(path)

print(f"[io] coords match: {ok_coords} | feats match: {ok_feats} | extra: {extra}")
assert ok_coords and ok_feats, "SLAT round-trip mismatch!"
assert extra.get("prompt") == "unit-test"
print("[io] PASS")
