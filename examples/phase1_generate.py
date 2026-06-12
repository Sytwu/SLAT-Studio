"""Phase 1, step 1/2: generate a TRELLIS-native asset and CACHE its SLAT.

Generates an asset from BASE_PROMPT, saves its structured latent to outputs/phase1_base.npz
(verifying a bit-exact .npz round-trip), and renders the base asset. Run before
phase1_restyle.py (a separate process so each stays within a single-generation memory
footprint on a 24GB card). Driven by scripts/run_phase1.sh.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
import torch

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from slat_studio.pipelines import text_to_slat
from slat_studio.io import save_slat, load_slat

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT, exist_ok=True)

BASE_PROMPT = "A rustic wooden treasure chest with iron bands."
SEED = 1
NPZ = os.path.join(OUT, "phase1_base.npz")

print("[p1-gen] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()

print(f"[p1-gen] generating base asset: {BASE_PROMPT!r}")
slat, out = text_to_slat(pipe, BASE_PROMPT, seed=SEED, formats=("gaussian",))
print(f"[p1-gen] base SLAT: {slat.coords.shape[0]} voxels, {slat.feats.shape[1]} channels")

# cache + round-trip check
save_slat(slat, NPZ, extra={"prompt": BASE_PROMPT, "seed": SEED})
rt = load_slat(NPZ, device="cuda")
rt_ok = torch.equal(rt.coords, slat.coords) and torch.equal(rt.feats, slat.feats)
print(f"[p1-gen] cached {NPZ}; round-trip bit-exact: {rt_ok}")
assert rt_ok, "SLAT round-trip mismatch"

# render base asset
vid = render_utils.render_video(out["gaussian"][0], num_frames=30)["color"]
imageio.mimsave(os.path.join(OUT, "phase1_base.mp4"), vid, fps=30)
imageio.imwrite(os.path.join(OUT, "phase1_base_frame.png"), vid[0])
print("[p1-gen] wrote phase1_base.mp4 / phase1_base_frame.png")
print("[p1-gen] DONE")
