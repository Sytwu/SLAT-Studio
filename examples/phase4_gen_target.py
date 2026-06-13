"""Phase 4, step 1/2: generate the morph TARGET asset and CACHE its SLAT.

The morph SOURCE is the Phase 1 wooden chest (outputs/phase1_base.npz). Here we generate a
structurally *different* asset (a teapot — rounded silhouette vs. the chest's box) so the
morph genuinely exercises sparse-voxel structure correspondence, and cache its SLAT to
outputs/phase4_target.npz. Run as its own process (a full text->3D generation + the gaussian
renderer exceed 24GB if shared). Driven by scripts/run_phase4.sh.
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

TARGET_PROMPT = "A round ceramic teapot with a curved spout and a lid."
SEED = 2
NPZ = os.path.join(OUT, "phase4_target.npz")

print("[p4-gen] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()

print(f"[p4-gen] generating target asset: {TARGET_PROMPT!r}")
slat, out = text_to_slat(pipe, TARGET_PROMPT, seed=SEED, formats=("gaussian",))
print(f"[p4-gen] target SLAT: {slat.coords.shape[0]} voxels, {slat.feats.shape[1]} channels")

save_slat(slat, NPZ, extra={"prompt": TARGET_PROMPT, "seed": SEED})
rt = load_slat(NPZ, device="cuda")
rt_ok = torch.equal(rt.coords, slat.coords) and torch.equal(rt.feats, slat.feats)
print(f"[p4-gen] cached {NPZ}; round-trip bit-exact: {rt_ok}")
assert rt_ok, "SLAT round-trip mismatch"

vid = render_utils.render_video(out["gaussian"][0], num_frames=30)["color"]
imageio.mimsave(os.path.join(OUT, "phase4_target.mp4"), vid, fps=30)
imageio.imwrite(os.path.join(OUT, "phase4_target_frame.png"), vid[0])
print("[p4-gen] wrote phase4_target.mp4 / phase4_target_frame.png")
print("[p4-gen] DONE")
