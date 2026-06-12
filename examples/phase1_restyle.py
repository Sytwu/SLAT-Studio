"""Phase 1, step 2/2: restyle a CACHED SLAT (freeze structure, new prompt).

Loads the SLAT cached by phase1_generate.py, keeps its structure {p_i} fixed, and re-runs
stage-2 conditioned on RESTYLE_PROMPT to get a new appearance. Verifies the structure is
preserved exactly (coords identical) while appearance changes (feats differ), then renders
the restyled asset and a side-by-side comparison against the base.

Runs as its own process (after phase1_generate.py) so only one generation's worth of memory
is live on a 24GB card. Driven by scripts/run_phase1.sh.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
import numpy as np
import torch

from trellis.pipelines import TrellisTextTo3DPipeline

from slat_studio.io import load_slat, read_extra
from slat_studio.style import restyle

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
NPZ = os.path.join(OUT, "phase1_base.npz")

RESTYLE_PROMPT = "A treasure chest made of polished gold inlaid with emerald gemstones."
SEED = 1

assert os.path.exists(NPZ), f"missing {NPZ} — run examples/phase1_generate.py first"

print("[p1-restyle] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
# restyle needs only stage-2 (slat_flow) + the gaussian decoder + text encoder. Park the big
# unused stage-1 flow and the mesh/rf decoders on CPU. Keep sparse_structure_decoder (the
# first model in the dict) on GPU: pipeline.device reports its device and sample_slat places
# its noise there. (It is tiny.)
for n in ("sparse_structure_flow_model", "slat_decoder_mesh", "slat_decoder_rf"):
    pipe.models[n].cpu()
torch.cuda.empty_cache()

# load the cached base SLAT
base = load_slat(NPZ, device="cuda")
base_prompt = read_extra(NPZ).get("prompt", "<unknown>")
print(f"[p1-restyle] loaded base SLAT ({base.coords.shape[0]} voxels), base prompt: {base_prompt!r}")

# restyle: freeze structure, new prompt
print(f"[p1-restyle] restyling with: {RESTYLE_PROMPT!r}")
slat2, out2 = restyle(pipe, base, RESTYLE_PROMPT, seed=SEED, formats=("gaussian",))

# verify structure preserved, appearance changed
coords_same = torch.equal(slat2.coords, base.coords)
feat_l2 = (slat2.feats - base.feats).norm(dim=1).mean().item()
print(f"[p1-restyle] structure preserved (coords identical): {coords_same}")
print(f"[p1-restyle] appearance changed (mean per-voxel |Δfeat|): {feat_l2:.4f}")
assert coords_same, "structure must be identical after restyle"
assert feat_l2 > 1e-3, "appearance should change after restyle"

# render restyled + side-by-side comparison vs base
from trellis.utils import render_utils
rs_vid = render_utils.render_video(out2["gaussian"][0], num_frames=30)["color"]
imageio.mimsave(os.path.join(OUT, "phase1_restyled.mp4"), rs_vid, fps=30)

base_frame_path = os.path.join(OUT, "phase1_base_frame.png")
if os.path.exists(base_frame_path):
    base_frame = imageio.imread(base_frame_path)
    compare = np.concatenate([base_frame, rs_vid[0]], axis=1)
    imageio.imwrite(os.path.join(OUT, "phase1_compare.png"), compare)
    print("[p1-restyle] wrote phase1_restyled.mp4 / phase1_compare.png")
else:
    print("[p1-restyle] wrote phase1_restyled.mp4 (no base frame for comparison)")
print("[p1-restyle] DONE")
