"""Phase 5: 3D inpainting / completion — fill a hole in a SLAT (training-free).

Loads the cached native SLAT (phase1_base.npz, the wooden chest), carves a hole (a voxel
bounding box — here a quarter wedge: the upper half along X and Z, full height), then
*completes* it with the two-stage RePaint inpainter:
    stage 1 regrows the missing occupancy (structure), stage 2 paints latents on the new voxels,
    everything outside the hole stays bit-exact.

Verification:
  * geometry: the hole (emptied to 0 voxels) is repopulated -> completed voxel count recovers
    toward the source; report how many voxels were regrown inside the hole.
  * preservation: every surviving voxel that reappears in the completed structure keeps its
    source latent bit-exact (max |Δ| == 0 on the survivor set).
  * visual: decode source / holed / completed to Gaussians (one at a time on 24GB) and render a
    3-panel comparison + a turntable of the completed asset.

Runs as one process (inpaint = stage-1 + stage-2 flow + the SS encoder/decoder + text encoder
≈ a generation footprint, fits 24GB; the three decodes happen afterward, one at a time, with the
flow models parked). Driven by scripts/run_phase5.sh.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import imageio
import numpy as np
import torch

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from slat_studio.io import load_slat, save_slat, read_extra
from slat_studio.inpainting import inpaint_slat, carve_hole, load_ss_encoder
from slat_studio.morph.interpolate import _lin

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
NPZ = os.path.join(OUT, "phase1_base.npz")
COMPLETED_NPZ = os.path.join(OUT, "phase5_completed.npz")
REPORT = os.path.join(OUT, "phase5_report.md")

SEED = 11
GRID = 64

assert os.path.exists(NPZ), f"missing {NPZ} — run examples/phase1_generate.py first"


def decode_gs(pipe, slat):
    return pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]


def render(gs, n=30):
    return np.stack(render_utils.render_video(gs, num_frames=n)["color"])


# ---------------------------------------------------------------- pipeline + SS encoder
print("[p5] loading TrellisTextTo3DPipeline + sparse-structure encoder...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
ss_encoder = load_ss_encoder(device="cuda")

# ---------------------------------------------------------------- source asset + hole box
source = load_slat(NPZ, device="cuda")
prompt = read_extra(NPZ).get("prompt", "<unknown>")
coords_xyz = source.coords[:, 1:].long()
mn = coords_xyz.min(0).values.tolist()
mx = coords_xyz.max(0).values.tolist()
midx = (mn[0] + mx[0] + 1) // 2
midz = (mn[2] + mx[2] + 1) // 2
# quarter wedge: upper half along X and Z, full extent along Y (vertical)
hole_bbox = (midx, mn[1], midz, mx[0] + 1, mx[1] + 1, mx[2] + 1)
print(f"[p5] source: {source.coords.shape[0]} voxels, prompt={prompt!r}")
print(f"[p5] voxel range min={mn} max={mx}; hole bbox (upper X&Z wedge)={hole_bbox}")

holed, n_removed = carve_hole(source, hole_bbox)
print(f"[p5] carved hole: removed {n_removed} voxels -> holed asset has {holed.coords.shape[0]}")

# ---------------------------------------------------------------- inpaint (no decode yet)
print("[p5] inpainting (stage-1 structure RePaint -> stage-2 latent RePaint)...")
completed, _, info = inpaint_slat(
    pipe, holed, hole_bbox, prompt, ss_encoder, seed=SEED, formats=(),
)
print(f"[p5] info: {info}")

# stages done — park the heavy models so the three GS decodes get the whole card
for n in ("sparse_structure_flow_model", "slat_flow_model", "slat_decoder_mesh", "slat_decoder_rf"):
    pipe.models[n].cpu()
pipe.text_cond_model["model"].cpu()
ss_encoder.cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- preservation check (survivors bit-exact)
res = GRID
key_comp = _lin(completed.coords, res)
key_src = _lin(source.coords, res)
in_hole_src = (
    (coords_xyz[:, 0] >= hole_bbox[0]) & (coords_xyz[:, 0] < hole_bbox[3]) &
    (coords_xyz[:, 1] >= hole_bbox[1]) & (coords_xyz[:, 1] < hole_bbox[4]) &
    (coords_xyz[:, 2] >= hole_bbox[2]) & (coords_xyz[:, 2] < hole_bbox[5])
)
survivor_keys = key_src[~in_hole_src]
is_survivor = torch.isin(key_comp, survivor_keys)
# gather source latents for the survivor voxels that reappear in the completed structure
order = torch.argsort(key_src)
idx = order[torch.searchsorted(key_src[order], key_comp[is_survivor])]
surv_max = (completed.feats[is_survivor] - source.feats[idx]).abs().max().item()

comp_xyz = completed.coords[:, 1:].long()
in_hole_comp = (
    (comp_xyz[:, 0] >= hole_bbox[0]) & (comp_xyz[:, 0] < hole_bbox[3]) &
    (comp_xyz[:, 1] >= hole_bbox[1]) & (comp_xyz[:, 1] < hole_bbox[4]) &
    (comp_xyz[:, 2] >= hole_bbox[2]) & (comp_xyz[:, 2] < hole_bbox[5])
)
n_filled = int(in_hole_comp.sum().item())
print(f"[p5] survivors that reappear: {int(is_survivor.sum())} | survivor latent max |Δ|={surv_max:.2e}")
print(f"[p5] voxels regrown inside the hole: {n_filled} (was emptied to 0)")
assert surv_max == 0.0, "surviving voxels' latents must be preserved bit-exact"
assert n_filled > 0, "the hole should be repopulated with new structure"

save_slat(completed, COMPLETED_NPZ, extra={"prompt": prompt, "hole_bbox": list(hole_bbox), **info})

# ---------------------------------------------------------------- decode + render (one at a time)
print("[p5] decoding + rendering source...")
src_vid = render(decode_gs(pipe, source)); torch.cuda.empty_cache()
print("[p5] decoding + rendering holed...")
hol_vid = render(decode_gs(pipe, holed)); torch.cuda.empty_cache()
print("[p5] decoding + rendering completed...")
comp_gs = decode_gs(pipe, completed)
comp_vid = render(comp_gs); torch.cuda.empty_cache()

# brightest shared view for the still comparison
v0 = int(src_vid.reshape(src_vid.shape[0], -1).mean(axis=1).argmax())
compare = np.concatenate([src_vid[v0], hol_vid[v0], comp_vid[v0]], axis=1)  # source | holed | completed
imageio.imwrite(os.path.join(OUT, "phase5_compare.png"), compare)
imageio.mimsave(os.path.join(OUT, "phase5_completed.mp4"), list(comp_vid), fps=30)

with open(REPORT, "w") as f:
    f.write("# Phase 5 — 3D inpainting / completion (two-stage RePaint, training-free)\n\n")
    f.write(f"Source asset prompt: {prompt!r}\n\n")
    f.write(f"- asset voxel range: min {mn}, max {mx}\n")
    f.write(f"- hole bbox (voxel idx, upper-X&Z quarter wedge): {list(hole_bbox)}\n")
    f.write(f"- voxels removed to make the hole: **{n_removed}** "
            f"(holed asset: {holed.coords.shape[0]} voxels)\n\n")
    f.write("## Completion result\n\n")
    f.write(f"- source voxels: **{info['n_source']}**\n")
    f.write(f"- completed voxels: **{info['n_completed']}** "
            f"(kept {info['n_kept_in_completed']} survivors + grew {info['n_grown']} new)\n")
    f.write(f"- voxels regrown *inside* the hole: **{n_filled}** (the hole was emptied to 0)\n\n")
    f.write("## Preservation (everything outside the hole)\n\n")
    f.write(f"- surviving voxels reappearing in the completed structure: "
            f"**{int(is_survivor.sum())}**\n")
    f.write(f"- their latent max |Δ| vs source: **{surv_max:.2e}** "
            f"(bit-exact by construction — stage-2 composites source latents back on survivors)\n\n")
    f.write("## How it works\n\n")
    f.write("- **Stage 1 (structure):** the holed occupancy is encoded by the sparse-structure "
            "VAE encoder; `DenseRepaintFlowSampler` regenerates only the hole's latent cells "
            "while pinning the rest, then the SS decoder yields a *filled* occupancy.\n")
    f.write("- **Stage 2 (latents):** on the new coords, survivor voxels keep their bit-exact "
            "source latent; the regrown voxels are RePaint-sampled by the Phase-3 "
            "`RepaintFlowSampler`.\n")
    f.write("- No training; nothing under `third_party/TRELLIS` is modified.\n\n")
    f.write("Artifacts: `phase5_compare.png` (source | holed | completed, brightest view), "
            "`phase5_completed.mp4`, `phase5_completed.npz`.\n")

print(f"[p5] wrote {os.path.basename(REPORT)} / phase5_compare.png / phase5_completed.mp4")
print("[p5] DONE")
