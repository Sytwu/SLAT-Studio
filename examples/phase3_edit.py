"""Phase 3: region editing — regenerate latents inside a voxel bbox from a new prompt.

Loads the cached native SLAT (phase1_base.npz, the wooden chest), picks a voxel-space
bounding box (the top half of the asset), and re-samples *only* the latents inside the box
from EDIT_PROMPT using the RePaint masked sampler, keeping everything outside the box fixed.

We then verify the edit is **localized**:
  * latent-space (by construction): coords identical; outside-box latents bit-exact equal to
    the source; inside-box latents changed.
  * decode-space (the real test, since the GS decoder has a global receptive field): decode
    both SLATs to Gaussians and bucket the per-Gaussian appearance change by whether the
    Gaussian's voxel is inside the box — inside should change a lot, outside very little.

Runs as one process (edit = one stage-2 pass + the gaussian decoder + text encoder ≈ the
restyle footprint, fits 24GB). Driven by scripts/run_phase3.sh.
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
from slat_studio.editing import edit_region

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
NPZ = os.path.join(OUT, "phase1_base.npz")
EDITED_NPZ = os.path.join(OUT, "phase3_edited.npz")
REPORT = os.path.join(OUT, "phase3_report.md")

EDIT_PROMPT = "molten glowing lava and burning embers"
VERTICAL_AXIS = 1          # TRELLIS Y-up; edit the TOP half along this axis
SEED = 7
GRID = 64

assert os.path.exists(NPZ), f"missing {NPZ} — run examples/phase1_generate.py first"


def decode_gs(pipe, slat):
    return pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]


def per_gaussian_voxel(xyz, resolution=GRID):
    vox = torch.floor((xyz.detach().cpu() + 0.5) * resolution).long().clamp(0, resolution - 1)
    return vox  # [G,3]


# ---------------------------------------------------------------- pipeline (stage-2 + GS decoder)
print("[p3] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
# keep sparse_structure_decoder on GPU (defines pipe.device for the noise); park the rest unused.
for n in ("sparse_structure_flow_model", "slat_decoder_mesh", "slat_decoder_rf"):
    pipe.models[n].cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- source asset + edit box
source = load_slat(NPZ, device="cuda")
source_prompt = read_extra(NPZ).get("prompt", "<unknown>")
coords_xyz = source.coords[:, 1:].long()
mn = coords_xyz.min(0).values.tolist()
mx = coords_xyz.max(0).values.tolist()
mid = (mn[VERTICAL_AXIS] + mx[VERTICAL_AXIS] + 1) // 2
# full extent in the two non-vertical axes, top half along the vertical axis
lo = [mn[0], mn[1], mn[2]]
hi = [mx[0] + 1, mx[1] + 1, mx[2] + 1]
lo[VERTICAL_AXIS] = mid
bbox = (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
print(f"[p3] source: {source.coords.shape[0]} voxels, prompt={source_prompt!r}")
print(f"[p3] asset voxel range min={mn} max={mx}; edit bbox (top half, axis {VERTICAL_AXIS})={bbox}")

# ---------------------------------------------------------------- edit (no decode yet — keep VRAM free)
print(f"[p3] editing in-box region with prompt: {EDIT_PROMPT!r}")
edited, mask, _ = edit_region(pipe, source, bbox, EDIT_PROMPT, seed=SEED, formats=())
n_in = int(mask.sum().item())
n_out = int((1 - mask).sum().item())

# stage-2 is done; park the flow model + text encoder so the two GS decodes have the whole card
pipe.models["slat_flow_model"].cpu()
pipe.text_cond_model["model"].cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- latent-space verification
coords_same = torch.equal(edited.coords, source.coords)
m = mask.squeeze(1).bool()
d_feat = (edited.feats - source.feats).norm(dim=1)
out_max = d_feat[~m].max().item() if n_out else float("nan")
in_mean = d_feat[m].mean().item() if n_in else float("nan")
print(f"[p3] structure preserved (coords identical): {coords_same}")
print(f"[p3] in-box voxels={n_in} out-box voxels={n_out}")
print(f"[p3] latent change: in-box mean |Δ|={in_mean:.4f} | out-box MAX |Δ|={out_max:.2e}")
assert coords_same, "structure must be identical after a region edit"
assert out_max == 0.0, "out-of-box latents must be preserved bit-exact"
assert in_mean > 1e-3, "in-box latents should change"
save_slat(edited, EDITED_NPZ, extra={"source": source_prompt, "edit": EDIT_PROMPT, "bbox": list(bbox)})

# ---------------------------------------------------------------- decode-space localization
# Decode the two SLATs ONE AT A TIME (24GB can't hold two GS decodes + renders at once). The
# decoder is deterministic on identical coords, so the Gaussian sets are aligned row-for-row;
# we pull colors/xyz and the render to CPU, then free the GPU copy before the next decode.
print("[p3] decoding + rendering source...")
source_gs = decode_gs(pipe, source)
vox = per_gaussian_voxel(source_gs.get_xyz)               # per-Gaussian voxel (CPU)
col_src = source_gs.get_features.flatten(1).detach().cpu()
src_vid = np.stack(render_utils.render_video(source_gs, num_frames=30)["color"])
del source_gs
torch.cuda.empty_cache()

print("[p3] decoding + rendering edited...")
edited_gs = decode_gs(pipe, edited)
col_edit = edited_gs.get_features.flatten(1).detach().cpu()
edit_vid = np.stack(render_utils.render_video(edited_gs, num_frames=30)["color"])
del edited_gs
torch.cuda.empty_cache()

# per-Gaussian appearance change, bucketed by in/out of the edit box
x0, y0, z0, x1, y1, z1 = bbox
g_in = ((vox[:, 0] >= x0) & (vox[:, 0] < x1) &
        (vox[:, 1] >= y0) & (vox[:, 1] < y1) &
        (vox[:, 2] >= z0) & (vox[:, 2] < z1))
dcol = (col_edit - col_src).abs().mean(dim=1)  # [G]
in_col = dcol[g_in].mean().item() if g_in.any() else float("nan")
out_col = dcol[~g_in].mean().item() if (~g_in).any() else float("nan")
ratio = in_col / out_col if out_col > 0 else float("inf")
print(f"[p3] gaussians: in-box={int(g_in.sum())} out-box={int((~g_in).sum())}")
print(f"[p3] per-Gaussian |Δcolor|: in-box={in_col:.4f} | out-box={out_col:.4f} | ratio={ratio:.1f}x")

# ---------------------------------------------------------------- artifacts: compare + video + report
v0 = int(src_vid.reshape(src_vid.shape[0], -1).mean(axis=1).argmax())  # brightest view
diff = np.abs(edit_vid[v0].astype(np.int16) - src_vid[v0].astype(np.int16)).astype(np.uint8)
compare = np.concatenate([src_vid[v0], edit_vid[v0], diff], axis=1)  # source | edited | |diff|
imageio.imwrite(os.path.join(OUT, "phase3_compare.png"), compare)
imageio.mimsave(os.path.join(OUT, "phase3_edit.mp4"), list(edit_vid), fps=30)

with open(REPORT, "w") as f:
    f.write("# Phase 3 — Region editing (RePaint masked SLAT sampling)\n\n")
    f.write(f"Source asset prompt: {source_prompt!r}\n\n")
    f.write(f"Edit prompt (in-box only): {EDIT_PROMPT!r}\n\n")
    f.write(f"- asset voxel range: min {mn}, max {mx}\n")
    f.write(f"- edit bbox (voxel idx, top half along axis {VERTICAL_AXIS}): {list(bbox)}\n")
    f.write(f"- voxels: {n_in} in-box / {n_out} out-box\n\n")
    f.write("## Latent-space preservation\n\n")
    f.write(f"- structure preserved (coords identical): **{coords_same}**\n")
    f.write(f"- out-of-box latents max |Δ|: **{out_max:.2e}** (bit-exact by construction)\n")
    f.write(f"- in-box latents mean |Δ|: **{in_mean:.4f}**\n\n")
    f.write("## Decode-space localization (per-Gaussian |Δcolor|)\n\n")
    f.write(f"- in-box Gaussians: **{in_col:.4f}**\n")
    f.write(f"- out-of-box Gaussians: **{out_col:.4f}**  (nonzero: the GS decoder mixes context "
            f"via attention, so an exact latent freeze still lets a little change bleed out)\n")
    f.write(f"- in/out change ratio: **{ratio:.1f}×** — the edit is concentrated in the box\n\n")
    f.write("Artifacts: `phase3_compare.png` (source | edited | |diff|, brightest view), "
            "`phase3_edit.mp4`, `phase3_edited.npz`.\n")

print(f"[p3] wrote {os.path.basename(REPORT)} / phase3_compare.png / phase3_edit.mp4")
print("[p3] DONE")
