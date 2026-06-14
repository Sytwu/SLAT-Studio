"""Re-run the region edit on a left/right half so the change is visible head-on.

The page's turntable starts with the camera facing the asset's +X face, so the viewer's
left<->right span is the Y axis (Z is vertical). The original edit cut a half along X — the
whole front-facing half — so head-on the entire visible face was lava with no clean wood next
to it. Cutting along Y instead splits the front face left/right (lava vs. original wood), which
reads clearly both head-on and as the chest rotates.

Overwrites outputs/my_asset_edited.{npz,ply} and the green region preview my_asset_region.ply.
Run with the trellis env + PYTHONPATH=third_party/TRELLIS (see project-page/README.md).
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import torch
from trellis.pipelines import TrellisTextTo3DPipeline

from slat_studio.io import load_slat, save_slat, read_extra
from slat_studio.editing import edit_region

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
SRC_NPZ = os.path.join(OUT, "my_asset.npz")
EDITED_NPZ = os.path.join(OUT, "my_asset_edited.npz")

EDIT_PROMPT = "molten glowing lava and burning embers"
SEED = 7
GRID = 64
SIDE_AXIS = 1   # Y = the viewer's left/right span at the turntable's start (Z=2 is vertical)

# region-preview tint (matches app.py)
SH_C0 = 0.28209479177387814
EDIT_TINT = (0.15, 0.85, 0.25)
TINT_ALPHA = 0.55

print("[edit] loading TrellisTextTo3DPipeline...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
for n in ("sparse_structure_flow_model", "slat_decoder_mesh", "slat_decoder_rf"):
    pipe.models[n].cpu()
torch.cuda.empty_cache()

source = load_slat(SRC_NPZ, device="cuda")
coords = source.coords[:, 1:].long()
mn = coords.min(0).values.tolist()
mx = coords.max(0).values.tolist()
mid = (mn[SIDE_AXIS] + mx[SIDE_AXIS] + 1) // 2
# full extent on every axis, then keep only the far half along the left/right axis
lo = [mn[0], mn[1], mn[2]]
hi = [mx[0] + 1, mx[1] + 1, mx[2] + 1]
lo[SIDE_AXIS] = mid
bbox = (lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
print(f"[edit] voxel range min={mn} max={mx}; side-half edit bbox (axis {SIDE_AXIS})={bbox}")

print(f"[edit] editing in-box region with prompt: {EDIT_PROMPT!r}")
edited, mask, _ = edit_region(pipe, source, bbox, EDIT_PROMPT, seed=SEED, formats=())
assert torch.equal(edited.coords, source.coords), "structure must be identical after a region edit"
save_slat(edited, EDITED_NPZ,
          extra={"source": read_extra(SRC_NPZ).get("prompt", "<unknown>"),
                 "edit": EDIT_PROMPT, "bbox": list(bbox)})

# stage-2 done; park the flow model + text encoder so the two decodes have the whole card
pipe.models["slat_flow_model"].cpu()
pipe.text_cond_model["model"].cpu()
torch.cuda.empty_cache()


def voxel_of(xyz):
    return torch.floor((xyz.detach().cpu() + 0.5) * GRID).long()


def in_box(vox):
    x0, y0, z0, x1, y1, z1 = bbox
    return ((vox[:, 0] >= x0) & (vox[:, 0] < x1) &
            (vox[:, 1] >= y0) & (vox[:, 1] < y1) &
            (vox[:, 2] >= z0) & (vox[:, 2] < z1))


print("[edit] decoding edited splat -> my_asset_edited.ply")
egs = pipe.decode_slat(edited, ["gaussian"])["gaussian"][0]
egs.save_ply(os.path.join(OUT, "my_asset_edited.ply"))
del egs
torch.cuda.empty_cache()

print("[edit] decoding source + green region preview -> my_asset_region.ply")
sgs = pipe.decode_slat(source, ["gaussian"])["gaussian"][0]
inb = in_box(voxel_of(sgs.get_xyz))
fdc = sgs._features_dc.clone()
rgb = 0.5 + SH_C0 * fdc[inb, 0, :]
rgb = (1 - TINT_ALPHA) * rgb + TINT_ALPHA * torch.tensor(EDIT_TINT, dtype=rgb.dtype, device=rgb.device)
fdc[inb, 0, :] = (rgb - 0.5) / SH_C0
sgs._features_dc = fdc
sgs.save_ply(os.path.join(OUT, "my_asset_region.ply"))
print("[edit] done")
