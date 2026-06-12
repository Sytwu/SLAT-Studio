"""Phase 2: encoding-bridge round-trip fidelity report.

Validates the external-3DGS -> SLAT bridge using a TRELLIS-native asset as a *stand-in*
"external" input, which gives us ground-truth references the real external path lacks:

  native SLAT (from Phase 1 cache)
     -> decode to 3DGS                                  == the "external" input asset
     -> [BRIDGE] render multiview -> voxelize means -> DINOv2 feats -> SLAT VAE encode
     -> bridge SLAT
     -> decode to 3DGS -> render

We then report two things:
  * structure recovery:  IoU(bridge voxels, native voxels)                 [latent-space]
  * appearance fidelity: PSNR / SSIM(bridge render, native render), all views [render-space]

If round-trip fidelity is poor, every downstream edit inherits the error — so this is the
first experiment for the external-asset path. The bridge needs only the GS decoder + DINOv2
+ the SLAT VAE encoder (no flow models), so it fits one process on a 24GB card.
"""
import os
os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import numpy as np
import torch
import imageio
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils

from slat_studio.io import load_slat, save_slat, read_extra
from slat_studio import bridge

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
NATIVE_NPZ = os.path.join(OUT, "phase1_base.npz")
BRIDGE_NPZ = os.path.join(OUT, "phase2_bridge.npz")
REPORT = os.path.join(OUT, "phase2_report.md")

NVIEWS = 150          # match TRELLIS's render distribution for feature averaging
RES = 512
GRID = 64

assert os.path.exists(NATIVE_NPZ), f"missing {NATIVE_NPZ} — run examples/phase1_generate.py first"


def decode_gs(pipe, slat):
    return pipe.decode_slat(slat, ["gaussian"])["gaussian"][0]


def render_with(gaussian, extr, intr):
    out = render_utils.render_frames(
        gaussian, extr, intr,
        {"resolution": RES, "bg_color": (0, 0, 0)}, verbose=False)
    return np.stack(out["color"])  # [V,H,W,3] uint8


def feat_agreement(native_slat, bridge_slat, resolution=GRID):
    """Per-voxel latent comparison on the shared occupied voxels (cos sim + L2)."""
    def to_map(slat):
        c = slat.coords.detach().cpu().long()
        c = c[:, 1:] if c.shape[1] == 4 else c
        keys = (c[:, 0] * resolution * resolution + c[:, 1] * resolution + c[:, 2]).tolist()
        return dict(zip(keys, slat.feats.detach().cpu().float()))
    na, br = to_map(native_slat), to_map(bridge_slat)
    shared = sorted(set(na) & set(br))
    if not shared:
        return {"shared": 0, "cos": float("nan"), "l2": float("nan")}
    A = torch.stack([na[k] for k in shared])
    B = torch.stack([br[k] for k in shared])
    cos = torch.nn.functional.cosine_similarity(A, B, dim=1).mean().item()
    l2 = (A - B).norm(dim=1).mean().item()
    return {"shared": len(shared), "cos": cos, "l2": l2}


# ---------------------------------------------------------------- load pipeline (GS decoder only)
print("[p2] loading TrellisTextTo3DPipeline (only the GS decoder is needed)...")
pipe = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
pipe.cuda()
for name, model in pipe.models.items():          # park everything except the gaussian decoder
    if name != "slat_decoder_gs":
        model.cpu()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- native asset = "external" input
native = load_slat(NATIVE_NPZ, device="cuda")
native_prompt = read_extra(NATIVE_NPZ).get("prompt", "<unknown>")
print(f"[p2] native SLAT: {native.coords.shape[0]} voxels; prompt={native_prompt!r}")

native_gs = decode_gs(pipe, native)
colors_native, extr, intr = bridge.render_multiview(native_gs, nviews=NVIEWS, resolution=RES)
colors_native = np.stack(colors_native)          # [V,H,W,3] uint8
print(f"[p2] rendered {colors_native.shape[0]} ground-truth views of the input asset")

# BRIDGE step 1: occupancy from the Gaussian means (the new glue)
bridge_coords = bridge.gaussian_means_to_coords(native_gs.get_xyz, resolution=GRID)
iou = bridge.structure_iou(bridge_coords, native.coords, resolution=GRID)
print(f"[p2] voxelized means -> {iou['num_a']} voxels | native {iou['num_b']} | "
      f"structure IoU={iou['iou']:.4f}")
del native_gs
torch.cuda.empty_cache()

# BRIDGE step 2: DINOv2 features at those voxels
print("[p2] loading DINOv2 and extracting per-voxel features...")
dinov2 = bridge.load_dinov2(device="cuda")
feats = bridge.extract_voxel_features(
    colors_native, extr, intr, bridge_coords, dinov2,
    resolution=GRID, batch_size=16, device="cuda")
del dinov2
torch.cuda.empty_cache()
print(f"[p2] DINOv2 features: {tuple(feats.shape)}")

# BRIDGE step 3: SLAT VAE encode
print("[p2] loading SLAT VAE encoder and encoding...")
encoder = bridge.load_slat_encoder(device="cuda")
bridge_slat = bridge.encode_to_slat(feats, bridge_coords, encoder, device="cuda")
del encoder, feats
torch.cuda.empty_cache()
save_slat(bridge_slat, BRIDGE_NPZ, extra={"source": "bridge", "from": native_prompt,
                                          "nviews": NVIEWS})
print(f"[p2] bridge SLAT: {bridge_slat.coords.shape[0]} voxels, {bridge_slat.feats.shape[1]} ch "
      f"-> cached {os.path.basename(BRIDGE_NPZ)}")

fa = feat_agreement(native, bridge_slat, resolution=GRID)
print(f"[p2] latent agreement on {fa['shared']} shared voxels: cos={fa['cos']:.4f} l2={fa['l2']:.4f}")

# ---------------------------------------------------------------- decode bridge + render-space metrics
bridge_gs = decode_gs(pipe, bridge_slat)
colors_bridge = render_with(bridge_gs, extr, intr)
del bridge_gs
torch.cuda.empty_cache()

psnrs, ssims = [], []
for a, b in zip(colors_native, colors_bridge):
    psnrs.append(sk_psnr(a, b, data_range=255))
    ssims.append(sk_ssim(a, b, channel_axis=2, data_range=255))
psnr_mean, ssim_mean = float(np.mean(psnrs)), float(np.mean(ssims))
print(f"[p2] render-space fidelity over {len(psnrs)} views: "
      f"PSNR={psnr_mean:.2f} dB | SSIM={ssim_mean:.4f}")

# ---------------------------------------------------------------- artifacts: side-by-side + video + report
# pick the best-lit view so the side-by-side is legible (not a dark back-angle)
v0 = int(colors_native.reshape(colors_native.shape[0], -1).mean(axis=1).argmax())
compare = np.concatenate([colors_native[v0], colors_bridge[v0]], axis=1)
imageio.imwrite(os.path.join(OUT, "phase2_compare.png"), compare)
imageio.mimsave(os.path.join(OUT, "phase2_bridge.mp4"), list(colors_bridge), fps=30)

with open(REPORT, "w") as f:
    f.write("# Phase 2 — Encoding-bridge round-trip fidelity report\n\n")
    f.write(f"Stand-in external input: TRELLIS-native asset (prompt: {native_prompt!r}).\n\n")
    f.write("Pipeline: native SLAT -> decode 3DGS -> **render -> voxelize means -> DINOv2 -> "
            "SLAT VAE encode** -> bridge SLAT -> decode 3DGS -> render.\n\n")
    f.write(f"- Views: {NVIEWS} @ {RES}px, grid {GRID}^3\n\n")
    f.write("## Structure recovery (latent-space)\n\n")
    f.write(f"- native voxels: **{iou['num_b']}**\n")
    f.write(f"- bridge voxels: **{iou['num_a']}**\n")
    f.write(f"- voxel IoU: **{iou['iou']:.4f}** ({iou['intersection']} shared / {iou['union']} union)\n\n")
    f.write(f"- latent agreement on shared voxels: cos **{fa['cos']:.4f}**, mean L2 **{fa['l2']:.4f}** "
            f"(note: native feats come from the *flow* model, bridge feats from the *encoder* — "
            f"two valid points in the same SLAT space, so exact match is not expected)\n\n")
    f.write("## Appearance fidelity (render-space, bridge-decode vs input render)\n\n")
    f.write(f"- mean PSNR: **{psnr_mean:.2f} dB**  (min {min(psnrs):.2f}, max {max(psnrs):.2f})\n")
    f.write(f"- mean SSIM: **{ssim_mean:.4f}**  (min {min(ssims):.4f}, max {max(ssims):.4f})\n\n")
    f.write("Artifacts: `phase2_compare.png` (view 0: input | bridge), `phase2_bridge.mp4`, "
            "`phase2_bridge.npz`.\n")

print(f"[p2] wrote {os.path.basename(REPORT)} / phase2_compare.png / phase2_bridge.mp4")
print("[p2] DONE")
