"""bridge: bring an existing 3D asset into SLAT space.

Pipeline (the one genuinely new, load-bearing component):
    1. render.py    -- multiview render of input 3DGS (reuse TRELLIS gaussian renderer)
    2. voxelize.py  -- voxel occupancy from Gaussian means -> active voxels (64^3)
    3. encode.py    -- DINOv2 features per voxel + frozen SLAT VAE encode -> z = {(z_i, p_i)}

For TRELLIS-native assets, skip this and reuse the SLAT produced at generation time.
First experiment (Phase 2): encode -> decode round-trip fidelity report.
"""
from .render import render_multiview
from .voxelize import gaussian_means_to_coords, structure_iou
from .encode import (
    load_dinov2,
    load_slat_encoder,
    extract_voxel_features,
    encode_to_slat,
    DEFAULT_ENCODER,
    DEFAULT_DINOV2,
)


def gaussian_to_slat(gaussian, encoder, dinov2, nviews=150, resolution=512,
                     grid=64, batch_size=16, device="cuda"):
    """Full bridge: external 3DGS -> SLAT.

    Renders the Gaussian, derives 64^3 occupancy from its means, lifts DINOv2 features onto
    those voxels, and encodes them with the frozen SLAT VAE.

    Returns:
        ``(slat, coords, views)`` — the SLAT SparseTensor, the recovered ``[M,3]`` voxel
        indices, and ``views = (colors, extrinsics, intrinsics)`` (kept for re-rendering /
        metrics so callers don't re-render).
    """
    colors, extrinsics, intrinsics = render_multiview(
        gaussian, nviews=nviews, resolution=resolution)
    coords = gaussian_means_to_coords(gaussian.get_xyz, resolution=grid)
    feats = extract_voxel_features(
        colors, extrinsics, intrinsics, coords, dinov2,
        resolution=grid, batch_size=batch_size, device=device)
    slat = encode_to_slat(feats, coords, encoder, device=device)
    return slat, coords, (colors, extrinsics, intrinsics)
