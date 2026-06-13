"""morph: SLAT interpolation & morphing between two assets.

Hard part: the two SLATs have different active-voxel sets {p_i} (different L, positions),
so structure correspondence is needed before interpolating positions + latents. We solve
it with a structure union + a stable per-voxel dissolve schedule (see :mod:`.interpolate`):
shared voxels lerp their latents, A-only voxels dissolve out and B-only voxels dissolve in
as t goes 0->1, with exact A/B endpoints. Every intermediate is a valid, decodable SLAT.

Pure composition of TRELLIS public methods + ``trellis.modules.sparse`` — the vendored
submodule is not modified. Reference: MorphAny3D (arXiv:2601.00204).
"""
from .interpolate import SlatMorpher, morph_sequence, harmonize_slat

__all__ = ["SlatMorpher", "morph_sequence", "harmonize_slat"]
