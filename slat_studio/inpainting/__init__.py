"""inpainting: training-free 3D completion of a SLAT with a hole.

Phase 5. Fill a missing region (a voxel bounding box) of a SLAT — both geometry and
appearance — by RePaint on TRELLIS's two stages, with *no* training and *no* edits to the
``third_party/TRELLIS`` submodule:

  * Stage 1 (structure) regenerates the occupancy inside the hole while pinning the known
    surrounding structure (``DenseRepaintFlowSampler``).
  * Stage 2 (latents) regenerates the latents on the newly-filled voxels while keeping the
    surviving voxels' latents bit-exact (``RepaintFlowSampler``, the Phase-3 sampler).
"""
from .complete import (
    inpaint_slat,
    carve_hole,
    load_ss_encoder,
    coords_to_occupancy,
    occupancy_to_coords,
    DEFAULT_SS_ENCODER,
)

__all__ = [
    "inpaint_slat",
    "carve_hole",
    "load_ss_encoder",
    "coords_to_occupancy",
    "occupancy_to_coords",
    "DEFAULT_SS_ENCODER",
]
