"""Occupancy voxelization for the encoding bridge.

TRELLIS's own ``dataset_toolkits/voxelize.py`` derives 64^3 occupancy from a **mesh**
(open3d ``create_from_triangle_mesh_within_bounds``). An external 3DGS has no mesh, so we
derive occupancy directly from the Gaussian means instead — this is the one genuinely new
piece of glue in the bridge.

Coordinate frame (must match TRELLIS exactly):
- The SLAT grid is 64^3. Voxel ``index`` <-> center mapping used everywhere in TRELLIS is
  ``center = (index + 0.5) / 64 - 0.5`` and inversely ``index = floor((center + 0.5) * 64)``.
- TRELLIS Gaussians use ``aabb = [-0.5,-0.5,-0.5, 1,1,1]`` so ``get_xyz`` already lives in
  ``[-0.5, 0.5]`` and clusters around the voxel centers it was decoded from. Flooring the
  means therefore recovers the originating voxel indices.
"""
import numpy as np


def gaussian_means_to_coords(xyz, resolution=64):
    """Map Gaussian means in ``[-0.5, 0.5]`` to unique occupied voxel indices.

    Args:
        xyz: ``[N, 3]`` Gaussian means (torch.Tensor or np.ndarray), TRELLIS frame.
        resolution: grid resolution (64 for TRELLIS SLAT).

    Returns:
        ``[M, 3]`` int64 torch.Tensor of unique voxel indices in ``[0, resolution)``,
        sorted lexicographically (so the ordering is deterministic).
    """
    import torch

    if isinstance(xyz, np.ndarray):
        xyz = torch.from_numpy(xyz)
    xyz = xyz.detach().float().cpu()

    idx = torch.floor((xyz + 0.5) * resolution).long()
    idx = idx.clamp(0, resolution - 1)

    uniq = torch.unique(idx, dim=0)  # [M,3], sorted
    return uniq


def structure_iou(coords_a, coords_b, resolution=64):
    """Voxel-set IoU between two ``[*, 3]`` index tensors (intersection-over-union).

    Used to score how well bridge-recovered occupancy matches a reference structure.
    Accepts coords with or without a leading batch column.
    """
    import torch

    def _to_keys(c):
        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c)
        c = c.detach().long().cpu()
        if c.shape[1] == 4:  # drop batch column
            c = c[:, 1:]
        # encode (x,y,z) into a single int key
        return c[:, 0] * resolution * resolution + c[:, 1] * resolution + c[:, 2]

    ka = set(_to_keys(coords_a).tolist())
    kb = set(_to_keys(coords_b).tolist())
    inter = len(ka & kb)
    union = len(ka | kb)
    return {
        "iou": inter / union if union else 0.0,
        "intersection": inter,
        "union": union,
        "num_a": len(ka),
        "num_b": len(kb),
    }
