"""io: serialization for SLAT and 3D assets.

- SLAT save/load as .npz (coords {p_i} + feats {z_i}).  [Phase 1, done]
- 3DGS load/save (.ply), mesh load, GLB/OBJ export (reuse TRELLIS postprocessing utils).

TODO: .ply load/save (Phase 2 bridge).
"""
from .slat_io import save_slat, load_slat, read_extra

__all__ = ["save_slat", "load_slat", "read_extra"]
