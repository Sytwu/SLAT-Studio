"""io: serialization for SLAT and 3D assets.

- SLAT save/load as .npz (coords {p_i} + feats {z_i}).
- 3DGS load/save (.ply), mesh load, GLB/OBJ export (reuse TRELLIS postprocessing utils).

TODO(Phase 1): SLAT .npz round-trip; .ply load/save.
"""
