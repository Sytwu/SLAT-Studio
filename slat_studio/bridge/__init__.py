"""bridge: bring an existing 3D asset into SLAT space.

Pipeline (the one genuinely new, load-bearing component):
    1. render.py    -- multiview render of input 3DGS (reuse TRELLIS gaussian renderer)
    2. voxelize.py  -- voxel occupancy from Gaussian means/density -> active voxels (64^3)
    3. encode.py    -- DINOv2 features per voxel + frozen SLAT VAE encode -> z = {(z_i, p_i)}

For TRELLIS-native assets, skip this and reuse the SLAT produced at generation time.

TODO(Phase 2): implement. First experiment = encode -> decode round-trip fidelity report.
"""
