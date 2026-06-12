"""morph: SLAT interpolation & morphing between two assets.

Hard part: the two SLATs have different active-voxel sets {p_i} (different L, positions),
so structure alignment / correspondence is needed before interpolating positions + latents.

TODO(Phase 4): structure alignment + latent interpolation; render t in {0,.25,.5,.75,1}.
Reference: MorphAny3D (arXiv:2601.00204).
"""
