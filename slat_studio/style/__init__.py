"""style: style transfer & material/appearance alteration.

Approach: freeze the SLAT structure {p_i} and re-run the stage-2 flow conditioned on a new
text prompt, so coarse geometry is preserved while appearance (z_i) changes.

Scope note:
    - appearance restyle  -> easy, Phase 1 (TRELLIS detail-variation extension).
    - true PBR/material fields -> needs an extra trained decoder (SLAT-Phys-style), Phase 5.

TODO(Phase 1): native-asset restyle. Reference: SLAT-Phys (arXiv:2603.23973).
"""
