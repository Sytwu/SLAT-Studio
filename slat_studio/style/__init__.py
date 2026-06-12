"""style: style transfer & material/appearance alteration.

Approach: freeze the SLAT structure {p_i} and re-run the stage-2 flow conditioned on a new
text prompt, so coarse geometry is preserved while appearance (z_i) changes.

Scope note:
    - appearance restyle  -> easy, Phase 1 (TRELLIS detail-variation extension).
    - true PBR/material fields -> needs an extra trained decoder (SLAT-Phys-style), Phase 5.

Status: native-asset restyle done (Phase 1). True PBR fields still TODO (Phase 5).
Reference: SLAT-Phys (arXiv:2603.23973).
"""
from .restyle import restyle

__all__ = ["restyle"]
