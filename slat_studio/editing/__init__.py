"""editing: region / local 3D editing in SLAT space.

Approach: RePaint-style masked two-stage sampling within a bounding box, implemented by
subclassing TRELLIS samplers (see slat_studio.samplers). Requires a source SLAT.

TODO(Phase 3): implement region edit. Verify outside-bbox voxels stay unchanged.
References: Easy3E (arXiv:2602.21499), InpaintSLat (arXiv:2605.00664).
"""
