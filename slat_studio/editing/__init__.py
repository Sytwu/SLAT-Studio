"""editing: region / local 3D editing in SLAT space.

Approach: RePaint-style masked two-stage sampling within a bounding box, implemented by
subclassing TRELLIS samplers (see slat_studio.samplers). Requires a source SLAT.

References: Easy3E (arXiv:2602.21499), InpaintSLat (arXiv:2605.00664).
"""
from .region_edit import edit_region, normalized_to_voxel_bbox
