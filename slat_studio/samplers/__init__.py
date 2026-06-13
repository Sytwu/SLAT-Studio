"""samplers: custom flow samplers built by subclassing TRELLIS samplers.

This is how we change sampling behaviour (RePaint region edit, structure-conditioned
stage-2) WITHOUT editing the third_party/TRELLIS submodule.
"""
from .repaint import RepaintFlowSampler, DenseRepaintFlowSampler, bbox_mask
