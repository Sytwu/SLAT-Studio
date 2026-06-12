"""pipelines: high-level, user-facing entry points.

Each pipeline wires bridge/encoding + a task (edit/style/morph) + decoding into one call.
Keeps a backbone abstraction so TRELLIS vs TRELLIS.2 can be swapped behind one interface.

Phase 1: text_to_slat (text-to-3D that also returns the intermediate SLAT).
TODO: add EditPipeline, RestylePipeline, MorphPipeline as the tasks land.
"""
from .generate import text_to_slat

__all__ = ["text_to_slat"]
